# My personal advanced GNN model.
# This applies GNN theory, covers the DGL lectures 1.1–4.6, and tries to beat baseline.py.

"""
advanced_GNN_model.py

Advanced inductive GNN for cfRNA -> placenta prediction
Covers:
- DGL graph construction (heterogeneous)
- GraphSAGE layers
- Neighbor sampling for mini-batch training
- Inductive testing on unseen nodes
- Heterogeneous edges (similarity + ancestry)
"""

import dgl
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from dgl.nn import SAGEConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import os

# -----------------------------
# 1. Load data
# -----------------------------
data_dir = '../data'

train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
edges_df = pd.read_csv(os.path.join(data_dir, 'graph_edges.csv'))
node_types_df = pd.read_csv(os.path.join(data_dir, 'node_types.csv'))

# -----------------------------
# 2. Construct heterogeneous DGL graph
# -----------------------------
def build_hetero_graph(edges_df, node_types_df):
    """
    Build a DGL heterogeneous graph with:
    - 'similarity' edges
    - 'ancestry' edges
    """
    # Create mapping: node_id -> int
    all_nodes = node_types_df['node_id'].tolist()
    node_map = {nid: i for i, nid in enumerate(all_nodes)}

    # Hetero edge dictionary
    edge_dict = {}
    for edge_type in edges_df['edge_type'].unique():
        df = edges_df[edges_df['edge_type'] == edge_type]
        src = [node_map[i] for i in df['src']]
        dst = [node_map[i] for i in df['dst']]
        # DGL uses tuple: (src_type, edge_type, dst_type)
        src_type = dst_type = 'node'
        edge_dict[(src_type, edge_type, dst_type)] = (torch.tensor(src), torch.tensor(dst))

    # Build heterogeneous graph
    g = dgl.heterograph(edge_dict)
    print(g)
    return g, node_map

g, node_map = build_hetero_graph(edges_df, node_types_df)

# -----------------------------
# 3. Assign node features
# -----------------------------
def assign_node_features(df, node_map):
    feature_cols = [c for c in df.columns if c not in ['target','node_id','sample_id']]
    features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    # Map to graph nodes
    node_indices = [node_map[nid] for nid in df['node_id']]
    x = torch.zeros((len(node_map), features.shape[1]))
    x[node_indices] = features
    return x

train_feats = assign_node_features(train_df, node_map)
test_feats = assign_node_features(test_df, node_map)

# Targets
train_labels = torch.tensor(train_df['target'].values, dtype=torch.long)
train_nids = torch.tensor([node_map[nid] for nid in train_df['node_id']])

# -----------------------------
# 4. GraphSAGE model
# -----------------------------
# -----------------------------
# 4. GraphSAGE model with BatchNorm
# -----------------------------
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # input layer
        self.layers.append(SAGEConv(in_feats, hidden_feats, 'mean'))
        self.batch_norms.append(nn.BatchNorm1d(hidden_feats))

        # hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_feats, hidden_feats, 'mean'))
            self.batch_norms.append(nn.BatchNorm1d(hidden_feats))

        # output layer
        self.layers.append(SAGEConv(hidden_feats, out_feats, 'mean'))
        # output layer typically no batch norm

    def forward(self, g, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:  # not last layer
                h = self.batch_norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h


# -----------------------------
# 5. Neighbor sampling & dataloaders (mini-batch)
# -----------------------------
sampler = MultiLayerNeighborSampler([10, 10])  # 2-layer SAGE, 10 neighbors per layer
train_dataloader = NodeDataLoader(
    g,
    train_nids,
    sampler,
    batch_size=32,
    shuffle=True,
    drop_last=False,
    num_workers=0
)

# -----------------------------
# 6. Training setup
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_feats = train_feats.shape[1]
hidden_feats = 128
out_feats = len(train_labels.unique())
model = GraphSAGE(in_feats, hidden_feats, out_feats).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Move features and labels to device
train_feats = train_feats.to(device)
train_labels = train_labels.to(device)

# -----------------------------
# 7. Training loop
# -----------------------------
def train(model, dataloader, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(device) for b in blocks]
            batch_feats = train_feats[input_nodes]
            batch_labels = train_labels[output_nodes]

            optimizer.zero_grad()
            batch_pred = model(blocks, batch_feats)
            loss = loss_fn(batch_pred, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")

# -----------------------------
# 8. Inductive testing (placenta nodes)
# -----------------------------
def inductive_test(model, g, feats, test_df, node_map):
    model.eval()
    test_nids = torch.tensor([node_map[nid] for nid in test_df['node_id']])
    with torch.no_grad():
        logits = model(g.to(device), feats.to(device))
        preds = logits[test_nids].argmax(dim=1).cpu().numpy()
    return preds

# -----------------------------
# 9. Run training and testing
# -----------------------------
train(model, train_dataloader, epochs=20)
test_preds = inductive_test(model, g, torch.cat([train_feats, test_feats]), test_df, node_map)

# Optional: save predictions
submission = pd.DataFrame({'node_id': test_df['node_id'], 'target': test_preds})
submission.to_csv(os.path.join('submissions', 'advanced_gnn_preds.csv'), index=False)
print("✅ Advanced GNN predictions saved")
