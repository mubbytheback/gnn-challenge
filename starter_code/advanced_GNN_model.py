# My personal advanced GNN model.
# This applies GNN theory, covers the DGL lectures 1.1–4.6, and tries to beat baseline.py.

"""
advanced_GNN_model.py

Advanced inductive GNN for cfRNA -> placenta prediction
Covers:
- DGL heterogeneous graph construction
- GraphSAGE layers with BatchNorm & Dropout
- Neighbor sampling for mini-batch training
- Inductive testing on unseen nodes
- Heterogeneous edges (similarity + ancestry)
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from dgl.nn import SAGEConv, HeteroGraphConv

# -----------------------------
# 1. Load data
# -----------------------------
data_dir = '../data'
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
edges_df = pd.read_csv(os.path.join(data_dir, 'graph_edges.csv'))
node_types_df = pd.read_csv(os.path.join(data_dir, 'node_types.csv'))

# -----------------------------
# 2. Build heterogeneous DGL graph
# -----------------------------
def build_hetero_graph(edges_df, node_types_df):
    # Map node_id -> int index
    all_nodes = node_types_df['node_id'].tolist()
    node_map = {nid: i for i, nid in enumerate(all_nodes)}

    edge_dict = {}
    for edge_type in edges_df['edge_type'].unique():
        df = edges_df[edges_df['edge_type'] == edge_type]
        src = [node_map[i] for i in df['src']]
        dst = [node_map[i] for i in df['dst']]
        edge_dict[('node', edge_type, 'node')] = (torch.tensor(src), torch.tensor(dst))

    g = dgl.heterograph(edge_dict)
    g = dgl.add_self_loop(g)
    return g, node_map

g, node_map = build_hetero_graph(edges_df, node_types_df)

# -----------------------------
# 3. Assign node features
# -----------------------------
def assign_node_features(df, node_map):
    feature_cols = [c for c in df.columns if c not in ['target', 'node_id', 'sample_id']]
    features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    node_indices = [node_map[nid] for nid in df['node_id']]
    x = torch.zeros((len(node_map), features.shape[1]))
    x[node_indices] = features
    return x

train_feats = assign_node_features(train_df, node_map)
test_feats = assign_node_features(test_df, node_map)

train_labels = torch.tensor(train_df['target'].values, dtype=torch.long)
train_nids = torch.tensor([node_map[nid] for nid in train_df['node_id']])

# -----------------------------
# 4. Advanced GraphSAGE model
# -----------------------------
class AdvancedGraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Use HeteroGraphConv for heterogeneous edges
        self.layers.append(HeteroGraphConv({
            'similarity': SAGEConv(in_feats, hidden_feats, 'mean'),
            'ancestry': SAGEConv(in_feats, hidden_feats, 'mean')
        }, aggregate='mean'))
        self.batch_norms.append(nn.BatchNorm1d(hidden_feats))

        for _ in range(num_layers - 2):
            self.layers.append(HeteroGraphConv({
                'similarity': SAGEConv(hidden_feats, hidden_feats, 'mean'),
                'ancestry': SAGEConv(hidden_feats, hidden_feats, 'mean')
            }, aggregate='mean'))
            self.batch_norms.append(nn.BatchNorm1d(hidden_feats))

        self.layers.append(HeteroGraphConv({
            'similarity': SAGEConv(hidden_feats, out_feats, 'mean'),
            'ancestry': SAGEConv(hidden_feats, out_feats, 'mean')
        }, aggregate='mean'))

    def forward(self, g, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(g, {'node': h})['node']
            if l != self.num_layers - 1:
                h = self.batch_norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h

# -----------------------------
# 5. Neighbor sampling & dataloaders
# -----------------------------
sampler = MultiLayerNeighborSampler([10, 10])  # 2-layer GraphSAGE
train_dataloader = NodeDataLoader(
    g, train_nids, sampler,
    batch_size=32, shuffle=True, drop_last=False, num_workers=0
)

# -----------------------------
# 6. Training setup
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_feats = train_feats.shape[1]
hidden_feats = 128
out_feats = len(train_labels.unique())
model = AdvancedGraphSAGE(in_feats, hidden_feats, out_feats).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

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
            batch_pred = model(blocks[0], batch_feats)  # single block for hetero
            loss = loss_fn(batch_pred, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")

# -----------------------------
# 8. Inductive testing
# -----------------------------
def inductive_test(model, g, feats, test_df, node_map):
    model.eval()
    test_nids = torch.tensor([node_map[nid] for nid in test_df['node_id']]).to(device)
    with torch.no_grad():
        logits = model(g.to(device), feats.to(device))
        preds = logits[test_nids].argmax(dim=1).cpu().numpy()
    return preds

# -----------------------------
# 9. Run training & testing
# -----------------------------
train(model, train_dataloader, epochs=20)

all_feats = torch.cat([train_feats, test_feats]).to(device)
test_preds = inductive_test(model, g, all_feats, test_df, node_map)

# Save predictions
submission_dir = 'submissions'
os.makedirs(submission_dir, exist_ok=True)
submission = pd.DataFrame({'node_id': test_df['node_id'], 'target': test_preds})
submission.to_csv(os.path.join(submission_dir, 'advanced_gnn_preds.csv'), index=False)
print("✅ Advanced GNN predictions saved")
