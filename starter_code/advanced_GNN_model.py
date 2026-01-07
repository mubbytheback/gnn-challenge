"""
advanced_GNN_model.py

Advanced inductive GNN for cfRNA → placenta prediction

✔ similarity edges used during training
✔ similarity + ancestry edges only enabled at test time
✔ GraphSAGE + BatchNorm + Dropout
✔ inductive generalization
✔ NO label leakage
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero

# -----------------------------
# 1. Load data
# -----------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "../data")

train_df = pd.read_csv(os.path.join(DATA, "train.csv"))   # cfRNA
test_df  = pd.read_csv(os.path.join(DATA, "test.csv"))    # placenta
edges_df = pd.read_csv(os.path.join(DATA, "graph_edges.csv"))
node_df  = pd.read_csv(os.path.join(DATA, "node_types.csv"))

# -----------------------------
# 2. Node indexing
# -----------------------------
node_ids = node_df["node_id"].tolist()
node_map = {nid: i for i, nid in enumerate(node_ids)}
NUM_NODES = len(node_ids)

# -----------------------------
# 3. Graph construction (edge-split)
# -----------------------------
def build_graph(allowed_edge_types):
    data = HeteroData()
    data["node"].num_nodes = NUM_NODES

    for etype in allowed_edge_types:
        df = edges_df[edges_df.edge_type == etype]
        src = torch.tensor([node_map[i] for i in df.src], dtype=torch.long)
        dst = torch.tensor([node_map[i] for i in df.dst], dtype=torch.long)
        data["node", etype, "node"].edge_index = torch.stack([src, dst])

    return data

# TRAIN: similarity only
train_graph = build_graph(["similarity"])

# TEST: similarity + ancestry
test_graph = build_graph(["similarity", "ancestry"])

# -----------------------------
# 4. Node features + labels
# -----------------------------
feat_cols = [c for c in train_df.columns if c not in ["node_id", "target", "sample_id"]]

X = torch.zeros((NUM_NODES, len(feat_cols)))

train_idx = torch.tensor([node_map[i] for i in train_df.node_id], dtype=torch.long)
test_idx  = torch.tensor([node_map[i] for i in test_df.node_id], dtype=torch.long)

X[train_idx] = torch.tensor(train_df[feat_cols].values, dtype=torch.float)
X[test_idx]  = torch.tensor(test_df[feat_cols].values, dtype=torch.float)

train_graph["node"].x = X
test_graph["node"].x  = X

# labels (train only!)
y = torch.full((NUM_NODES,), -1, dtype=torch.long)
y[train_idx] = torch.tensor(train_df['target'].values, dtype=torch.long)

# -----------------------------
# 5. GraphSAGE model
# -----------------------------
class SAGEBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = SAGEConv(in_c, out_c)
        self.bn = nn.BatchNorm1d(out_c)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        return F.relu(x)

class GNN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.l1 = SAGEBlock(in_c, hid_c)
        self.l2 = SAGEBlock(hid_c, hid_c)
        self.cls = SAGEConv(hid_c, out_c)

    def forward(self, x, edge_index):
        x = self.l1(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.l2(x, edge_index)
        return self.cls(x, edge_index)

base_model = GNN(
    in_c=X.size(1),
    hid_c=128,
    out_c=len(train_df.target.unique())
)

# Heterogeneous conversion
model = to_hetero(base_model, train_graph.metadata(), aggr="mean")

# -----------------------------
# 6. Training setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
train_graph = train_graph.to(device)
test_graph  = test_graph.to(device)
y = y.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# -----------------------------
# 7. Training (FULL-BATCH, inductive-safe)
# -----------------------------
print("Starting training...")

for epoch in range(1, 31):
    model.train()
    optimizer.zero_grad()

    out = model(
        train_graph.x_dict,
        train_graph.edge_index_dict
    )["node"]

    loss = criterion(out[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")

# ========================================
# 8. Inductive testing (placenta)
# ========================================
print("\nGenerating test predictions (inductive on unseen placenta nodes)...")

model.eval()
with torch.no_grad():
    logits = model(
        test_graph.x_dict,
        test_graph.edge_index_dict
    )["node"]

# Hard predictions
preds = logits[test_idx].argmax(dim=1).cpu().numpy()

# Soft predictions (probabilities)
proba = torch.softmax(logits[test_idx], dim=1).cpu().numpy()

# ========================================
# 9. Save predictions (with confidence)
# ========================================
os.makedirs("submissions", exist_ok=True)

# Hard predictions (primary submission)
submission_hard = pd.DataFrame({
    "node_id": test_df.node_id,
    "target": preds
})
submission_hard.to_csv("submissions/advanced_gnn_preds.csv", index=False)

# Soft predictions (for analysis)
submission_soft = pd.DataFrame({
    "node_id": test_df.node_id,
    "target": preds,
    "confidence_class_0": proba[:, 0],
    "confidence_class_1": proba[:, 1]
})
submission_soft.to_csv("submissions/advanced_gnn_preds_with_confidence.csv", index=False)

print("✅ Hard predictions saved: submissions/advanced_gnn_preds.csv")
print("✅ Soft predictions saved: submissions/advanced_gnn_preds_with_confidence.csv")
print(f"   Total predictions: {len(preds)}")
print(f"   Class distribution: {np.bincount(preds)}")
print("\n✅ Predictions saved correctly (inductive, no label leakage)")

