"""
advanced_GNN_model.py

Advanced inductive GNN for cfRNA → placenta prediction

✔ similarity edges used during training
✔ similarity + ancestry edges only enabled at test time
✔ GraphSAGE + BatchNorm + Dropout
✔ inductive generalization
✔ NO label leakage
✔ Handles float/int target issues & class weighting
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
# DROP ROWS WITHOUT TARGETS
# -----------------------------
missing_targets = train_df['disease_labels'].isna().sum()
print(f"{missing_targets} missing values in target column")

if missing_targets > 0:
    print("Dropping rows without target labels from training set...")
    train_df = train_df.dropna(subset=['disease_labels'])
    print(f"Training set now has {len(train_df)} samples with valid labels")

# Identify target column (train uses 'disease_labels' or 'target')
target_col = 'disease_labels' if 'disease_labels' in train_df.columns else 'target'
has_test_labels = target_col in test_df.columns
# ========================================
# Display Target Distribution
# ========================================
print("\n" + "="*70)
print("  📊 TARGET FEATURE DISTRIBUTION")
print("="*70)

print("\n🔹 TRAINING DATA (cfRNA):")
train_counts = train_df['disease_labels'].value_counts().sort_index()
print(f"   Total samples: {len(train_df)}")
for target_val, count in train_counts.items():
    pct = (count / len(train_df)) * 100
    label = "control" if target_val == 0 else "preeclampsia"
    print(f"   Class {target_val} ({label}): {count} samples ({pct:.1f}%)")

if has_test_labels:
    print("\n🔹 TESTING DATA (Placenta):")
    test_counts = test_df[target_col].value_counts().sort_index()
    print(f"   Total samples: {len(test_df)}")
    for target_val, count in test_counts.items():
        pct = (count / len(test_df)) * 100
        label = "control" if target_val == 0 else "preeclampsia"
        print(f"   Class {target_val} ({label}): {count} samples ({pct:.1f}%)")
else:
    print("\n🔹 TESTING DATA (Placenta):")
    print(f"   Total samples: {len(test_df)}")
    print("   ⚠️  No labels (inductive task - labels hidden for evaluation)")

print("="*70 + "\n")

# -----------------------------
# 2. Node indexing
# -----------------------------
node_ids = node_df["node_id"].tolist()
node_map = {nid: i for i, nid in enumerate(node_ids)}
NUM_NODES = len(node_ids)

# -----------------------------
# 3. Graph construction
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

train_graph = build_graph(["similarity"])
test_graph  = build_graph(["similarity", "ancestry"])

# -----------------------------
# 4. Node features
# -----------------------------
# Only use columns that exist in both train and test datasets
train_cols = set(train_df.columns)
test_cols = set(test_df.columns)
shared_cols = train_cols.intersection(test_cols)
feat_cols = [c for c in shared_cols if c not in ["node_id", target_col, "sample_id"]]
feat_cols = sorted(feat_cols)  # For consistent ordering

X = torch.zeros((NUM_NODES, len(feat_cols)))

train_idx = torch.tensor([node_map[i] for i in train_df.node_id], dtype=torch.long)
test_idx  = torch.tensor([node_map[i] for i in test_df.node_id], dtype=torch.long)

X[train_idx] = torch.tensor(train_df[feat_cols].values, dtype=torch.float)
X[test_idx]  = torch.tensor(test_df[feat_cols].values, dtype=torch.float)

train_graph["node"].x = X
test_graph["node"].x  = X

# -----------------------------
# 5. Labels (train only)
# -----------------------------
y = -1 * np.ones(NUM_NODES, dtype=int)  # default for all nodes
y[train_idx] = train_df[target_col].values.astype(int)
y = torch.tensor(y, dtype=torch.long)
print(f"✅ Labels assigned. Train nodes: {len(train_idx)}, Total nodes: {NUM_NODES}")

# -----------------------------
# 6. GraphSAGE model
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

num_classes = len(train_df[target_col].unique())
base_model = GNN(X.size(1), hid_c=128, out_c=num_classes)
model = to_hetero(base_model, train_graph.metadata(), aggr="mean")

# -----------------------------
# 7. Training setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
train_graph = train_graph.to(device)
test_graph = test_graph.to(device)
y = y.to(device)

# Compute class weights safely
class_counts = train_df[target_col].value_counts()
weights = torch.tensor([
    class_counts.get(1, 0)/class_counts.sum(),
    class_counts.get(0, 0)/class_counts.sum()
], dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# 8. Training (FULL-BATCH, inductive-safe)
# -----------------------------
print("Starting training...")
for epoch in range(1, 31):
    model.train()
    optimizer.zero_grad()
    out = model(train_graph.x_dict, train_graph.edge_index_dict)["node"]
    loss = criterion(out[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")

# -----------------------------
# 9. Inductive testing (placenta)
# -----------------------------
print("\nGenerating inductive predictions for placenta nodes...")
model.eval()
with torch.no_grad():
    logits = model(test_graph.x_dict, test_graph.edge_index_dict)["node"]
    preds = logits[test_idx].argmax(dim=1).cpu().numpy()
    proba = torch.softmax(logits[test_idx], dim=1).cpu().numpy()

# Evaluate on training set (to verify model performance)
print("\n" + "="*70)
print("  📊 TRAINING SET EVALUATION METRICS")
print("="*70)
with torch.no_grad():
    train_logits = model(train_graph.x_dict, train_graph.edge_index_dict)["node"]
    train_preds = train_logits[train_idx].argmax(dim=1).cpu().numpy()
    train_proba = torch.softmax(train_logits[train_idx], dim=1).cpu().numpy()
    train_true = y[train_idx].cpu().numpy()

train_acc = accuracy_score(train_true, train_preds)
train_prec = precision_score(train_true, train_preds, zero_division=0)
train_rec = recall_score(train_true, train_preds, zero_division=0)
train_f1 = f1_score(train_true, train_preds, zero_division=0)
train_cm = confusion_matrix(train_true, train_preds)

print(f"\n  Accuracy:     {train_acc:.4f}")
print(f"  Precision:    {train_prec:.4f}")
print(f"  Recall:       {train_rec:.4f}")
print(f"  F1-Score:     {train_f1:.4f}")
print(f"\n  Confusion Matrix:")
print(f"     TN={train_cm[0,0]:3d}  FP={train_cm[0,1]:3d}")
print(f"     FN={train_cm[1,0]:3d}  TP={train_cm[1,1]:3d}")

# Test set prediction statistics
print("\n" + "="*70)
print("  🔮 TEST SET PREDICTIONS (INDUCTIVE - PLACENTA)")
print("="*70)
print("\n⚠️  NOTE: Test set is inductive (no ground truth labels available)")
print("   Predictions are from model trained on cfRNA (training set)\n")

pred_counts = np.bincount(preds, minlength=2)
print(f"📌 Predicted Labels for {len(preds)} Placenta Nodes:")
print(f"   Class 0 (control):       {pred_counts[0]:3d} nodes ({pred_counts[0]/len(preds)*100:.1f}%)")
print(f"   Class 1 (preeclampsia):  {pred_counts[1]:3d} nodes ({pred_counts[1]/len(preds)*100:.1f}%)")

print(f"\n📊 Prediction Confidence Analysis:")
max_conf = proba.max(axis=1)
print(f"   Mean max confidence: {max_conf.mean():.4f}")
print(f"   Min confidence:      {max_conf.min():.4f}")
print(f"   Max confidence:      {max_conf.max():.4f}")
print(f"   Std deviation:       {max_conf.std():.4f}")

# Count high confidence predictions
high_conf_mask = max_conf >= 0.9
print(f"\n   High confidence (≥0.90): {high_conf_mask.sum()} predictions ({high_conf_mask.sum()/len(preds)*100:.1f}%)")
med_conf_mask = (max_conf >= 0.7) & (max_conf < 0.9)
print(f"   Medium confidence (0.70-0.89): {med_conf_mask.sum()} predictions ({med_conf_mask.sum()/len(preds)*100:.1f}%)")
low_conf_mask = max_conf < 0.7
print(f"   Low confidence (<0.70): {low_conf_mask.sum()} predictions ({low_conf_mask.sum()/len(preds)*100:.1f}%)")

print("\n" + "="*70)

# -----------------------------
# 10. Save predictions
# -----------------------------
os.makedirs("submissions", exist_ok=True)

# Hard predictions
submission_hard = pd.DataFrame({
    "node_id": test_df.node_id,
    "target": preds
})
submission_hard.to_csv("submissions/advanced_gnn_preds.csv", index=False)

# Soft predictions with confidence
submission_soft = pd.DataFrame({
    "node_id": test_df.node_id,
    "target": preds,
    "confidence_control": proba[:, 0],
    "confidence_preeclampsia": proba[:, 1]
})
submission_soft.to_csv("submissions/advanced_gnn_preds_with_confidence.csv", index=False)

print("✅ Predictions saved successfully!")
print(f"   Hard: submissions/advanced_gnn_preds.csv")
print(f"   Soft: submissions/advanced_gnn_preds_with_confidence.csv")
print(f"   Total predictions: {len(preds)}")
