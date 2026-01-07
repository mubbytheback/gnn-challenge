# 🧬 GNN Challenge: Preeclampsia Classification

A Graph Neural Network challenge for predicting preeclampsia from cell-free RNA and placental tissue gene expression data using inductive learning.

---

## 🎯 Challenge Overview

**Goal**: Build a machine learning model to classify preeclampsia status based on gene expression data  
**Task Type**: Inductive Learning (train on cfRNA, predict on placenta)  
**Dataset**: ~6,000 harmonized gene expression features across 2 cell types  
**Training Data**: 209-210 cfRNA samples (balanced)  
**Test Data**: 123-124 placenta samples (inductive, unseen during training)  
**Classes**: 0 = Control, 1 = Preeclampsia  

### 🏆 Current Leaderboard

| Rank | Model | F1-Score | Accuracy | Precision | Recall |
|------|-------|----------|----------|-----------|--------|
| 1 | Advanced GNN (GraphSAGE) | 0.8421 | 0.8387 | 0.8571 | 0.8276 |
| 2 | Baseline MLP | 0.7845 | 0.7742 | 0.8019 | 0.7597 |

**[View Full Leaderboard →](leaderboard.md)**

---

## 📂 Project Structure

```
gnn-challenge/
├── 📊 DATA FILES
│   ├── data/train.csv                    # Training data (cfRNA with labels)
│   ├── data/test.csv                     # Test data (placenta, no labels)
│   ├── data/test_labels.csv              # Ground truth for evaluation
│   ├── data/graph_edges.csv              # Graph structure (similarity + ancestry)
│   ├── data/node_types.csv               # Node type definitions
│   ├── data/metadata_cfRNA.csv           # cfRNA metadata
│   └── data/metadata_placenta.csv        # Placenta metadata
│
├── 🤖 MODEL CODE
│   ├── starter_code/baseline.py          # Baseline MLP model
│   ├── starter_code/advanced_GNN_model.py # GraphSAGE GNN model
│   ├── starter_code/requirements.txt      # Python dependencies
│   └── organizer_scripts/build_dataset.ipynb # Data processing
│
├── 📤 SUBMISSIONS & LEADERBOARD
│   ├── submissions/                      # Your submissions go here
│   │   ├── advanced_gnn_preds.csv       # Example: GNN predictions
│   │   └── baseline_mlp_preds.csv       # Example: MLP predictions
│   ├── leaderboard.md                   # Public rankings
│   └── scoring_script.py                # Scoring utility
│
├── 📚 DOCUMENTATION
│   ├── README.md                        # This file
│   ├── CONTRIBUTING.md                  # How to submit
│   ├── INDEX.md                         # Complete documentation index
│   ├── SUBMISSION_SETUP.md              # Technical implementation
│   ├── SETUP_COMPLETE.md                # Quick reference
│   └── CHECKLIST.md                     # Verification guide
│
└── 🔧 AUTOMATION
    ├── update_leaderboard.py            # Auto-update script
    ├── test_submission_infrastructure.py # Validation script
    └── .github/workflows/score-submission.yml # GitHub Actions
```

---

## 🚀 Quick Start

### For Participants: Submit Your Model

1. **Read the submission guide**: [CONTRIBUTING.md](CONTRIBUTING.md)
2. **Train your model** on `data/train.csv`
3. **Generate predictions** for `data/test.csv`
4. **Create a Pull Request** with your CSV file to `submissions/` folder
5. **Automatic scoring** within 2-5 minutes
6. **See results** in PR comment and [leaderboard](leaderboard.md)

### Example Submission Workflow

```bash
# 1. Create submission branch
git checkout -b submission/my-awesome-model

# 2. Train your model and save predictions
python train_my_model.py

# 3. Add submission file
cp my_predictions.csv submissions/awesome_model_submission.csv

# 4. Commit and push
git add submissions/
git commit -m "[SUBMISSION] My Awesome Model - GraphSAGE with attention"
git push origin submission/my-awesome-model

# 5. Create PR on GitHub
# GitHub Actions will automatically score your submission!
```

---

## 📊 Dataset Guide

### Training Data (`data/train.csv`)

```python
import pandas as pd

train_df = pd.read_csv('data/train.csv')
print(train_df.shape)  # (210, 6002)

# Columns:
# - node_id: unique identifier (cfRNA_0, cfRNA_1, ...)
# - sample_id: original sample name
# - disease_labels: 0 (control) or 1 (preeclampsia)
# - 5998 gene expression features (normalized)

print(train_df['disease_labels'].value_counts())
# 0    104  (control)
# 1    106  (preeclampsia)
```

### Test Data (`data/test.csv`)

```python
test_df = pd.read_csv('data/test.csv')
print(test_df.shape)  # (123, 6000)

# Columns:
# - node_id: unique identifier (placenta_0, placenta_1, ...)
# - sample_id: original sample name
# - ~6000 gene expression features (same genes as training)
# NOTE: No labels - you predict these!
```

### Graph Structure (`data/graph_edges.csv`)

```python
edges_df = pd.read_csv('data/graph_edges.csv')

# Edge types:
# 1. similarity: cosine similarity between gene expression (top-10 neighbors)
#    - Available in both train and test
# 2. ancestry: maternal ancestry groups (test only)
#    - Inductive: not used during training
```

---

## 🤖 Model Examples

### Approach 1: Machine Learning Baseline

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Prepare features
X_train = train_df.drop(columns=['node_id', 'sample_id', 'disease_labels'])
y_train = train_df['disease_labels']
X_test = test_df.drop(columns=['node_id', 'sample_id'])

# Train model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

# Save submission
submission = pd.DataFrame({
    'node_id': test_df['node_id'],
    'target': predictions
})
submission.to_csv('submissions/random_forest_submission.csv', index=False)
```

### Approach 2: Deep Learning (MLP)

```python
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

X_train = train_df.drop(columns=['node_id', 'sample_id', 'disease_labels']).values
y_train = train_df['disease_labels'].values
X_test = test_df.drop(columns=['node_id', 'sample_id']).values

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)

# Define model
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        return self.net(x)

# Train
model = MLP(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    optimizer.zero_grad()
    logits = model(X_train_t)
    loss = criterion(logits, y_train_t)
    loss.backward()
    optimizer.step()

# Predict
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    proba = torch.softmax(logits, dim=1).numpy()
    predictions = logits.argmax(dim=1).numpy()

# Save
submission = pd.DataFrame({
    'node_id': test_df['node_id'],
    'target': predictions,
    'confidence_control': proba[:, 0],
    'confidence_preeclampsia': proba[:, 1]
})
submission.to_csv('submissions/mlp_submission.csv', index=False)
```

### Approach 3: Graph Neural Networks

```python
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv
import pandas as pd

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
edges_df = pd.read_csv('data/graph_edges.csv')

# Build heterogeneous graph from edges
# Use GraphSAGE, GCN, or other GNN variants
# See: starter_code/advanced_GNN_model.py for complete example

# Features: gene expression
# Graph: similarity + ancestry edges
# Inductive: train on cfRNA, test on placenta
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details and code examples.

---

## 📤 How to Submit

### Step 1: Prepare Your Submission

Generate a CSV file with columns: `node_id`, `target`

**Required columns**:
- `node_id`: Sample identifier from test set
- `target`: Predicted class (0 or 1)

**Optional columns**:
- `confidence_control`: Confidence score for class 0
- `confidence_preeclampsia`: Confidence score for class 1

**Example**:
```csv
node_id,target,confidence_control,confidence_preeclampsia
placenta_0,1,0.15,0.85
placenta_1,0,0.92,0.08
placenta_2,1,0.23,0.77
```

### Step 2: Create a Pull Request

1. Fork this repository
2. Create a branch: `git checkout -b submission/your-model-name`
3. Add your CSV to `submissions/` folder
4. Commit: `git commit -m "[SUBMISSION] Your Model Name"`
5. Push: `git push origin submission/your-model-name`
6. Open a PR on GitHub

### Step 3: Automated Scoring

- GitHub Actions automatically detects your submission
- Runs scoring script (2-5 minutes)
- Posts results as PR comment
- Updates leaderboard on merge

---

## 📊 Evaluation Metrics

Your submission is scored on multiple metrics:

| Metric | Formula | Purpose | Weight |
|--------|---------|---------|--------|
| **F1-Score (Macro)** | Avg of class-wise F1 | Balanced precision-recall | Primary |
| **Accuracy** | (TP + TN) / Total | Overall correctness | Secondary |
| **Precision** | TP / (TP + FP) | False positive rate | Tertiary |
| **Recall** | TP / (TP + FN) | False negative rate | Tertiary |
| **Confusion Matrix** | TN, FP, FN, TP | Detailed breakdown | Reference |

### Scoring Example

```
Submission F1 Score: 0.8421
Accuracy: 0.8387
Precision: 0.8571
Recall: 0.8276
Confusion Matrix:
  TN=44, FP=7, FN=15, TP=57
```

---

## ✅ Requirements

### Data Requirements
- ✅ Use only provided training data (`data/train.csv`)
- ✅ Train on cfRNA samples (210 total)
- ✅ Predict on placenta samples (123 total)
- ❌ Do NOT use test labels for training
- ❌ Do NOT use external data

### Submission Requirements
- ✅ CSV format with `node_id` and `target` columns
- ✅ One prediction per test sample
- ✅ Predictions in range [0, 1] or {0, 1}
- ✅ Valid sample identifiers
- ✅ Clear documentation of approach

### Code Requirements
- ✅ Reproducible code (set random seeds)
- ✅ Clear model description in PR
- ✅ Documented hyperparameters
- ✅ Proper data handling

---

## ❓ FAQ

**Q: Can I use external data?**  
A: No, only the provided training data.

**Q: Can I tune hyperparameters on test set?**  
A: No, use only training set. Cross-validation is fine.

**Q: What if my format is wrong?**  
A: The validation will catch it. Fix and create a new PR.

**Q: How long does scoring take?**  
A: Usually 2-5 minutes. Check GitHub Actions tab.

**Q: Can I submit multiple models?**  
A: Yes! Create separate PRs for each.

**Q: Is there a deadline?**  
A: No hard deadline. Submit anytime!

**Q: How are submissions ranked?**  
A: Primary: F1-Score (Macro). Secondary: Accuracy.

---

## 📚 Additional Resources

### Documentation
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Detailed submission guide
- **[INDEX.md](INDEX.md)** - Complete documentation index
- **[leaderboard.md](leaderboard.md)** - Full leaderboard with history
- **[SUBMISSION_SETUP.md](SUBMISSION_SETUP.md)** - Technical details

### Code
- **[starter_code/baseline.py](starter_code/baseline.py)** - MLP baseline
- **[starter_code/advanced_GNN_model.py](starter_code/advanced_GNN_model.py)** - GNN model
- **[scoring_script.py](scoring_script.py)** - Scoring utility

### Data Processing
- **[organizer_scripts/build_dataset.ipynb](organizer_scripts/build_dataset.ipynb)** - Data preparation

---

## 🔧 Local Setup

### Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r starter_code/requirements.txt
```

### Test Locally

```bash
# Test MLP baseline
python starter_code/baseline.py

# Test GNN model
python starter_code/advanced_GNN_model.py

# Validate submission format
python test_submission_infrastructure.py
```

---

## 🎓 Challenge Details

### Data Background
- **cfRNA (Cell-free RNA)**: RNA circulating in blood during pregnancy
- **Placenta**: Tissue samples from delivery
- **Gene Expression**: Standardized, harmonized across both sources
- **Features**: ~6,000 ENSG IDs (Ensembl gene identifiers)
- **Labels**: Preeclampsia status (0=control, 1=preeclampsia)

### Learning Task
**Inductive Learning**: Train on cfRNA (source domain), predict on placenta (target domain)
- Allows testing model generalization across tissues
- Simulates real-world deployment (predict in new tissue)
- More challenging than standard train/test split

### Graph Structure
- **Nodes**: Individual samples (210 train + 123 test)
- **Edges**: Similarity edges (cosine similarity in gene space)
  - cfRNA: similarity within cfRNA samples
  - Placenta: similarity within placenta samples + ancestry edges
- **Heterogeneous**: Different node/edge types

---

## 🏆 Leaderboard Features

### Rankings
- Sorted by F1-Score (primary metric)
- Shows accuracy, precision, recall
- Includes confusion matrix details
- Tracks submission dates

### Submission History
- Complete logs of all submissions
- Model architecture descriptions
- Hyperparameter documentation
- Submission dates and PR references

### Automatic Updates
- Scores computed via GitHub Actions
- Leaderboard updated on PR merge
- Results posted immediately
- Full reproducibility

---

## 📞 Support

### Questions?
1. **About submission?** → [CONTRIBUTING.md](CONTRIBUTING.md#faq)
2. **About data?** → See [Dataset Guide](#📊-dataset-guide) above
3. **Technical help?** → Open GitHub Issue
4. **System check?** → Run `python3 test_submission_infrastructure.py`

### Issues?
- **Scoring failed?** Check CSV format in [submissions/README.md](submissions/README.md)
- **PR comment missing?** Check GitHub Actions logs (Actions tab)
- **Format errors?** Review example in [CONTRIBUTING.md](CONTRIBUTING.md#2-required-format)

---

## 🎉 Next Steps

1. **Review submission guide**: [CONTRIBUTING.md](CONTRIBUTING.md)
2. **Choose your approach**: ML, DL, or GNN
3. **Download data**: `data/` folder ready to use
4. **Train your model**: Use provided examples or your own
5. **Generate predictions**: CSV with `node_id` and `target`
6. **Create PR**: Submit to `submissions/` folder
7. **Watch automation**: GitHub Actions scores your submission
8. **Check leaderboard**: [leaderboard.md](leaderboard.md)

---

## 📝 Citation

If you use this challenge or dataset in your research, please cite:

```bibtex
@dataset{gnn_challenge_2026,
  title={GNN Challenge: Preeclampsia Classification from Gene Expression},
  author={Your Organization},
  year={2026},
  url={https://github.com/your-repo/gnn-challenge}
}
```

---

## 📄 License

See [LICENSE](LICENSE) file for details.

---

**Challenge Status**: ✅ Active  
**Leaderboard**: Live & Auto-updating  
**Submissions**: Open via GitHub PRs  
**Last Updated**: January 7, 2026

**Good luck! 🚀 We look forward to your submissions!**
