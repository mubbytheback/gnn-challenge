# Submissions Directory

This folder contains model prediction submissions for the **GNN Challenge**.

## 📤 How to Submit

**Full instructions**: See [../CONTRIBUTING.md](../CONTRIBUTING.md) or [../README.md](../README.md)

### Quick Steps
1. Train your model on `../data/train.csv`
2. Generate predictions for `../data/test.csv`
3. Save as CSV with columns: `node_id`, `target`
4. Create Pull Request adding your file to this folder
5. GitHub Actions automatically scores your submission

### File Format

**Requirements**:
- Format: CSV
- Required columns: `node_id`, `target`
- Optional columns: `confidence_control`, `confidence_preeclampsia`
- Rows: One per test sample (123-124 rows)
- File name: `{model_name}_submission.csv`

**Example**:
```csv
node_id,target,confidence_control,confidence_preeclampsia
placenta_0,1,0.15,0.85
placenta_1,0,0.92,0.08
placenta_2,1,0.23,0.77
```

## 📊 Current Leaderboard

| Rank | Model | F1-Score | Accuracy | Date |
|------|-------|----------|----------|------|
| 1 | Advanced GNN (GraphSAGE) | 0.8421 | 0.8387 | 2026-01-07 |
| 2 | Baseline MLP | 0.7845 | 0.7742 | 2026-01-07 |

**[View full leaderboard](../leaderboard.md)**

## 📋 Submission Checklist

Before submitting, verify:
- [ ] CSV file in correct format
- [ ] `node_id` column matches test samples
- [ ] `target` values are 0 or 1
- [ ] Row count = 123-124
- [ ] File saved to `submissions/` folder
- [ ] PR description includes model details

## ✅ Metrics Computed

Each submission is automatically evaluated on:
- **F1-Score (Macro)** - Primary ranking metric
- **Accuracy** - Overall correctness
- **Precision** - False positive rate
- **Recall** - False negative rate
- **Confusion Matrix** - Detailed breakdown

## 🚫 Important Rules

- ✅ Use only `../data/train.csv` for training
- ✅ Document your approach in PR description
- ❌ Do NOT use `../data/test_labels.csv` for training
- ❌ Do NOT use external data
- ❌ Do NOT share test labels

## 📚 Examples

### Example 1: Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

X_train = train_df.drop(columns=['node_id', 'sample_id', 'disease_labels'])
y_train = train_df['disease_labels']
X_test = test_df.drop(columns=['node_id', 'sample_id'])

scaler = StandardScaler()
model = RandomForestClassifier(n_estimators=100)
model.fit(scaler.fit_transform(X_train), y_train)
predictions = model.predict(scaler.transform(X_test))

submission = pd.DataFrame({
    'node_id': test_df['node_id'],
    'target': predictions
})
submission.to_csv('submissions/random_forest_submission.csv', index=False)
```

### Example 2: MLP with PyTorch
See [../CONTRIBUTING.md](../CONTRIBUTING.md#approach-2-deep-learning-mlp) for complete example

### Example 3: Graph Neural Network
See [../starter_code/advanced_GNN_model.py](../starter_code/advanced_GNN_model.py) for full implementation

## ❓ Help

- **How do I submit?** → [../CONTRIBUTING.md](../CONTRIBUTING.md)
- **What's the data format?** → [../README.md#-dataset-guide](../README.md#-dataset-guide)
- **Need code examples?** → [../CONTRIBUTING.md#common-approaches](../CONTRIBUTING.md#common-approaches)
- **System not working?** → Run `python3 ../test_submission_infrastructure.py`

---

**[← Back to README](../README.md)**
