# Contributing Submissions to GNN Challenge

Welcome! We're excited to see your solutions to the preeclampsia classification challenge. Follow this guide to submit your model predictions.

## Quick Start

### 1. Prepare Your Submission

Train your model on the training data and generate predictions for the test set:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Extract features and labels
X_train = train_df.drop(columns=['node_id', 'sample_id', 'disease_labels'])
y_train = train_df['disease_labels']
X_test = test_df.drop(columns=['node_id', 'sample_id'])

# Train your model (example: MLP, GNN, etc.)
# model = train_your_model(X_train, y_train)

# Generate predictions
predictions = model.predict(X_test)

# Create submission file (probabilities or hard labels)
submission = pd.DataFrame({
    'id': test_df['node_id'],
    'y_pred': predictions  # probability in [0,1] or hard 0/1
})

submission.to_csv('predictions.csv', index=False)
```

### 2. Required Format

Your submission CSV must have:
- **Columns**: `id`, `y_pred`
- **Rows**: One row per test sample (123-124 rows)
- **Values**: `y_pred` can be a probability (0–1) or a hard label (0/1)

Example:
```csv
id,y_pred
placenta_0,0.85
placenta_1,0.08
...
```

### 3. Submit via Pull Request

1. **Fork** this repository
2. **Create a new branch**:
   ```bash
   git checkout -b submission/my-model-name
   ```

3. **Create submission folder** and add files:
   ```
   submissions/inbox/<team>/<run_id>/predictions.csv
   submissions/inbox/<team>/<run_id>/metadata.json
   ```

   Example metadata.json:
   ```json
   {
     "team": "my_team",
     "run_id": "run_001",
     "model_name": "My GNN v1",
     "model_type": "human"
   }
   ```

4. **Add the files**:
   ```bash
   git add submissions/inbox/<team>/<run_id>/predictions.csv
   git add submissions/inbox/<team>/<run_id>/metadata.json
   git commit -m "Add submission: My Model Name"
   git push origin submission/my-model-name
   ```

5. **Create a Pull Request** with details:
   - **Title**: `[SUBMISSION] Your Model Name`
   - **Description**: Include:
     - Model architecture/approach
     - Training methodology
     - Key hyperparameters
     - Any insights or findings

6. **Wait for automated scoring**:
   - GitHub Actions will automatically score your submission (in CI against the hidden test set)
   - Results posted as comment on your PR
   - If approved, leaderboard will be updated

## Model Guidelines

### Do's 
- Use only the provided training data (`train.csv`)
- Leverage gene expression features (harmonized across datasets)
- Use the graph structure (`graph_edges.csv`) for GNN approaches
- Implement proper train/test splits for development
- Document your approach clearly

### Don'ts 
- Don't use external data sources
- Don't share code/ideas from other participants (without permission)
- Don't submit duplicate models

## Evaluation Metrics

Your submission will be scored on:

| Metric | Weight | Purpose |
|--------|--------|---------|
| **F1-Score (Macro)** | Primary | Balanced precision-recall across both classes |
| **Accuracy** | Secondary | Overall correctness |
| **Precision** | Tertiary | False positive rate (important for disease detection) |
| **Recall** | Tertiary | False negative rate (important for disease detection) |

View the [leaderboard.md](leaderboard.md) for current rankings.

## Common Approaches


### Approach 1: Deep Learning MLP
```python
import torch.nn as nn

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
```

### Approach 2: Graph Neural Networks
```python
from torch_geometric.nn import GraphSAGE
from torch_geometric.data import HeteroData

# Build heterogeneous graph from graph_edges.csv
# Use GraphSAGE, GCN, or other GNN variants
# Leverage both similarity and ancestry edges
```

## FAQ

**Q: Can I use external pre-trained models?**  
A: No, you must train from scratch on the provided data only.

**Q: Can I tune hyperparameters on the test set?**  
A: No, use only the training set for model development. Use cross-validation if needed.

**Q: What if my submission has errors?**  
A: The GitHub Actions workflow will catch errors and comment on your PR. Fix and push again.

**Q: How long does scoring take?**  
A: Usually 2-5 minutes. Check the "Actions" tab for status.

**Q: Can I submit multiple models?**  
A: Yes! Create separate PRs for each model. Use descriptive names.

**Q: Is there a deadline?**  
A: No hard deadline, but submissions earlier may have more visibility.

## Need Help?

- **Questions about the data?** Check [README.md](README.md)
- **Issues with submission?** Open an issue on GitHub
- **Want feedback?** Post in Discussions (if enabled)

---

**Good luck! 🎯 We look forward to seeing your solutions!**
