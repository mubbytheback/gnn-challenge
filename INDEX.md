# 🎯 GNN Challenge - Complete Implementation Guide

Welcome to the complete submission and leaderboard system for the GNN Challenge! This guide helps you understand everything that was built.

## 📚 Documentation Index

### 🚀 Getting Started
- **[SETUP_COMPLETE.md](SETUP_COMPLETE.md)** - Quick reference after setup
- **[CHECKLIST.md](CHECKLIST.md)** - Verify all components are working
- **[test_submission_infrastructure.py](test_submission_infrastructure.py)** - Run tests

### 👥 For Participants
- **[CONTRIBUTING.md](CONTRIBUTING.md)** ⭐ **START HERE**
  - How to submit your model
  - CSV format requirements
  - Example code for different approaches
  - FAQ and troubleshooting

- **[submissions/README.md](submissions/README.md)**
  - File naming conventions
  - Submission format details
  - Current leaderboard status

### 🏆 Leaderboard
- **[leaderboard.md](leaderboard.md)** - Public rankings and metrics
  - Current top submissions
  - Detailed submission logs
  - Metrics explained

### 🔧 For Developers/Organizers
- **[SUBMISSION_SETUP.md](SUBMISSION_SETUP.md)** - Implementation details
  - Manual vs. Automated options
  - GitHub Actions workflow explanation
  - Customization guide

## 🎯 Quick Links

| Need | Document | Purpose |
|------|----------|---------|
| **Submit your model** | [CONTRIBUTING.md](CONTRIBUTING.md) | Detailed submission guide |
| **Check standings** | [leaderboard.md](leaderboard.md) | See current rankings |
| **File format help** | [submissions/README.md](submissions/README.md) | CSV format specification |
| **Understand the system** | [SUBMISSION_SETUP.md](SUBMISSION_SETUP.md) | Architecture & design |
| **Verify setup works** | [CHECKLIST.md](CHECKLIST.md) | Test all components |
| **Run tests** | [test_submission_infrastructure.py](test_submission_infrastructure.py) | Validate infrastructure |

## 🚀 System Overview

```
┌─────────────────────────────────────────────────────┐
│          GNN Challenge Submission System             │
├─────────────────────────────────────────────────────┤
│                                                      │
│  📥 INPUT                                           │
│  └─ Participant submits CSV via PR                 │
│                                                      │
│  ⚙️  PROCESSING                                     │
│  ├─ GitHub Actions triggered automatically         │
│  ├─ Scoring script runs                            │
│  ├─ Metrics computed (F1, Accuracy, etc.)          │
│  └─ Results validated                              │
│                                                      │
│  📤 OUTPUT                                          │
│  ├─ PR comment with scores                         │
│  ├─ Leaderboard updated on merge                   │
│  └─ Public rankings visible                        │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## 📊 What Gets Measured

Each submission is evaluated on:

| Metric | Description | Formula |
|--------|-------------|---------|
| **F1-Score (Macro)** | Primary ranking metric | Avg of class-wise F1 scores |
| **Accuracy** | Overall correctness | (TP + TN) / Total |
| **Precision** | False positive rate | TP / (TP + FP) |
| **Recall** | False negative rate | TP / (TP + FN) |
| **Confusion Matrix** | Detailed breakdown | TN, FP, FN, TP counts |

## 🎓 How to Participate

### Step 1: Understand the Task
- Training data: `data/train.csv` with labels (`disease_labels`)
- Test data: `data/test.csv` without labels
- Goal: Predict labels for test samples (0=control, 1=preeclampsia)
- Features: ~6,000 gene expression values

### Step 2: Train Your Model
Use any approach:
- Deep Learning (MLP)
- Graph Neural Networks (GCN, GraphSAGE, etc.)


See [CONTRIBUTING.md](CONTRIBUTING.md) for example code.

### Step 3: Generate Predictions
```python
predictions = model.predict(test_data)
submission_df = pd.DataFrame({
    'node_id': test_df['node_id'],
    'target': predictions,
    'confidence_class0': probabilities[:, 0],  # Optional
    'confidence_class1': probabilities[:, 1]   # Optional
})
submission_df.to_csv('submissions/my_model_submission.csv', index=False)
```

### Step 4: Submit via Pull Request
1. Fork the repository
2. Add your CSV to `submissions/` folder
3. Create a PR with your submission
4. Watch GitHub Actions automatically score it
5. See results in PR comments and leaderboard

## 🏆 Current Leaderboard

| Rank | Model | F1-Score | Accuracy | Type |
|------|-------|----------|----------|------|
| 1 | Advanced GNN (GraphSAGE) | 0.8421 | 0.8387 | Baseline |
| 2 | Baseline MLP | 0.7845 | 0.7742 | Baseline |

**[View full leaderboard →](leaderboard.md)**

## 🔧 System Components

### Files Created

```
📁 Project Root
├── 📄 leaderboard.md                      # Public rankings
├── 📄 CONTRIBUTING.md                     # Participant guide
├── 📄 SUBMISSION_SETUP.md                 # Implementation details
├── 📄 SETUP_COMPLETE.md                   # Quick reference
├── 📄 CHECKLIST.md                        # Verification guide
├── 📄 INDEX.md                            # This file
├── 🐍 update_leaderboard.py               # Auto-update script
├── 🐍 test_submission_infrastructure.py   # Test script
│
├── 📁 .github/workflows
│   └── 📄 score-submission.yml            # GitHub Actions CI/CD
│
└── 📁 submissions
    ├── 📄 README.md                       # Submission guide
    ├── 📊 advanced_gnn_preds.csv         # Baseline 1
    ├── 📊 baseline_mlp_preds.csv         # Baseline 2
    └── 📊 [your_model_submission.csv]    # Your submission here!

```

### Key Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scoring_script.py` | Compute metrics | `python scoring_script.py submissions/file.csv` |
| `update_leaderboard.py` | Auto-update rankings | Run automatically by GitHub Actions |
| `test_submission_infrastructure.py` | Validate system | `python3 test_submission_infrastructure.py` |

## 🎯 Evaluation Metrics Explained

### F1-Score (Primary Metric)
- Balances precision and recall
- Important when both false positives and false negatives matter
- Macro-averaged for balance across classes
- Range: 0-1 (higher is better)

### Accuracy
- Percentage of correct predictions
- Can be misleading with imbalanced data
- Good for overall model performance assessment
- Range: 0-1 (higher is better)

### Precision
- Of predicted positives, how many were correct
- Important when false positives are costly
- Range: 0-1 (higher is better)

### Recall
- Of actual positives, how many we found
- Important when missing positives is costly
- Critical for disease detection (minimize missed cases)
- Range: 0-1 (higher is better)

### Confusion Matrix
```
                Predicted
           Class 0    Class 1
Actual  ┌──────────┬──────────┐
Class 0 │    TN    │    FP    │  (True Negatives, False Positives)
        ├──────────┼──────────┤
Class 1 │    FN    │    TP    │  (False Negatives, True Positives)
        └──────────┴──────────┘
```

## ❓ FAQ

**Q: Can I use external data?**  
A: No, only use the provided training data.

**Q: Can I tune hyperparameters on the test set?**  
A: No, use only training set for development.

**Q: What if my submission format is wrong?**  
A: The validation will catch it and you can resubmit in a new PR.

**Q: How long does scoring take?**  
A: Usually 2-5 minutes from PR creation.

**Q: Can I submit multiple times?**  
A: Yes! Create separate PRs for each submission.

**Q: When is the deadline?**  
A: No hard deadline - submit anytime.

See [CONTRIBUTING.md](CONTRIBUTING.md) for more FAQ.

## 🚀 Getting Started

### For Participants
1. Read [CONTRIBUTING.md](CONTRIBUTING.md)
2. Check example code in that file
3. Train your model locally
4. Generate predictions CSV
5. Submit PR with CSV to `submissions/` folder

### For Organizers
1. Review [SUBMISSION_SETUP.md](SUBMISSION_SETUP.md)
2. Run [test_submission_infrastructure.py](test_submission_infrastructure.py)
3. Enable GitHub Actions in repo settings
4. Test with a sample PR
5. Share leaderboard link with community

## 📞 Need Help?

| Question | Resource |
|----------|----------|
| How to submit? | [CONTRIBUTING.md](CONTRIBUTING.md) |
| CSV format? | [submissions/README.md](submissions/README.md) |
| How scoring works? | [SUBMISSION_SETUP.md](SUBMISSION_SETUP.md) |
| System not working? | [CHECKLIST.md](CHECKLIST.md) or run tests |
| Example code? | [CONTRIBUTING.md](CONTRIBUTING.md#approach-2-deep-learning-mlp) |

## 🎓 Learning Resources

- **Gene Expression Data**: StandardScaler normalization applied per gene
- **Graph Structure**: Cosine similarity edges (top-10 neighbors)
- **Task Type**: Inductive learning (train on one cell type, test on another)
- **Classes**: Binary (0=control, 1=preeclampsia)
- **Challenge**: Domain adaptation (cfRNA to placenta)

## ✅ Verification

Run this to verify everything is working:
```bash
python3 test_submission_infrastructure.py
```

Expected output: **6/6 tests passing** ✅

## 🎉 Ready to Participate?

1. Fork the repository
2. Create a branch: `submission/your-model-name`
3. Train your model
4. Generate predictions
5. Create PR with your CSV
6. Watch the automatic scoring!

**Good luck! 🚀**

---

**Last Updated**: January 7, 2026  
**Challenge Status**: ✅ Active  
**Submissions**: Accepting via GitHub PRs  
**Leaderboard**: Auto-updated on merge
