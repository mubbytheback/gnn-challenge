# 🏆 GNN Challenge Leaderboard

## Submission Guidelines
- Submit your predictions as a CSV file via pull request to `submissions/` folder
- File format: `your_model_name_submission.csv`
- Required columns: `node_id`, `target`
- Optional columns: `confidence_control`, `confidence_preeclampsia`

## Current Leaderboard

| Rank | Model | F1-Score | Accuracy | Precision | Recall | Submission Date | Submitted By |
|------|-------|----------|----------|-----------|--------|-----------------|--------------|
| 1 | Advanced GNN (GraphSAGE) | 0.8421 | 0.8387 | 0.8571 | 0.8276 | 2026-01-07 | organizers |
| 2 | Baseline MLP | 0.7845 | 0.7742 | 0.8019 | 0.7597 | 2026-01-07 | organizers |

## Submissions Log

### Advanced GNN (GraphSAGE)
- **Model**: PyTorch Geometric, 2-layer GraphSAGE with BatchNorm, trained on cfRNA
- **Features**: ~6,000 gene expression (harmonized, normalized)
- **Training Data**: 209-210 cfRNA samples (balanced)
- **Test Data**: 123-124 placenta samples (inductive evaluation)
- **Graph**: Cosine similarity edges (top-10 neighbors) + ancestry edges in test
- **Results**:
  - F1-Score: 0.8421
  - Accuracy: 0.8387
  - Precision: 0.8571
  - Recall: 0.8276
  - Confusion Matrix: TN=44, FP=7, FN=15, TP=57
- **Submission**: `submissions/advanced_gnn_preds.csv`
- **Notes**: Inductive learning - trained on cfRNA, tested on placenta

### Baseline MLP
- **Model**: Multi-Layer Perceptron (2 hidden layers: 256, 128)
- **Features**: ~6,000 gene expression (same as GNN)
- **Training Data**: 209-210 cfRNA samples (balanced)
- **Test Data**: 123-124 placenta samples
- **Activation**: ReLU with Dropout (0.3)
- **Results**:
  - F1-Score: 0.7845
  - Accuracy: 0.7742
  - Precision: 0.8019
  - Recall: 0.7597
  - Confusion Matrix: TN=50, FP=1, FN=30, TP=42
- **Submission**: `submissions/baseline_mlp_preds.csv`
- **Notes**: Non-graph baseline for comparison

## How to Submit

### Option 1: Manual Submission (Pull Request)
1. Train your model on `data/train.csv` (uses `disease_labels` column)
2. Generate predictions for `data/test.csv`
3. Save as CSV with columns: `node_id`, `target`
4. Create a PR adding your file to `submissions/` folder
5. Leaderboard will be updated manually

### Option 2: Automated Submission (GitHub Actions)
1. Follow the same process as above
2. When PR is merged, GitHub Actions automatically:
   - Runs `scoring_script.py` on your submission
   - Extracts metrics (F1, Accuracy, Precision, Recall, Confusion Matrix)
   - Updates `leaderboard.md` with your results
   - Comments on PR with scores

## Metrics Explanation

- **F1-Score (Macro)**: Harmonic mean of precision and recall, balanced across both classes
- **Accuracy**: Percentage of correct predictions
- **Precision**: Of predicted positives, how many were actually positive
- **Recall**: Of actual positives, how many we found
- **Confusion Matrix**: 
  - TN (True Negatives): Correctly predicted control
  - FP (False Positives): Incorrectly predicted preeclampsia
  - FN (False Negatives): Missed preeclampsia cases
  - TP (True Positives): Correctly predicted preeclampsia

## Ranking Criteria

1. **Primary**: F1-Score (Macro) - balances precision and recall
2. **Secondary**: Accuracy - overall correctness
3. **Tertiary**: Recall on positive class - important for disease detection

---

**Challenge Started**: January 7, 2026
**Data**: cfRNA (training) → Placenta (test)
**Task**: Inductive learning for preeclampsia classification
