# Contributing Submissions to GNN Challenge

This [repository](https://github.com/Mubarraqqq/gnn-challenge) accepts **prediction files only**. No participant code is executed.

## Quick Start (External Participants)

1) **Fork** the repo on GitHub.  
2) **Clone** your fork:
```bash
git clone https://github.com/<their-username>/gnn-challenge.git
cd gnn-challenge
```
3) **Create a branch**:
```bash
git checkout -b new_submission
```
4) **Train your model** using `data/public/train.csv` and generate predictions for `data/public/test.csv`.
   - Create `predictions.csv` with columns: `id`, `y_pred`.
   - IDs **must** match `data/public/test_nodes.csv`.

5) **Create submission folder**:
```
submissions/inbox/<team>/<run_id>/predictions.csv
submissions/inbox/<team>/<run_id>/metadata.json
```

Example `metadata.json`:
```json
{
  "team": "my_team",
  "run_id": "run_001",
  "model_name": "My GNN v1",
  "model_type": "human"
}
```

6) **Commit + push**:
```bash
git add submissions/inbox/<team>/<run_id>/predictions.csv
git add submissions/inbox/<team>/<run_id>/metadata.json
git commit -m "Add submission: My GNN v1"
git push origin submission/my-model
```

7) **Open a PR** to the main repo. CI validates + scores. On merge, the leaderboard updates.

---

## Submission Format

Required:
- `predictions.csv` with columns `id`, `y_pred`
- `metadata.json`

Notes:
- `y_pred` can be probability (0–1) or hard label (0/1)
- IDs must match `data/public/test_nodes.csv`

---
## Minimal ML Baseline (Quick Start)

If you just want a working baseline, here’s a tiny logistic‑regression example:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("data/public/train.csv")
test = pd.read_csv("data/public/test.csv")

X = train.drop(columns=["node_id", "sample_id", "disease_labels"], errors="ignore")
y = train["disease_labels"]
X_test = test.drop(columns=["node_id", "sample_id"], errors="ignore")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

proba = model.predict_proba(X_test_scaled)[:, 1]
predictions = pd.DataFrame({"id": test["node_id"], "y_pred": proba})
predictions.to_csv("predictions.csv", index=False)
```

---
## Leaderboard

Your submission can be viewed [here](https://mubarraqqq.github.io/gnn-challenge/leaderboard.html) after PR has been merged by the [organizer](www.github.com/mubarraqqq)

---

## Evaluation Metrics

Primary metric: **Macro F1**  
Also reported: Accuracy, Precision, Recall.

---

## FAQ

**Do I need to run any code in this repo?**  
No. You only submit prediction files.

**Who merges PRs?**  
Maintainers. The leaderboard updates after merge.

**Can I submit multiple runs?**  
Yes. Use a new `<run_id>` for each submission.
