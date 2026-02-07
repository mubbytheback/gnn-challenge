# Contributing Submissions to GNN Challenge

This repository accepts **prediction files only**. No participant code is executed.

## Quick Start (External Participants)

1) **Fork** the repo on GitHub.  
2) **Clone** your fork:
```bash
git clone https://github.com/<their-username>/gnn-challenge.git
cd gnn-challenge
```
3) **Create a branch**:
```bash
git checkout -b submission/my-model
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
