# Submission & Leaderboard Setup

**This project uses automated scoring only.** Submissions are evaluated in CI against a hidden test set and the leaderboard updates on merge.

This project uses fully automated scoring and leaderboard updates via GitHub Actions.

## Automated Scoring with GitHub Actions

### How It Works

1. **Participant** submits PR with CSV file
2. **GitHub Actions** automatically:
   - Detects new submission
   - Runs scoring script
   - Posts results as PR comment
   - Updates leaderboard (on merge)

### Files Used

- **`.github/workflows/score-submission.yml`** - GitHub Actions workflow
- **`update_leaderboard.py`** - Automated leaderboard updater
- **`scoring_script.py`** - Scoring utility
- **`leaderboard.md`** - Auto-updated leaderboard

### Workflow

```
Participant Creates PR
    ↓
GitHub Actions Triggered
    ↓
Checkout Code + Install Dependencies
    ↓
Find New Submission File
    ↓
Run scoring_script.py
    ↓
Post Results as PR Comment
    ↓
(On Merge) Run update_leaderboard.py
    ↓
Leaderboard Auto-Updated
```

### Advantages ✅
- Fully automated
- Fast feedback (2-5 minutes)
- Consistent scoring
- Transparent scoring
- Professional appearance
- Scales to many submissions

### Disadvantages ❌
- Requires GitHub Actions setup
- More complex workflow
- Less control over submissions
- Requires proper CI/CD permissions

### GitHub Actions Requirements

```yaml
# Permissions needed in .github/workflows/score-submission.yml
permissions:
  pull-requests: write        # To comment on PRs
  contents: write             # To update leaderboard.md
```

### Workflow Trigger

```yaml
on:
  pull_request:
    paths:
      - 'submissions/*.csv'   # Only triggers when CSV files change
  workflow_dispatch:          # Manual trigger option
```

---

## 🚀 Setting Up Option 2 (Recommended)

### Step 1: Ensure GitHub Actions is Enabled

1. Go to **Settings** → **Actions** → **General**
2. Enable **"Allow all actions and reusable workflows"**
3. Select **"Read and write permissions"** for workflows

### Step 2: Add Files to Your Repo

✅ Already created:
- `.github/workflows/score-submission.yml`
- `update_leaderboard.py`
- `leaderboard.md`
- `CONTRIBUTING.md`

### Step 3: Test the Workflow

```bash
# Push a test submission to trigger the workflow
git checkout -b test/submission
cp submissions/baseline_mlp_preds.csv submissions/test_model_submission.csv
git add submissions/test_model_submission.csv
git commit -m "Test: Automated scoring"
git push origin test/submission
```

Then create a PR and watch GitHub Actions run!

### Step 4: Monitor Workflow

1. Go to **Actions** tab
2. Select **"Score Submission"** workflow
3. Click on your PR's workflow run
4. View logs and results

---

## 📊 Leaderboard Features

The `update_leaderboard.py` script:
1. Reads all submission files in `submissions/`
2. Scores each against the hidden test labels in CI
3. Sorts by F1-Score (descending)
4. Updates table in `leaderboard.md`
5. Adds submission log entries
6. Commits changes back to repo

## 📝 Example PR Comment (Automated)

When a participant submits via GitHub Actions:

```
## 🏆 Submission Scored

Submission F1 Score: 0.8234
Accuracy: 0.8065
Precision: 0.8367
Recall: 0.8108
Confusion Matrix:
  TN=46, FP=5, FN=17, TP=55

Your submission has been evaluated against the test set. 
Check the leaderboard for rankings!
```

---

## 🔧 Customization

### Change Scoring Metric

Edit `update_leaderboard.py`:
```python
# Change primary metric from F1 to Accuracy
all_results.sort(key=lambda x: x["accuracy"], reverse=True)
```

### Change GitHub Actions Trigger

Edit `.github/workflows/score-submission.yml`:
```yaml
on:
  pull_request:
    paths:
      - 'submissions/*.csv'    # Only CSV files in submissions/
      - 'starter_code/**'      # Or include other directories
  schedule:                     # Or run on schedule
    - cron: '0 0 * * 0'        # Every Sunday
```

### Add More Metrics

Edit `update_leaderboard.py`:
```python
# Add ROC-AUC
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(test_true, proba[:, 1])

result = {
    # ... existing fields ...
    "roc_auc": roc_auc,
}
```

---

## ✅ Checklist

- [x] Create `leaderboard.md` with initial submissions
- [x] Create `update_leaderboard.py` for automation
- [x] Set up `.github/workflows/score-submission.yml`
- [x] Create `CONTRIBUTING.md` with submission guide
- [x] Create `submissions/README.md`
- [x] Test scoring script works
- [ ] Enable GitHub Actions
- [ ] Test workflow with sample submission
- [ ] Update main `README.md` to link to submission guide
- [ ] Document evaluation metrics

---

## 🎓 Next Steps

1. **Participants** should read [CONTRIBUTING.md](CONTRIBUTING.md)
2. **Test** the submission process with a test PR
3. **Announce** the challenge and leaderboard
4. **Monitor** submissions and engagement
5. **Celebrate** top performers!

---

## 📞 Support

For issues or questions:
- Check [CONTRIBUTING.md](CONTRIBUTING.md) FAQ
- Open GitHub Issue for bugs
- Start Discussion for questions
