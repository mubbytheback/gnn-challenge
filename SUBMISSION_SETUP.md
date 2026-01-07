# Submission & Leaderboard Setup

This project supports two approaches for hosting submissions and managing a leaderboard on GitHub.

## 📋 Overview

| Aspect | Manual Submission | Automated Scoring |
|--------|------------------|-------------------|
| **Setup Effort** | Low | Medium |
| **User Experience** | Simple, flexible | Fast, transparent |
| **Automation** | Manual updates | Fully automated |
| **Best For** | Small competitions | Large-scale challenges |

---

## Option 1: Manual Submission with Leaderboard.md

### How It Works

1. **Participants** submit predictions via Pull Request
2. **Organizers** manually run scoring script
3. **Leaderboard** updated manually in `leaderboard.md`

### Files Used

- **`leaderboard.md`** - Central leaderboard document
- **`scoring_script.py`** - Manual scoring utility
- **`submissions/`** - Submission storage directory

### Workflow

```
Participant Submits PR
    ↓
Organizer Reviews PR
    ↓
Run: python scoring_script.py submissions/{file}.csv
    ↓
Organizer Updates leaderboard.md
    ↓
PR Merged + Leaderboard Updated
```

### Advantages ✅
- Simple to set up
- Full control over submissions
- Can verify quality before adding to leaderboard
- No CI/CD required

### Disadvantages ❌
- Requires manual effort
- Slower feedback to participants
- Error-prone leaderboard updates
- Doesn't scale well

### Commands

```bash
# Score a single submission
python scoring_script.py submissions/my_model_submission.csv

# Output example:
# Submission F1 Score: 0.8421
# Accuracy: 0.8387
# Precision: 0.8571
# Recall: 0.8276
```

---

## Option 2: Automated Scoring with GitHub Actions

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
- No manual errors
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

### Manual Mode (`leaderboard.md`)

```markdown
| Rank | Model | F1-Score | Accuracy | Date | By |
|------|-------|----------|----------|------|-----|
| 1 | Advanced GNN | 0.8421 | 0.8387 | 2026-01-07 | organizers |
| 2 | Baseline MLP | 0.7845 | 0.7742 | 2026-01-07 | organizers |
```

### Automated Mode

The `update_leaderboard.py` script:
1. Reads all submission files in `submissions/`
2. Scores each against `test_labels.csv`
3. Sorts by F1-Score (descending)
4. Updates table in `leaderboard.md`
5. Adds submission log entries
6. Commits changes back to repo

---

## 🎯 Recommendation: Use Both

### Hybrid Approach

1. **Use GitHub Actions** for:
   - Automatic scoring and feedback
   - Transparent ranking
   - Reducing manual work

2. **Use `leaderboard.md`** for:
   - Final official rankings
   - Detailed submission logs
   - Challenge information

This provides the best user experience while maintaining quality control.

---

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
- [ ] Enable GitHub Actions (if using Option 2)
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
