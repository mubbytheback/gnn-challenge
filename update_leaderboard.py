#!/usr/bin/env python3
"""
update_leaderboard.py

Automatically update leaderboard.md by scoring all submission files.
Runs after successful submission PR merge.
"""

import os
import re
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from datetime import datetime

# Paths
SUBMISSIONS_DIR = "submissions"
DATA_DIR = "data"
LEADERBOARD_FILE = "leaderboard.md"

# Load ground truth
test_labels = pd.read_csv(os.path.join(DATA_DIR, "test_labels.csv"), index_col=0)
test_true = test_labels.iloc[:, 0].values.astype(int)

# Find all submission files (excluding the organizing committee's files)
submission_files = list(Path(SUBMISSIONS_DIR).glob("*_preds.csv"))
EXCLUDE_FILES = ["advanced_gnn_preds.csv", "baseline_mlp_preds.csv"]
submission_files = [f for f in submission_files if f.name not in EXCLUDE_FILES]

print(f"🔍 Found {len(submission_files)} new submission(s)")

# Score each submission
results = []

for sub_file in submission_files:
    print(f"\n📊 Scoring {sub_file.name}...")
    
    try:
        submission = pd.read_csv(sub_file)
        
        if "target" not in submission.columns or len(submission) != len(test_true):
            print(f"   ⚠️ Invalid format or length mismatch. Skipping.")
            continue
        
        preds = submission["target"].values.astype(int)
        
        # Compute metrics
        f1 = f1_score(test_true, preds, average="macro", zero_division=0)
        acc = accuracy_score(test_true, preds)
        prec = precision_score(test_true, preds, zero_division=0)
        rec = recall_score(test_true, preds, zero_division=0)
        cm = confusion_matrix(test_true, preds)
        
        # Extract model name from filename
        model_name = sub_file.stem.replace("_preds", "").replace("_", " ").title()
        
        result = {
            "model_name": model_name,
            "file": sub_file.name,
            "f1_score": f1,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "tn": cm[0, 0],
            "fp": cm[0, 1],
            "fn": cm[1, 0],
            "tp": cm[1, 1],
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
        results.append(result)
        print(f"   ✅ F1={f1:.4f}, Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")
        
    except Exception as e:
        print(f"   ❌ Error scoring {sub_file.name}: {e}")

if not results:
    print("\n⚠️ No new valid submissions to add.")
    exit(0)

# Read current leaderboard
with open(LEADERBOARD_FILE, "r") as f:
    content = f.read()

# Extract current table rows (between the header row and next section)
table_pattern = r'(\| 1 \|.*?\n)(.*?)(\n##)'
match = re.search(table_pattern, content, re.DOTALL)

current_rows = []
if match:
    current_rows_text = match.group(2)
    for line in current_rows_text.strip().split("\n"):
        if line.startswith("|") and "---" not in line:
            # Parse existing row
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 7:
                current_rows.append({
                    "rank": int(parts[0]),
                    "model_name": parts[1],
                    "f1_score": float(parts[2]),
                    "accuracy": float(parts[3]),
                    "precision": float(parts[4]),
                    "recall": float(parts[5]),
                    "date": parts[6]
                })

# Combine and sort
all_results = current_rows + [
    {
        "rank": 0,  # Will be reassigned
        "model_name": r["model_name"],
        "f1_score": r["f1_score"],
        "accuracy": r["accuracy"],
        "precision": r["precision"],
        "recall": r["recall"],
        "date": r["date"]
    }
    for r in results
]

# Sort by F1 score (descending)
all_results.sort(key=lambda x: x["f1_score"], reverse=True)

# Reassign ranks
for i, result in enumerate(all_results, 1):
    result["rank"] = i

# Build new table
header = """| Rank | Model | F1-Score | Accuracy | Precision | Recall | Submission Date | Submitted By |
|------|-------|----------|----------|-----------|--------|-----------------|--------------|"""

rows = []
for r in all_results:
    submitted_by = "organizers" if r["model_name"] in ["Advanced Gnn (Graphsage)", "Baseline Mlp"] else "participant"
    row = f"| {r['rank']} | {r['model_name']} | {r['f1_score']:.4f} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['date']} | {submitted_by} |"
    rows.append(row)

table = header + "\n" + "\n".join(rows)

# Replace table in leaderboard
new_content = re.sub(
    r'(\| Rank \| Model.*?\n\|.*?\n)(.*?)(\n## Submissions Log)',
    r'\1' + "\n".join(rows) + r'\3',
    content,
    flags=re.DOTALL
)

# Add submission logs for new results
submissions_log = "\n\n### " + "\n\n### ".join([
    f"{r['model_name']}\n"
    f"- **F1-Score**: {r['f1_score']:.4f}\n"
    f"- **Accuracy**: {r['accuracy']:.4f}\n"
    f"- **Precision**: {r['precision']:.4f}\n"
    f"- **Recall**: {r['recall']:.4f}\n"
    f"- **Confusion Matrix**: TN={r['tn']}, FP={r['fp']}, FN={r['fn']}, TP={r['tp']}\n"
    f"- **Submission**: `{r['file']}`\n"
    f"- **Date**: {r['date']}"
    for r in results
])

new_content = new_content.replace(
    "## Submissions Log",
    f"## Submissions Log\n{submissions_log}\n\n## Previous Submissions"
)

# Write updated leaderboard
with open(LEADERBOARD_FILE, "w") as f:
    f.write(new_content)

print(f"\n✅ Leaderboard updated with {len(results)} new submission(s)")
print(f"   Top model: {all_results[0]['model_name']} (F1={all_results[0]['f1_score']:.4f})")
