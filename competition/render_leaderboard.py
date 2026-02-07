"""
render_leaderboard.py

Render leaderboard.md from leaderboard/leaderboard.csv.
"""

import pandas as pd
from pathlib import Path

LEADERBOARD_CSV = Path("leaderboard/leaderboard.csv")
LEADERBOARD_MD = Path("leaderboard.md")

if not LEADERBOARD_CSV.exists():
    raise FileNotFoundError("leaderboard/leaderboard.csv not found")

leaderboard = pd.read_csv(LEADERBOARD_CSV)

lines = [
    "# 🏆 GNN Challenge Leaderboard",
    "",
    "## Current Leaderboard",
    "",
    "| Rank | Team | Run | Model | Type | F1-Score | Accuracy | Precision | Recall | Date | Submitter |",
    "|------|------|-----|-------|------|----------|----------|-----------|--------|------|-----------|",
]

for _, row in leaderboard.iterrows():
    submitter = row.get("submitter", "")
    submitter_url = row.get("submitter_url", "")
    submitter_md = f"[{submitter}]({submitter_url})" if submitter and submitter_url else submitter
    lines.append(
        "| {rank} | {team} | {run_id} | {model} | {model_type} | {f1_score:.4f} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {submission_date} | {submitter} |".format(
            rank=int(row["rank"]),
            team=row["team"],
            run_id=row["run_id"],
            model=row["model"],
            model_type=row["model_type"],
            f1_score=row["f1_score"],
            accuracy=row["accuracy"],
            precision=row["precision"],
            recall=row["recall"],
            submission_date=row["submission_date"],
            submitter=submitter_md,
        )
    )

lines.extend([
    "",
    "## Notes",
    "- This leaderboard is auto-generated from `leaderboard/leaderboard.csv`.",
    "- Submissions must follow the `submissions/inbox/<team>/<run_id>/predictions.csv` format.",
])

LEADERBOARD_MD.write_text("\n".join(lines), encoding="utf-8")
