
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# Load previously calculated validation results
df = pd.read_csv(
    RESULTS_DIR / "recall_threshold_sweep.csv"
)

comparison = []

# Compare current threshold and recall targets
settings = [
    ("Current setting", None, 0.14),
    ("70% Recall Target", 0.70, None),
    ("80% Recall Target", 0.80, None),
]

for name, target, threshold in settings:

    if threshold is not None:
        selected = df.loc[
            (df["threshold"] - threshold).abs().idxmin()
        ]
    else:
        eligible = df[df["recall"] >= target]

        if eligible.empty:
            print(f"No threshold achieves {target:.0%} recall")
            continue

        # Highest threshold satisfying the recall target
        selected = eligible.loc[
            eligible["threshold"].idxmax()
        ]

    comparison.append({
        "Setting": name,
        "Threshold": round(selected["threshold"], 3),
        "Recall (%)": round(selected["recall"] * 100, 2),
        "Precision (%)": round(selected["precision"] * 100, 2),
        "Accuracy (%)": round(selected["accuracy"] * 100, 2),
        "Detected": int(selected["true_positives"]),
        "Missed": int(selected["false_negatives"]),
        "False Alerts": int(selected["false_positives"]),
        "Total Flagged": int(selected["flagged_encounters"]),
        "Alert Burden (%)": round(
            selected["alert_burden_pct"], 2
        )
    })

comparison_df = pd.DataFrame(comparison)

print("\n" + "=" * 70)
print("VALIDATION RECALL TARGET COMPARISON")
print("=" * 70)
print(comparison_df.to_string(index=False))

output_path = RESULTS_DIR / "recall_target_comparison.csv"

comparison_df.to_csv(output_path, index=False)

print("\nSaved:", output_path)
print("\nComparison completed successfully!")
