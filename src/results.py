import os
import json
import pandas as pd
from glob import glob
from collections import defaultdict

# ----------------------------------------------------------
# Helper: load experiment
# ----------------------------------------------------------
def load_experiment_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

# ----------------------------------------------------------
# Collect experiment files: algorithm/dataset/seed/experiment.json
# ----------------------------------------------------------
def collect_experiment_files(root):
    experiment_files = defaultdict(list)
    all_jsons = glob(os.path.join(root, "**", "experiment.json"), recursive=True)

    if len(all_jsons) == 0:
        all_jsons = glob(os.path.join(root, "**", "experiment_results.json"), recursive=True)

    print(f"Found {len(all_jsons)} experiment files.")

    for json_path in all_jsons:
        parts = json_path.split(os.sep)
        # [..., log_root, alg, dataset, seed, experiment.json]
        dataset = parts[-3]
        experiment_files[dataset].append(json_path)

    return experiment_files

# ----------------------------------------------------------
# Extract flattened metrics
# ----------------------------------------------------------
def extract_metrics(exp):
    metrics = {}

    # Number of trainable params (directly from JSON)
    if "n_trainable_params" in exp:
        metrics["n_trainable_params"] = exp["n_trainable_params"]

    # Training metrics
    train = exp.get("training", {})
    for k, v in train.items():
        metrics[f"train_{k}"] = v

    # Test results
    results = exp.get("results", {})
    for k, v in results.items():
        metrics[k] = v

    return metrics



# ----------------------------------------------------------
# Summary with structure unchanged:
# mean, std, best (but best = metrics from best test_cl_iou run)
# ----------------------------------------------------------
def summarize_metrics(metrics_list):
    df = pd.DataFrame(metrics_list)

    summary = {"dataset": None}

    # Compute mean & std normally
    for col in df.columns:
        summary[f"{col}_mean"] = df[col].mean()
        summary[f"{col}_std"] = df[col].std()

    # Identify best run: highest test_cl_iou
    if "test_cl_iou" in df.columns:
        best_idx = df["test_cl_iou"].idxmax()
        best_row = df.loc[best_idx]
    else:
        print("Warning: test_cl_iou missing, using first run as fallback.")
        best_row = df.iloc[0]

    # For every column, best = value from best-row run
    for col in df.columns:
        summary[f"{col}_best"] = best_row[col]

    return summary

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main(log_root, output_csv):
    experiment_files = collect_experiment_files(log_root)
    summaries = []

    for dataset, files in experiment_files.items():
        print(f"\nProcessing dataset: {dataset} ({len(files)} runs)")

        metrics_list = []

        for json_path in files:
            exp = load_experiment_json(json_path)
            if exp is None:
                continue

            metrics = extract_metrics(exp)
            metrics_list.append(metrics)

        if not metrics_list:
            print(f"No valid metrics for dataset {dataset}")
            continue

        summary = summarize_metrics(metrics_list)
        summary["dataset"] = dataset
        summaries.append(summary)

    df_summary = pd.DataFrame(summaries)
    param_cols = [c for c in df_summary.columns if "n_trainable_params" in c]
    other_cols = [c for c in df_summary.columns if c not in param_cols]

    df_summary = df_summary[other_cols + param_cols]
    df_summary.to_csv(output_csv, index=False)
    print(f"\nSaved summary to {output_csv}")


if __name__ == "__main__":
    log_root = "logs/deepcrackz/100"
    output_csv = os.path.join(log_root, "summary.csv")
    main(log_root, output_csv)
