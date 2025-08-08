import os
import pandas as pd

# === File paths ===
results_path = "results.txt"
outliers_path = "results_with_outliers.txt"
logs_folder = "outlier_logs"

# === Parse results.txt ===
def parse_results_file(filepath, with_outliers=False):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if 'ERROR' in parts:
                continue
            if with_outliers and len(parts) == 4:
                dataset, model, technique, acc = parts
            elif not with_outliers and len(parts) == 3:
                dataset, model, acc = parts
            else:
                continue
            try:
                acc = float(acc)
                if 0.3 <= acc <= 0.99:
                    if with_outliers:
                        data.append((dataset, model, technique, acc))
                    else:
                        data.append((dataset, model, acc))
            except:
                continue
    return data

# === Parse outlier logs ===
def parse_outlier_logs():
    outlier_data = {}
    for fname in os.listdir(logs_folder):
        if not fname.endswith("_outliers.txt"):
            continue
        dataset = fname.replace("_outliers.txt", "")
        with open(os.path.join(logs_folder, fname), 'r') as f:
            content = f.read().strip().split("\n\n")
            for block in content:
                lines = block.strip().split('\n')
                if len(lines) != 3:
                    continue
                header = lines[0].strip()
                ds, rest = header.split(" - ")
                split, technique = rest.strip("():").split("(")
                detected = int(lines[1].split(":")[1].strip())
                total = int(lines[2].split(":")[1].strip())
                percent = detected / total * 100 if total > 0 else 0.0
                outlier_data[(dataset, split.strip(), technique.strip())] = percent
    return outlier_data

# === Load data ===
results = parse_results_file(results_path)
results_df = pd.DataFrame(results, columns=['dataset', 'model', 'accuracy'])

results_outliers = parse_results_file(outliers_path, with_outliers=True)
outliers_df = pd.DataFrame(results_outliers, columns=['dataset', 'model', 'technique', 'accuracy'])

outlier_logs = parse_outlier_logs()

# ------------------------------
# === Task 1: Wide format table ===
# Rows: dataset, Columns: model, Values: accuracy
task1_df = results_df.pivot(index="dataset", columns="model", values="accuracy")
task1_df.to_csv("task1_accuracy_matrix.csv")

# ------------------------------
# === Task 2: MLP vs XGBoost accuracy + IsolationForest test outliers ===
mlp_xgb_df = results_df[results_df['model'].isin(['mlp', 'xgboost'])].pivot(index="dataset", columns="model", values="accuracy")
mlp_xgb_df = mlp_xgb_df.dropna()
mlp_xgb_df['diff'] = mlp_xgb_df['xgboost'] - mlp_xgb_df['mlp']
mlp_xgb_df['IsolationForest_test_outliers(%)'] = mlp_xgb_df.index.map(
    lambda ds: outlier_logs.get((ds, 'test', 'IsolationForest'), 0.0)
)
mlp_xgb_df.to_csv("task2_mlp_vs_xgboost.csv")

# ------------------------------
# === Task 3: Table of % of outliers in test set ===
# Rows: dataset, Columns: [IsolationForest, LocalOutlierFactor, OneClassSVM]
methods = ['IsolationForest', 'LocalOutlierFactor', 'OneClassSVM', 'ZScore', 'ModifiedZScore', 'IQR']
task3_data = {}
for dataset, _, _ in results:
    row = {}
    for method in methods:
        row[method] = outlier_logs.get((dataset, 'test', method), 0.0)
    task3_data[dataset] = row
task3_df = pd.DataFrame.from_dict(task3_data, orient='index')
task3_df.to_csv("task3_outlier_test_percentages.csv")

# ------------------------------
# === Task 4: Accuracy difference with outliers - no outliers ===
# Exclude socmob, elevators
excluded_datasets = {'socmob', 'elevators'}
methods_set = set(methods)

# Create base DataFrame
merged = pd.merge(results_df, outliers_df, on=["dataset", "model"], suffixes=("_no_outliers", "_with_outliers"))
merged = merged[~merged['dataset'].isin(excluded_datasets)]
merged = merged[merged['technique'].isin(methods)]

# Pivot to desired format
merged['diff'] = merged['accuracy_with_outliers'] - merged['accuracy_no_outliers']
pivot_df = merged.pivot_table(index=["dataset", "technique"], columns="model", values="diff")
pivot_df.to_csv("task4_accuracy_differences.csv")

# Summary row: win counts
greater_count = (pivot_df > 0).sum().sum()
lower_count = (pivot_df < 0).sum().sum()
summary_df = pd.DataFrame([{
    'count_with_outliers_greater': greater_count,
    'count_without_outliers_greater': lower_count,
    'difference': greater_count - lower_count
}])
summary_df.to_csv("task4_summary_row.csv", index=False)
