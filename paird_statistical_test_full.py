import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef

REQUIRED_KEYS = ("sample_id", "y_true", "y_pred")


def parse_comparisons(items):
    comparisons = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid comparison '{item}'. Use ModelName=/path/to/file.npz")
        name, path = item.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"Invalid comparison '{item}'.")
        if name in comparisons:
            raise ValueError(f"Duplicate model name: {name}")
        comparisons[name] = path
    return comparisons


def scalar_text(value, default="unknown"):
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return str(item)


def load_predictions(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data.files]
        if missing:
            raise KeyError(f"{path} is missing {missing}. Available fields: {data.files}")
        result = {key: np.asarray(data[key]) for key in data.files}
    result["sample_id"] = result["sample_id"].reshape(-1)
    result["y_true"] = result["y_true"].astype(np.int64).reshape(-1)
    result["y_pred"] = result["y_pred"].astype(np.int64).reshape(-1)
    n_samples = len(result["y_true"])
    if len(result["sample_id"]) != n_samples or len(result["y_pred"]) != n_samples:
        raise ValueError(f"Inconsistent sample lengths in {path}")
    if "y_prob" in result:
        result["y_prob"] = np.asarray(result["y_prob"], dtype=np.float64)
        if result["y_prob"].ndim != 2 or result["y_prob"].shape[0] != n_samples:
            raise ValueError(f"Invalid y_prob shape in {path}: {result['y_prob'].shape}")
    result["_path"] = str(path)
    result["_algorithm"] = scalar_text(result.get("algorithm"), path.parent.name)
    result["_seed"] = scalar_text(result.get("seed"), "unknown")
    return result


def verify_pairing(reference, comparison, reference_name, comparison_name):
    if len(reference["y_true"]) != len(comparison["y_true"]):
        raise ValueError(f"Sample-count mismatch: {reference_name}={len(reference['y_true'])}, {comparison_name}={len(comparison['y_true'])}")
    if not np.array_equal(reference["sample_id"], comparison["sample_id"]):
        mismatch = np.flatnonzero(reference["sample_id"] != comparison["sample_id"])
        first = int(mismatch[0]) if len(mismatch) else -1
        raise ValueError(f"sample_id mismatch between {reference_name} and {comparison_name}; first mismatch at position {first}.")
    if not np.array_equal(reference["y_true"], comparison["y_true"]):
        mismatch = np.flatnonzero(reference["y_true"] != comparison["y_true"])
        first = int(mismatch[0]) if len(mismatch) else -1
        raise ValueError(f"y_true mismatch between {reference_name} and {comparison_name}; first mismatch at position {first}.")
    if "y_prob" in reference and "y_prob" in comparison and reference["y_prob"].shape != comparison["y_prob"].shape:
        raise ValueError(f"y_prob shape mismatch: {reference_name}={reference['y_prob'].shape}, {comparison_name}={comparison['y_prob'].shape}")


def macro_sensitivity_specificity(y_true, y_pred, class_labels):
    matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
    total = matrix.sum()
    sensitivities = []
    specificities = []
    for class_index in range(len(class_labels)):
        true_positive = float(matrix[class_index, class_index])
        false_negative = float(matrix[class_index, :].sum() - true_positive)
        false_positive = float(matrix[:, class_index].sum() - true_positive)
        true_negative = float(total - true_positive - false_negative - false_positive)
        sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else np.nan
        specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive > 0 else np.nan
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    return float(np.nanmean(sensitivities)), float(np.nanmean(specificities))


def metric_value(name, y_true, y_pred, class_labels):
    if name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if name == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if name == "qwk":
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    if name == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro", labels=class_labels, zero_division=0))
    if name == "f1_micro":
        return float(f1_score(y_true, y_pred, average="micro", labels=class_labels, zero_division=0))
    if name == "mcc":
        return float(matthews_corrcoef(y_true, y_pred))
    if name == "sensitivity":
        sensitivity, _ = macro_sensitivity_specificity(y_true, y_pred, class_labels)
        return sensitivity
    if name == "specificity":
        _, specificity = macro_sensitivity_specificity(y_true, y_pred, class_labels)
        return specificity
    raise ValueError(f"Unsupported metric: {name}")


def exact_mcnemar(y_true, reference_pred, comparison_pred, alternative):
    reference_correct = reference_pred == y_true
    comparison_correct = comparison_pred == y_true
    both_correct = int(np.sum(reference_correct & comparison_correct))
    reference_only = int(np.sum(reference_correct & ~comparison_correct))
    comparison_only = int(np.sum(~reference_correct & comparison_correct))
    both_wrong = int(np.sum(~reference_correct & ~comparison_correct))
    discordant = reference_only + comparison_only
    p_value = 1.0 if discordant == 0 else float(binomtest(reference_only, n=discordant, p=0.5, alternative=alternative).pvalue)
    return {"both_correct": both_correct, "reference_only_correct": reference_only, "comparison_only_correct": comparison_only, "both_wrong": both_wrong, "discordant": discordant, "mcnemar_p": p_value}


def paired_wilcoxon_ordinal_error(y_true, reference_pred, comparison_pred, alternative):
    reference_error = np.abs(reference_pred.astype(np.float64) - y_true.astype(np.float64))
    comparison_error = np.abs(comparison_pred.astype(np.float64) - y_true.astype(np.float64))
    error_improvement = comparison_error - reference_error
    nonzero_count = int(np.count_nonzero(error_improvement))
    if nonzero_count == 0:
        return {"reference_mae": float(np.mean(reference_error)), "comparison_mae": float(np.mean(comparison_error)), "mean_error_improvement": 0.0, "median_error_improvement": 0.0, "wilcoxon_statistic": 0.0, "wilcoxon_p": 1.0, "nonzero_pairs": 0}
    try:
        result = wilcoxon(error_improvement, zero_method="pratt", alternative=alternative, method="approx")
    except TypeError:
        result = wilcoxon(error_improvement, zero_method="pratt", alternative=alternative, mode="approx")
    return {"reference_mae": float(np.mean(reference_error)), "comparison_mae": float(np.mean(comparison_error)), "mean_error_improvement": float(np.mean(error_improvement)), "median_error_improvement": float(np.median(error_improvement)), "wilcoxon_statistic": float(result.statistic), "wilcoxon_p": float(result.pvalue), "nonzero_pairs": nonzero_count}


def draw_bootstrap_indices(y_true, rng, mode):
    if mode == "ordinary":
        return rng.integers(0, len(y_true), size=len(y_true))
    sampled_parts = []
    for class_label in np.unique(y_true):
        class_indices = np.flatnonzero(y_true == class_label)
        sampled_parts.append(rng.choice(class_indices, size=len(class_indices), replace=True))
    sampled_indices = np.concatenate(sampled_parts)
    rng.shuffle(sampled_indices)
    return sampled_indices


def paired_bootstrap(y_true, reference_pred, comparison_pred, metric, n_bootstrap, seed, alternative, class_labels, bootstrap_mode):
    rng = np.random.default_rng(seed)
    reference_score = metric_value(metric, y_true, reference_pred, class_labels)
    comparison_score = metric_value(metric, y_true, comparison_pred, class_labels)
    observed_difference = reference_score - comparison_score
    differences = []
    for _ in range(n_bootstrap):
        indices = draw_bootstrap_indices(y_true, rng, bootstrap_mode)
        difference = metric_value(metric, y_true[indices], reference_pred[indices], class_labels) - metric_value(metric, y_true[indices], comparison_pred[indices], class_labels)
        if np.isfinite(difference):
            differences.append(difference)
    if not differences:
        raise RuntimeError(f"No finite bootstrap results for {metric}")
    differences = np.asarray(differences, dtype=np.float64)
    ci_low, ci_high = np.percentile(differences, [2.5, 97.5])
    if alternative == "greater":
        p_value = (np.sum(differences <= 0) + 1) / (len(differences) + 1)
    elif alternative == "less":
        p_value = (np.sum(differences >= 0) + 1) / (len(differences) + 1)
    else:
        lower_tail = (np.sum(differences <= 0) + 1) / (len(differences) + 1)
        upper_tail = (np.sum(differences >= 0) + 1) / (len(differences) + 1)
        p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return {"reference_score": reference_score, "comparison_score": comparison_score, "difference": observed_difference, "ci_low_95": float(ci_low), "ci_high_95": float(ci_high), "bootstrap_p": float(p_value), "valid_bootstrap_samples": int(len(differences))}


def save_csv(rows, path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def significance_text(p_value, alpha):
    return "YES" if p_value < alpha else "NO"


def print_final_summary(summary_rows, alpha):
    print("\n" + "=" * 180)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 180)
    print(f"{'Comparison':<18}{'�QWK':>10}{'pQWK':>12}{'�Acc':>10}{'McNemar':>12}{'�BAcc':>10}{'�Sens':>10}{'�Spec':>10}{'�F1M':>10}{'�MCC':>10}{'�OrdErr':>11}{'Wilcoxon':>12}")
    print("-" * 180)
    for row in summary_rows:
        qwk = row.get("qwk_difference", np.nan)
        qwk_p = row.get("qwk_bootstrap_p", np.nan)
        accuracy = row.get("accuracy_difference", np.nan)
        balanced_accuracy = row.get("balanced_accuracy_difference", np.nan)
        sensitivity = row.get("sensitivity_difference", np.nan)
        specificity = row.get("specificity_difference", np.nan)
        f1_macro = row.get("f1_macro_difference", np.nan)
        mcc = row.get("mcc_difference", np.nan)
        print(f"{row['comparison_model']:<18}{qwk:>+10.4f}{qwk_p:>12.4g}{accuracy:>+10.4f}{row['mcnemar_p']:>12.4g}{balanced_accuracy:>+10.4f}{sensitivity:>+10.4f}{specificity:>+10.4f}{f1_macro:>+10.4f}{mcc:>+10.4f}{row['mean_ordinal_error_improvement']:>+11.4f}{row['wilcoxon_p']:>12.4g}")
    print("-" * 180)
    print(f"Positive metric deltas favour the reference model. Positive �OrdErr means the reference model has lower absolute ordinal error. Statistical threshold: p < {alpha}.")
    print("Sensitivity is macro one-vs-rest recall and is therefore expected to equal balanced accuracy in this multiclass setting.")


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap, McNemar, and Wilcoxon tests for OrdDist versus USB predictions.")
    parser.add_argument("--reference", required=True, help="OrdDist/reference NPZ file")
    parser.add_argument("--reference_name", default="OrdDist")
    parser.add_argument("--comparisons", nargs="+", required=True, metavar="NAME=PATH", help="Example: FixMatch=/path/file.npz CGMatch=/path/file.npz")
    parser.add_argument("--metrics", nargs="+", default=["qwk", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1_macro", "mcc"], choices=["qwk", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1_macro", "f1_micro", "mcc"])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap_mode", choices=["stratified", "ordinary"], default="stratified")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--alternative", choices=["greater", "less", "two-sided"], default="two-sided", help="Use two-sided for standard journal reporting; greater tests whether the reference model is better.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output_dir", default="./statistical_results")
    args = parser.parse_args()

    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1")

    comparisons = parse_comparisons(args.comparisons)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_predictions(args.reference)
    y_true = reference["y_true"]
    reference_pred = reference["y_pred"]
    class_labels = np.unique(y_true)

    print("=" * 96)
    print("PAIRED STATISTICAL COMPARISON")
    print("=" * 96)
    print(f"Reference       : {args.reference_name}")
    print(f"Reference file  : {reference['_path']}")
    print(f"Reference seed  : {reference['_seed']}")
    print(f"Test samples    : {len(y_true)}")
    print(f"Classes         : {class_labels.tolist()}")
    print(f"Alternative     : {args.alternative}")
    print(f"Bootstrap mode  : {args.bootstrap_mode}")
    print(f"Bootstrap runs  : {args.bootstrap}")
    print(f"Alpha           : {args.alpha}")

    detailed_rows = []
    summary_rows = []
    json_results = {"reference_name": args.reference_name, "reference_file": reference["_path"], "reference_seed": reference["_seed"], "alternative": args.alternative, "bootstrap_mode": args.bootstrap_mode, "bootstrap_samples": args.bootstrap, "alpha": args.alpha, "comparisons": {}}

    for model_index, (comparison_name, comparison_path) in enumerate(comparisons.items()):
        comparison = load_predictions(comparison_path)
        verify_pairing(reference, comparison, args.reference_name, comparison_name)
        comparison_pred = comparison["y_pred"]
        mcnemar = exact_mcnemar(y_true, reference_pred, comparison_pred, args.alternative)
        wilcoxon_result = paired_wilcoxon_ordinal_error(y_true, reference_pred, comparison_pred, args.alternative)

        print("\n" + "-" * 96)
        print(f"{args.reference_name} vs {comparison_name}")
        print("Pairing check   : PASSED")
        print(f"Discordant      : {args.reference_name}-only correct={mcnemar['reference_only_correct']}, {comparison_name}-only correct={mcnemar['comparison_only_correct']}")
        print(f"McNemar p       : {mcnemar['mcnemar_p']:.8f} | significant={significance_text(mcnemar['mcnemar_p'], args.alpha)}")
        print(f"Ordinal MAE     : {args.reference_name}={wilcoxon_result['reference_mae']:.6f} | {comparison_name}={wilcoxon_result['comparison_mae']:.6f}")
        print(f"Error reduction : {wilcoxon_result['mean_error_improvement']:+.6f}")
        print(f"Wilcoxon W      : {wilcoxon_result['wilcoxon_statistic']:.6f}")
        print(f"Wilcoxon p      : {wilcoxon_result['wilcoxon_p']:.8f} | significant={significance_text(wilcoxon_result['wilcoxon_p'], args.alpha)}")
        print(f"Nonzero pairs   : {wilcoxon_result['nonzero_pairs']}")

        model_json = {"comparison_file": comparison["_path"], "comparison_seed": comparison["_seed"], "mcnemar": mcnemar, "wilcoxon_ordinal_error": wilcoxon_result, "metrics": {}}
        summary_row = {"reference_model": args.reference_name, "comparison_model": comparison_name, "reference_seed": reference["_seed"], "comparison_seed": comparison["_seed"], "mcnemar_p": mcnemar["mcnemar_p"], "mcnemar_significant": significance_text(mcnemar["mcnemar_p"], args.alpha), "reference_ordinal_mae": wilcoxon_result["reference_mae"], "comparison_ordinal_mae": wilcoxon_result["comparison_mae"], "mean_ordinal_error_improvement": wilcoxon_result["mean_error_improvement"], "wilcoxon_statistic": wilcoxon_result["wilcoxon_statistic"], "wilcoxon_p": wilcoxon_result["wilcoxon_p"], "wilcoxon_significant": significance_text(wilcoxon_result["wilcoxon_p"], args.alpha), "n_test_samples": len(y_true)}

        for metric_index, metric in enumerate(args.metrics):
            result = paired_bootstrap(y_true, reference_pred, comparison_pred, metric, args.bootstrap, args.random_seed + model_index * 1000 + metric_index, args.alternative, class_labels, args.bootstrap_mode)
            model_json["metrics"][metric] = result
            summary_row[f"{metric}_reference_score"] = result["reference_score"]
            summary_row[f"{metric}_comparison_score"] = result["comparison_score"]
            summary_row[f"{metric}_difference"] = result["difference"]
            summary_row[f"{metric}_ci_low_95"] = result["ci_low_95"]
            summary_row[f"{metric}_ci_high_95"] = result["ci_high_95"]
            summary_row[f"{metric}_bootstrap_p"] = result["bootstrap_p"]
            summary_row[f"{metric}_significant"] = significance_text(result["bootstrap_p"], args.alpha)
            detailed_rows.append({"reference_model": args.reference_name, "comparison_model": comparison_name, "reference_seed": reference["_seed"], "comparison_seed": comparison["_seed"], "metric": metric, "reference_score": result["reference_score"], "comparison_score": result["comparison_score"], "difference_reference_minus_comparison": result["difference"], "ci_low_95": result["ci_low_95"], "ci_high_95": result["ci_high_95"], "bootstrap_p": result["bootstrap_p"], "bootstrap_significant": significance_text(result["bootstrap_p"], args.alpha), "mcnemar_p": mcnemar["mcnemar_p"], "mcnemar_significant": significance_text(mcnemar["mcnemar_p"], args.alpha), "reference_ordinal_mae": wilcoxon_result["reference_mae"], "comparison_ordinal_mae": wilcoxon_result["comparison_mae"], "mean_ordinal_error_improvement": wilcoxon_result["mean_error_improvement"], "wilcoxon_statistic": wilcoxon_result["wilcoxon_statistic"], "wilcoxon_p": wilcoxon_result["wilcoxon_p"], "wilcoxon_significant": significance_text(wilcoxon_result["wilcoxon_p"], args.alpha), "wilcoxon_nonzero_pairs": wilcoxon_result["nonzero_pairs"], "reference_only_correct": mcnemar["reference_only_correct"], "comparison_only_correct": mcnemar["comparison_only_correct"], "n_test_samples": len(y_true), "valid_bootstrap_samples": result["valid_bootstrap_samples"]})
            print(f"{metric:18s} ref={result['reference_score']:.6f} | comparison={result['comparison_score']:.6f} | delta={result['difference']:+.6f} | 95% CI=[{result['ci_low_95']:+.6f}, {result['ci_high_95']:+.6f}] | p={result['bootstrap_p']:.8f} | significant={significance_text(result['bootstrap_p'], args.alpha)}")

        summary_rows.append(summary_row)
        json_results["comparisons"][comparison_name] = model_json

    detailed_csv_path = output_dir / "paired_statistical_results.csv"
    summary_csv_path = output_dir / "paired_statistical_summary.csv"
    json_path = output_dir / "paired_statistical_results.json"
    save_csv(detailed_rows, detailed_csv_path)
    save_csv(summary_rows, summary_csv_path)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(json_results, file, indent=2)

    print_final_summary(summary_rows, args.alpha)

    print("\n" + "=" * 96)
    print(f"Detailed CSV : {detailed_csv_path}")
    print(f"Summary CSV  : {summary_csv_path}")
    print(f"JSON         : {json_path}")
    print("=" * 96)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)