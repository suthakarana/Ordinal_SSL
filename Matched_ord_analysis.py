import argparse
import importlib.util
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
SHORT_NAMES = ["NoDR", "Mild", "Mod", "Sev", "PDR"]


 

class MLP(nn.Module):
    def __init__(self, dim_in, out_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim_in, dim_in), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(dim_in, out_dim))

    def forward(self, x):
        return self.layers(x)


class MyResNet(nn.Module):
    def __init__(self, num_classes=5, useBothEyes=False, model_name="resnet18"):
        super().__init__()
        self.useBothEyes = useBothEyes
        self.model_name = model_name

        if model_name == "resnet18":
            resnet = models.resnet18(weights=None)
        elif model_name == "resnet50":
            resnet = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        nfea = resnet.fc.in_features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        feature_dim = nfea * (2 if useBothEyes else 1)
        self.dout = nn.Dropout(p=0.2)
        self.mse_feat = MLP(feature_dim, feature_dim, dropout=0.1)
        self.ce_feat = MLP(feature_dim, feature_dim, dropout=0.1)
        self.ce_w_feat = MLP(feature_dim, feature_dim, dropout=0.1)
        self.fc_mse = nn.Linear(feature_dim, 1)
        self.fc_ce = nn.Linear(feature_dim, num_classes)
        self.fc_ce_w = nn.Linear(feature_dim, num_classes)

    def extract(self, x):
        return torch.flatten(self.backbone(x), 1)

    def forward(self, x1, x2=None):
        x1 = self.extract(x1)

        if self.useBothEyes and x2 is not None:
            x2 = self.extract(x2)
            raw_feats = torch.cat((x1, (x1 + x2) / 2), dim=1)
        else:
            raw_feats = x1

        raw_feats = self.dout(raw_feats)
        ce_out = self.fc_ce(self.ce_feat(raw_feats))
        ce_w_out = self.fc_ce_w(self.ce_w_feat(raw_feats))
        mse_out = self.fc_mse(self.mse_feat(raw_feats))

        return ce_out, ce_w_out, mse_out


 

class IndexedTestDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.fnArr = base_dataset.fnArr
        self.lblArr = base_dataset.lblArr

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, target = self.base_dataset[index]
        return image, int(target), int(index)

 

def import_dataloader_module(dataloader_file):
    dataloader_file = Path(dataloader_file).resolve()

    if not dataloader_file.is_file():
        raise FileNotFoundError(f"Dataloader file not found: {dataloader_file}")

    spec = importlib.util.spec_from_file_location("eyepacs_training_dataloader", dataloader_file)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import dataloader file: {dataloader_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for required_name in ("get_datasetsDR", "DataSetDR"):
        if not hasattr(module, required_name):
            raise AttributeError(f"The dataloader file does not define {required_name}.")

    return module


def extract_normalization(transform_test):
    if isinstance(transform_test, transforms.Normalize):
        return list(transform_test.mean), list(transform_test.std)

    if hasattr(transform_test, "transforms"):
        for transform_step in transform_test.transforms:
            if isinstance(transform_step, transforms.Normalize):
                return list(transform_step.mean), list(transform_step.std)

    return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]


def denormalize_batch(batch_tensor, mean, std):
    mean_tensor = torch.tensor(mean, dtype=batch_tensor.dtype, device=batch_tensor.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=batch_tensor.dtype, device=batch_tensor.device).view(1, 3, 1, 1)
    return torch.clamp(batch_tensor * std_tensor + mean_tensor, 0.0, 1.0)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    for key in ("state_dict", "model_state_dict", "model", "net", "network", "ema_state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], nn.Module):
            return checkpoint[key].state_dict()
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint

    raise KeyError("No model state dictionary was found in the checkpoint.")


def strip_common_prefixes(state_dict):
    cleaned = dict(state_dict)
    prefixes = ("module.", "model.", "network.", "student.", "ema_model.")
    changed = True

    while changed:
        changed = False
        for prefix in prefixes:
            if cleaned and all(key.startswith(prefix) for key in cleaned):
                cleaned = {key[len(prefix):]: value for key, value in cleaned.items()}
                changed = True

    return cleaned


def load_checkpoint(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(strip_common_prefixes(extract_state_dict(checkpoint)), strict=True)
    return model


def resolve_device(device_text):
    text = str(device_text).strip()
    text = f"cuda:{text}" if text.isdigit() else text

    if text.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(text)


def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


 

def calculate_quality_batch(display_batch):
    focus_values, brightness_values, contrast_values = [], [], []

    for image_tensor in display_batch:
        image_rgb = image_tensor.permute(1, 2, 0).detach().cpu().numpy()
        image_uint8 = np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)

        focus_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        brightness_values.append(float(gray.mean() / 255.0))
        contrast_values.append(float(gray.std() / 255.0))

    return focus_values, brightness_values, contrast_values


 

def ordinal_stats(probs, tau):
    probs = np.asarray(probs, dtype=np.float64)
    class_ids = np.arange(len(probs), dtype=np.float64)
    prediction = int(np.argmax(probs))
    expected = float(np.sum(probs * class_ids))
    distances = np.abs(class_ids - expected)
    weights = np.exp(-distances / float(tau))
    contributions = probs * weights
    confidence = float(np.sum(contributions))
    maxprob = float(np.max(probs))

    return {"prediction": prediction, "expected": expected, "distances": distances, "weights": weights, "contributions": contributions, "confidence": confidence, "maxprob": maxprob}


@torch.no_grad()
def model_probabilities(model, images):
    ce_logits, ce_w_logits, _ = model(images)
    ce_probs = torch.softmax(ce_logits, dim=1)
    ce_w_probs = torch.softmax(ce_w_logits, dim=1)
    aggregate_probs = 0.5 * (ce_probs + ce_w_probs)
    return ce_probs, ce_w_probs, aggregate_probs

 

def scan_test_set(test_dataset, model, device, batch_size, workers, mean, std, tau):
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    model.eval()
    rows = []

    print(f"\n[1/4] Scanning {len(test_dataset)} EyePACS test images...")

    with torch.inference_mode():
        progress_bar = tqdm(loader, total=len(loader), desc="Probability-space inference", unit="batch", dynamic_ncols=True)

        for images, labels, indices in progress_bar:
            display_batch = denormalize_batch(images, mean, std)
            focus_values, brightness_values, contrast_values = calculate_quality_batch(display_batch)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            ce_probs, ce_w_probs, aggregate_probs = model_probabilities(model, images)
            predictions = aggregate_probs.argmax(dim=1)

            ce_probs_np = ce_probs.detach().cpu().numpy()
            ce_w_probs_np = ce_w_probs.detach().cpu().numpy()
            aggregate_probs_np = aggregate_probs.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            progress_bar.set_postfix(acc=f"{(predictions == labels).float().mean().item() * 100.0:.1f}%")

            for b in range(images.shape[0]):
                dataset_index = int(indices[b].item())
                label = int(labels_np[b])
                ce_p = ce_probs_np[b]
                ce_w_p = ce_w_probs_np[b]
                aggregate_p = aggregate_probs_np[b]
                prediction = int(np.argmax(aggregate_p))
                stats = ordinal_stats(aggregate_p, tau)
                image_path = str(test_dataset.fnArr[dataset_index])

                row = {"dataset_index": dataset_index, "image_id": Path(image_path).stem, "image_path": image_path, "label": label, "class_name": CLASS_NAMES[label], "prediction": prediction, "prediction_name": CLASS_NAMES[prediction], "ordinal_error": abs(prediction - label), "correct": int(prediction == label), "maxprob": stats["maxprob"], "expected_grade": stats["expected"], "orddist": stats["confidence"], "focus": focus_values[b], "brightness": brightness_values[b], "contrast": contrast_values[b]}

                for k in range(len(CLASS_NAMES)):
                    row[f"ce_p{k}"] = float(ce_p[k])
                    row[f"cew_p{k}"] = float(ce_w_p[k])
                    row[f"p{k}"] = float(aggregate_p[k])
                    row[f"weight{k}"] = float(stats["weights"][k])
                    row[f"contribution{k}"] = float(stats["contributions"][k])

                rows.append(row)

    frame = pd.DataFrame(rows)
    frame["focus_rank"] = frame.groupby("label")["focus"].rank(method="average", pct=True)
    frame["contrast_rank"] = frame.groupby("label")["contrast"].rank(method="average", pct=True)
    frame["brightness_score"] = (1.0 - (frame["brightness"] - 0.45).abs() / 0.45).clip(lower=0.0, upper=1.0)
    frame["quality_score"] = 0.50 * frame["focus_rank"] + 0.25 * frame["contrast_rank"] + 0.25 * frame["brightness_score"]
    frame["error_group"] = np.where(frame["ordinal_error"] == 0, "Correct", np.where(frame["ordinal_error"] == 1, "Adjacent", "Distant"))

    print(f"[1/4] Finished scanning {len(frame)} images.")
    return frame

 

def find_matched_pair(pool, tolerance, same_prediction=True):
    best_pair = None
    best_score = -np.inf

    for i, row_a in pool.iterrows():
        candidates = pool.drop(index=i).copy()

        if same_prediction:
            candidates = candidates[candidates["prediction"] == int(row_a["prediction"])].copy()

        if candidates.empty:
            continue

        candidates["maxprob_gap_tmp"] = np.abs(candidates["maxprob"] - float(row_a["maxprob"]))
        candidates = candidates[candidates["maxprob_gap_tmp"] <= float(tolerance)].copy()

        if candidates.empty:
            continue

        candidates["orddist_gap_tmp"] = np.abs(candidates["orddist"] - float(row_a["orddist"]))
        candidates["quality_tmp"] = (candidates["quality_score"] + float(row_a["quality_score"])) / 2.0
        candidates["pair_score_tmp"] = candidates["orddist_gap_tmp"] - 4.0 * candidates["maxprob_gap_tmp"] + 0.02 * candidates["quality_tmp"]

        row_b = candidates.sort_values(["pair_score_tmp", "orddist_gap_tmp", "maxprob_gap_tmp"], ascending=[False, False, True]).iloc[0]

        if float(row_b["pair_score_tmp"]) > best_score:
            best_score = float(row_b["pair_score_tmp"])
            best_pair = row_a.copy(), row_b.copy()

    if best_pair is None:
        return None

    row_a, row_b = best_pair

    if float(row_a["orddist"]) >= float(row_b["orddist"]):
        return row_a.copy(), row_b.copy()

    return row_b.copy(), row_a.copy()


def select_panel_a_pair(frame, maxprob_tol=0.02, min_maxprob=0.55, max_maxprob=0.90, top_n=1500):
    pool = frame[(frame["maxprob"] >= float(min_maxprob)) & (frame["maxprob"] <= float(max_maxprob))].copy()
    pool = pool.sort_values(["quality_score", "dataset_index"], ascending=[False, True]).head(int(top_n)).copy()

    print(f"\n[2/4] Matched-example pool: {len(pool)} samples")

    pair = None
    same_prediction = True
    used_tolerance = None

    for tolerance in sorted(set([float(maxprob_tol), 0.03, 0.04, 0.05, 0.08, 0.10])):
        pair = find_matched_pair(pool, tolerance, same_prediction=True)
        if pair is not None:
            used_tolerance = tolerance
            break

    if pair is None:
        same_prediction = False

        for tolerance in sorted(set([float(maxprob_tol), 0.03, 0.04, 0.05, 0.08, 0.10])):
            pair = find_matched_pair(pool, tolerance, same_prediction=False)
            if pair is not None:
                used_tolerance = tolerance
                break

    if pair is None:
        pool = frame.sort_values(["quality_score", "dataset_index"], ascending=[False, True]).head(int(top_n)).copy()
        same_prediction = True

        for tolerance in (float(maxprob_tol), 0.03, 0.04, 0.05, 0.08, 0.10):
            pair = find_matched_pair(pool, tolerance, same_prediction=True)
            if pair is not None:
                used_tolerance = tolerance
                break

    if pair is None:
        same_prediction = False
        pair = find_matched_pair(pool, 1.0, same_prediction=False)
        used_tolerance = 1.0

    if pair is None:
        raise RuntimeError("Unable to select two usable matched-MaxProb examples.")

    high_row, low_row = pair
    maxprob_gap = abs(float(high_row["maxprob"]) - float(low_row["maxprob"]))
    orddist_gap = float(high_row["orddist"]) - float(low_row["orddist"])

    print("=" * 105)
    print("MATCHED-MAXPROB EXAMPLES")
    print("=" * 105)
    print(f"Same predicted class    : {same_prediction}")
    print(f"Tolerance used          : {used_tolerance:.3f}")
    print(f"Higher coherence        : {high_row['image_id']}")
    print(f"Lower coherence         : {low_row['image_id']}")
    print(f"MaxProb                 : {float(high_row['maxprob']):.4f} vs {float(low_row['maxprob']):.4f}")
    print(f"Delta MaxProb           : {maxprob_gap:.4f}")
    print(f"OrdDist                 : {float(high_row['orddist']):.4f} vs {float(low_row['orddist']):.4f}")
    print(f"Delta OrdDist           : {orddist_gap:.4f}")
    print("=" * 105)

    return high_row, low_row


 

def spearman_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 2:
        return np.nan

    rx = pd.Series(x[valid]).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y[valid]).rank(method="average").to_numpy(dtype=np.float64)

    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan

    return float(np.corrcoef(rx, ry)[0, 1])


def mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, 0

    mean = float(np.mean(values))

    if len(values) == 1:
        return mean, 0.0, 1

    ci95 = float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, ci95, len(values)


def confidence_summary_by_error(frame, metric):
    rows = []

    for distance in range(len(CLASS_NAMES)):
        mean, ci95, n = mean_ci95(frame.loc[frame["ordinal_error"] == distance, metric].values)
        rows.append({"distance": distance, "mean": mean, "ci95": ci95, "n": n})

    return pd.DataFrame(rows)


 

def build_matched_maxprob_bins(frame, min_bin_count=5):
    edges = np.asarray([0.50, 0.60, 0.70, 0.80, 0.90, 1.000001], dtype=np.float64)
    labels = ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]

    work = frame.copy()
    work["maxprob_bin"] = pd.cut(work["maxprob"], bins=edges, labels=labels, include_lowest=True, right=False)

    rows = []

    for bin_index, label in enumerate(labels):
        current = work[work["maxprob_bin"] == label].copy()

        for group_name, mask in (("near", current["ordinal_error"] <= 1), ("distant", current["ordinal_error"] >= 2)):
            values = current.loc[mask, "orddist"].to_numpy(dtype=np.float64)
            mean, ci95, n = mean_ci95(values)
            rows.append({"bin_index": bin_index, "maxprob_bin": label, "group": group_name, "mean_orddist": mean, "ci95": ci95, "n": n, "display": int(n >= int(min_bin_count))})

    result = pd.DataFrame(rows)

    if len(result[(result["group"] == "distant") & (result["display"] == 1)]) < 2 and min_bin_count > 2:
        print(f"Panel D: fewer than two distant-error bins have n >= {min_bin_count}; visualization will use n >= 2.")
        result["display"] = (result["n"] >= 2).astype(int)

    return result


def build_bin_difference_table(matched_bins):
    rows = []

    for bin_index in sorted(matched_bins["bin_index"].unique()):
        current = matched_bins[matched_bins["bin_index"] == bin_index]
        near = current[current["group"] == "near"]
        distant = current[current["group"] == "distant"]

        if near.empty or distant.empty:
            continue

        near_mean = float(near.iloc[0]["mean_orddist"])
        distant_mean = float(distant.iloc[0]["mean_orddist"])
        rows.append({"bin_index": int(bin_index), "maxprob_bin": str(near.iloc[0]["maxprob_bin"]), "near_mean_orddist": near_mean, "distant_mean_orddist": distant_mean, "delta_orddist": near_mean - distant_mean, "near_n": int(near.iloc[0]["n"]), "distant_n": int(distant.iloc[0]["n"])})

    return pd.DataFrame(rows)


 

def get_display_image(test_dataset, index, mean, std):
    image_tensor, _, _ = test_dataset[int(index)]
    display = denormalize_batch(image_tensor.unsqueeze(0), mean, std)[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return np.clip(display, 0.0, 1.0)


 

def plot_example_row(fig, gs, row_index, row, title, test_dataset, mean, std, tau, show_legend):
    image_ax = fig.add_subplot(gs[row_index, 0])
    prob_ax = fig.add_subplot(gs[row_index, 1])
    contribution_ax = fig.add_subplot(gs[row_index, 2])
    info_ax = fig.add_subplot(gs[row_index, 3])

    image = get_display_image(test_dataset, row["dataset_index"], mean, std)
    probs = np.asarray([row[f"p{k}"] for k in range(len(CLASS_NAMES))], dtype=np.float64)
    stats = ordinal_stats(probs, tau)
    x = np.arange(len(CLASS_NAMES))

    image_ax.imshow(image, interpolation="lanczos")
    image_ax.axis("off")
    image_ax.set_title(title, fontsize=11.2, fontweight="bold", pad=6)

    prob_ax.bar(x, probs, width=0.62)
    prob_ax.axvline(stats["expected"], linestyle="--", linewidth=1.5)
    prob_ax.set_ylim(0.0, 1.05)
    prob_ax.set_ylabel("Probability", fontsize=9.5)
    prob_ax.set_xticks(x)
    prob_ax.set_xticklabels(SHORT_NAMES, fontsize=8.5)
    prob_ax.set_title("Class probability distribution", fontsize=10.2)
    prob_ax.grid(True, axis="y", alpha=0.18)

    for xi, yi in zip(x, probs):
        if yi >= 0.015:
            prob_ax.text(xi, yi + 0.025, f"{yi:.2f}", ha="center", va="bottom", fontsize=8.0)

    contribution_ax.bar(x, stats["contributions"], width=0.62, label="Weighted contribution")
    contribution_ax.set_ylim(0.0, 1.05)
    contribution_ax.set_ylabel("Contribution", fontsize=9.5)
    contribution_ax.set_xticks(x)
    contribution_ax.set_xticklabels(SHORT_NAMES, fontsize=8.5)
    contribution_ax.set_title("Distance-weighted OrdDist contribution", fontsize=10.2)
    contribution_ax.grid(True, axis="y", alpha=0.18)

    weight_ax = contribution_ax.twinx()
    weight_ax.plot(x, stats["weights"], marker="o", linestyle="--", linewidth=1.5, label="Distance weight")
    weight_ax.set_ylim(0.0, 1.05)
    weight_ax.set_ylabel("Weight", fontsize=9.0)

    if show_legend:
        handles1, labels1 = contribution_ax.get_legend_handles_labels()
        handles2, labels2 = weight_ax.get_legend_handles_labels()
        contribution_ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=7.8, frameon=True)

    info_ax.axis("off")
    info_ax.text(0.02, 0.82, f"GT: {CLASS_NAMES[int(row['label'])]}", ha="left", va="center", fontsize=9.5)
    info_ax.text(0.02, 0.65, f"Pred: {CLASS_NAMES[int(row['prediction'])]}", ha="left", va="center", fontsize=9.5)
    info_ax.text(0.02, 0.48, rf"$d={int(row['ordinal_error'])}$", ha="left", va="center", fontsize=9.5)
    info_ax.text(0.02, 0.31, f"MaxProb = {stats['maxprob']:.3f}", ha="left", va="center", fontsize=9.5)
    info_ax.text(0.02, 0.14, rf"$\mu={stats['expected']:.2f}$   OrdDist = {stats['confidence']:.3f}", ha="left", va="center", fontsize=9.5)

    return stats


def make_figure_a(frame, test_dataset, mean, std, tau, output_dir, maxprob_tol, panel_a_min_maxprob, panel_a_max_maxprob):
    output_dir = Path(output_dir)

    high_row, low_row = select_panel_a_pair(frame, maxprob_tol=maxprob_tol, min_maxprob=panel_a_min_maxprob, max_maxprob=panel_a_max_maxprob)

    examples = pd.DataFrame([high_row, low_row])
    examples["role"] = ["higher_ordinal_coherence", "lower_ordinal_coherence"]
    examples.to_csv(output_dir / "matched_ordinal_coherence_examples.csv", index=False)

    fig = plt.figure(figsize=(15.2, 7.0))
    gs = fig.add_gridspec(2, 4, width_ratios=[0.72, 1.42, 1.48, 0.72], hspace=0.42, wspace=0.34)

    high_stats = plot_example_row(fig, gs, 0, high_row, "Higher ordinal coherence", test_dataset, mean, std, tau, True)
    low_stats = plot_example_row(fig, gs, 1, low_row, "Lower ordinal coherence", test_dataset, mean, std, tau, False)

    fig.suptitle("Matched-MaxProb examples with contrasting ordinal coherence", fontsize=14.0, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.91, bottom=0.065)

    png_path = output_dir / "matched_ordinal_coherence.png"
    pdf_path = output_dir / "matched_ordinal_coherence.pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    maxprob_gap = abs(high_stats["maxprob"] - low_stats["maxprob"])
    orddist_gap = high_stats["confidence"] - low_stats["confidence"]

    print(f"\nSaved matched-example PNG : {png_path}")
    print(f"Saved matched-example PDF : {pdf_path}")
    print(f"Delta MaxProb             : {maxprob_gap:.4f}")
    print(f"Delta OrdDist             : {orddist_gap:.4f}")

 

def make_population_figure(frame, output_dir, min_bin_count):
    output_dir = Path(output_dir)

    confidence_max = confidence_summary_by_error(frame, "maxprob")
    confidence_ord = confidence_summary_by_error(frame, "orddist")
    confidence_max["metric"] = "MaxProb"
    confidence_ord["metric"] = "OrdDist"

    confidence_summary = pd.concat([confidence_max, confidence_ord], ignore_index=True)
    confidence_summary.to_csv(output_dir / "confidence_vs_ordinal_error.csv", index=False)

    rho_max = spearman_corr(frame["ordinal_error"].values, frame["maxprob"].values)
    rho_ord = spearman_corr(frame["ordinal_error"].values, frame["orddist"].values)

    pd.DataFrame([{"metric": "MaxProb", "spearman_rho": rho_max}, {"metric": "OrdDist", "spearman_rho": rho_ord}]).to_csv(output_dir / "spearman_confidence_vs_error.csv", index=False)

    matched_bins = build_matched_maxprob_bins(frame, min_bin_count=min_bin_count)
    matched_bins.to_csv(output_dir / "matched_maxprob_bin_analysis.csv", index=False)

    bin_difference = build_bin_difference_table(matched_bins)
    bin_difference.to_csv(output_dir / "matched_maxprob_bin_differences.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.45))
    axB, axC, axD = axes

 

    for summary, label in ((confidence_max, "MaxProb"), (confidence_ord, "OrdDist")):
        valid = summary["n"] > 0
        axB.errorbar(summary.loc[valid, "distance"], summary.loc[valid, "mean"], yerr=summary.loc[valid, "ci95"], marker="o", linewidth=2.0, capsize=3.5, label=label)

    axB.set_title("Confidence vs ordinal error", fontsize=11.2, fontweight="bold")
    axB.set_xlabel(r"Ordinal error distance $|\hat{y}-y|$", fontsize=9.6)
    axB.set_ylabel("Confidence", fontsize=9.6)
    axB.set_xticks(np.arange(len(CLASS_NAMES)))
    axB.set_ylim(0.0, 1.02)
    axB.grid(True, alpha=0.20)
    axB.legend(frameon=True, fontsize=8.7)
    axB.text(0.04, 0.045, r"Mean $\pm$ 95% CI", transform=axB.transAxes, fontsize=8.1)

 
    group_conditions = [frame["ordinal_error"] == 0, frame["ordinal_error"] == 1, frame["ordinal_error"] >= 2]
    group_labels = ["Correct\n" + r"($d=0$)", "Adjacent\n" + r"($d=1$)", "Distant\n" + r"($d\geq2$)"]

    max_data = [frame.loc[condition, "maxprob"].dropna().to_numpy(dtype=np.float64) for condition in group_conditions]
    ord_data = [frame.loc[condition, "orddist"].dropna().to_numpy(dtype=np.float64) for condition in group_conditions]

    max_data = [values if len(values) > 0 else np.asarray([np.nan]) for values in max_data]
    ord_data = [values if len(values) > 0 else np.asarray([np.nan]) for values in ord_data]

    positions_max = np.asarray([0.82, 1.82, 2.82])
    positions_ord = np.asarray([1.18, 2.18, 3.18])

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_max = colors[0]
    color_ord = colors[1]

    bp_max = axC.boxplot(max_data, positions=positions_max, widths=0.30, patch_artist=True, showfliers=False, medianprops={"linewidth": 1.5}, whiskerprops={"linewidth": 1.0}, capprops={"linewidth": 1.0})
    bp_ord = axC.boxplot(ord_data, positions=positions_ord, widths=0.30, patch_artist=True, showfliers=False, medianprops={"linewidth": 1.5}, whiskerprops={"linewidth": 1.0}, capprops={"linewidth": 1.0})

    for box in bp_max["boxes"]:
        box.set_facecolor(color_max)
        box.set_alpha(0.42)

    for box in bp_ord["boxes"]:
        box.set_facecolor(color_ord)
        box.set_alpha(0.42)

    for element in ("medians", "whiskers", "caps"):
        for artist in bp_max[element]:
            artist.set_color(color_max)
        for artist in bp_ord[element]:
            artist.set_color(color_ord)

    axC.set_title("Confidence by ordinal-error severity", fontsize=11.2, fontweight="bold")
    axC.set_ylabel("Confidence", fontsize=9.6)
    axC.set_xticks([1.0, 2.0, 3.0])
    axC.set_xticklabels(group_labels, fontsize=8.8)
    axC.set_xlim(0.50, 3.50)
    axC.set_ylim(0.0, 1.02)
    axC.grid(True, axis="y", alpha=0.20)
    axC.legend([bp_max["boxes"][0], bp_ord["boxes"][0]], ["MaxProb", "OrdDist"], frameon=True, fontsize=8.7)

 

    bin_labels = ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]
    x_positions = np.arange(len(bin_labels))

    for group_name, display_label, marker in (("near", r"Correct/adjacent ($d\leq1$)", "o"), ("distant", r"Distant ($d\geq2$)", "s")):
        current = matched_bins[(matched_bins["group"] == group_name) & (matched_bins["display"] == 1)].copy()
        y = np.full(len(bin_labels), np.nan, dtype=np.float64)
        ci = np.full(len(bin_labels), np.nan, dtype=np.float64)

        for _, row in current.iterrows():
            index = int(row["bin_index"])
            y[index] = float(row["mean_orddist"])
            ci[index] = float(row["ci95"])

        valid = np.isfinite(y)
        axD.errorbar(x_positions[valid], y[valid], yerr=ci[valid], marker=marker, linewidth=2.0, capsize=3.5, label=display_label)

    axD.set_title("OrdDist within matched MaxProb bins", fontsize=11.2, fontweight="bold")
    axD.set_xlabel("MaxProb interval", fontsize=9.6)
    axD.set_ylabel("OrdDist confidence", fontsize=9.6)
    axD.set_xticks(x_positions)
    axD.set_xticklabels(bin_labels, rotation=25, ha="right", fontsize=8.3)
    axD.set_ylim(0.0, 1.02)
    axD.grid(True, alpha=0.20)
    axD.legend(frameon=True, fontsize=8.3)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.92, bottom=0.20, wspace=0.30)

    png_path = output_dir / "orddist_population_analysis.png"
    pdf_path = output_dir / "orddist_population_analysis.pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print("\n" + "=" * 110)
    print("POPULATION-LEVEL ANALYSIS")
    print("=" * 110)
    print(f"Spearman rho | MaxProb : {rho_max:.4f}")
    print(f"Spearman rho | OrdDist : {rho_ord:.4f}")

    if not bin_difference.empty:
        print("\nMatched-MaxProb OrdDist separation:")
        print(bin_difference.to_string(index=False))

    print(f"\nSaved population PNG    : {png_path}")
    print(f"Saved population PDF    : {pdf_path}")
    print("=" * 110)

 

def parse_arguments():
    parser = argparse.ArgumentParser(description="EyePACS probability-space analysis of Distance-Weighted Ordinal Concentration.")

    parser.add_argument("--dataloader-file", type=Path, required=True)
    parser.add_argument("--orddist-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", choices=("resnet18", "resnet50"), default="resnet18")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--label-percentage", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ord-tau", type=float, required=True)
    parser.add_argument("--maxprob-match-tol", type=float, default=0.02)
    parser.add_argument("--panel-a-min-maxprob", type=float, default=0.55)
    parser.add_argument("--panel-a-max-maxprob", type=float, default=0.90)
    parser.add_argument("--panel-d-min-bin-count", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():
    args = parse_arguments()

    if args.num_classes != len(CLASS_NAMES):
        raise ValueError(f"The script defines {len(CLASS_NAMES)} classes but num_classes={args.num_classes}.")

    if args.ord_tau <= 0:
        raise ValueError("--ord-tau must be > 0.")

    if args.maxprob_match_tol <= 0:
        raise ValueError("--maxprob-match-tol must be > 0.")

    if not 0.0 <= args.panel_a_min_maxprob < args.panel_a_max_maxprob <= 1.0:
        raise ValueError("Panel A MaxProb bounds must satisfy 0 <= min < max <= 1.")

    if args.panel_d_min_bin_count < 1:
        raise ValueError("--panel-d-min-bin-count must be >= 1.")

    set_reproducibility(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    print("=" * 110)
    print("EyePACS | OrdDist Probability-Space Analysis")
    print("=" * 110)
    print(f"Model                  : {args.model_name}")
    print("Probability source     : mean(CE probability, CE-W probability)")
    print(f"Label percentage       : {args.label_percentage:.2f}")
    print(f"Seed                   : {args.seed}")
    print(f"Ordinal temperature    : {args.ord_tau:.4f}")
    print(f"Matched MaxProb range  : [{args.panel_a_min_maxprob:.2f}, {args.panel_a_max_maxprob:.2f}]")
    print(f"Preferred MaxProb gap  : <= {args.maxprob_match_tol:.3f}")
    print(f"Device                 : {device}")
    print("=" * 110)

    dataloader_module = import_dataloader_module(args.dataloader_file)
    _, _, _, _, _, _, xte, yte = dataloader_module.get_datasetsDR(args.label_percentage, args.seed)

    base_testset = dataloader_module.DataSetDR("DR", xte, yte, "TEST")
    test_dataset = IndexedTestDataset(base_testset)
    mean, std = extract_normalization(base_testset.transformTest)

    print(f"Loaded test images     : {len(test_dataset)}")

    if device.type == "cuda":
        print(f"GPU                    : {torch.cuda.get_device_name(device)}")

    model = load_checkpoint(MyResNet(num_classes=args.num_classes, useBothEyes=False, model_name=args.model_name), args.orddist_checkpoint).to(device)
    model.eval()

    frame = scan_test_set(test_dataset=test_dataset, model=model, device=device, batch_size=args.batch_size, workers=args.workers, mean=mean, std=std, tau=args.ord_tau)

    prediction_csv = args.output_dir / "orddist_probability_analysis_predictions.csv"
    frame.to_csv(prediction_csv, index=False)

    print(f"\nSaved predictions     : {prediction_csv}")

    make_figure_a(frame=frame, test_dataset=test_dataset, mean=mean, std=std, tau=args.ord_tau, output_dir=args.output_dir, maxprob_tol=args.maxprob_match_tol, panel_a_min_maxprob=args.panel_a_min_maxprob, panel_a_max_maxprob=args.panel_a_max_maxprob)

    print("\n[3/4] Creating population analysis...")

    make_population_figure(frame=frame, output_dir=args.output_dir, min_bin_count=args.panel_d_min_bin_count)

    print("\n[4/4] Analysis completed.")


if __name__ == "__main__":
    main()