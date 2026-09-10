import csv
import math
import os

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


OVERALL_METRIC_NAMES = [
    "Acc",
    "BAcc",
    "QWK",
    "F1_micro",
    "F1_macro",
    "AUC",
    "Precision",
    "Recall",
    "Sensitivity",
    "MCC",
    "Specificity",
]


CLASSWISE_METRIC_NAMES = [
    "Acc",
    "BAcc",
    "F1_micro",
    "F1_macro",
    "AUC",
    "Precision",
    "Recall",
    "Sensitivity",
    "MCC",
    "Specificity",
]


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles=7.0 / 16.0,
    last_epoch=-1,
):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        no_progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def getLinear(probs):
    class_count = probs.size(1)
    class_values = torch.arange(
        class_count,
        device=probs.device,
        dtype=probs.dtype,
    )
    return (probs * class_values).sum(dim=1)


def _resolve_class_labels(gt_cpu, pl_cpu, num_classes=None, probs=None):
    if num_classes is None and probs is not None:
        probs_array = np.asarray(probs)
        if probs_array.ndim == 2:
            num_classes = probs_array.shape[1]

    if num_classes is None:
        maximum_label = -1
        if np.asarray(gt_cpu).size > 0:
            maximum_label = max(maximum_label, int(np.max(gt_cpu)))
        if np.asarray(pl_cpu).size > 0:
            maximum_label = max(maximum_label, int(np.max(pl_cpu)))
        num_classes = maximum_label + 1

    return np.arange(int(num_classes), dtype=np.int64)


def _safe_macro_ovr_auc(gt_cpu, probs, class_labels):

    if probs is None:
        return np.nan

    probs = np.asarray(probs)
    if probs.ndim != 2 or probs.shape[0] != len(gt_cpu):
        return np.nan

    auc_values = []
    for class_id in class_labels:
        class_id = int(class_id)
        if class_id >= probs.shape[1]:
            continue

        binary_target = (gt_cpu == class_id).astype(np.int64)
        if np.unique(binary_target).size < 2:
            continue

        try:
            auc_values.append(
                roc_auc_score(binary_target, probs[:, class_id])
            )
        except ValueError:
            continue

    return float(np.mean(auc_values)) if auc_values else np.nan


def getScore_fromPred(gt_cpu, pred_gpu):
    probs = pred_gpu.detach().cpu().numpy()
    predicted_labels = torch.argmax(pred_gpu, dim=1).detach().cpu().numpy()
    return getScores(
        gt_cpu,
        predicted_labels,
        probs=probs,
        num_classes=pred_gpu.size(1),
    )


def getScores(gt_cpu, pl_cpu, probs=None, num_classes=None):

    gt_cpu = np.asarray(gt_cpu, dtype=np.int64).reshape(-1)
    pl_cpu = np.asarray(pl_cpu, dtype=np.int64).reshape(-1)
    class_labels = _resolve_class_labels( gt_cpu, pl_cpu, num_classes=num_classes, probs=probs)
    acc = accuracy_score(gt_cpu, pl_cpu) * 100.0
    bacc = balanced_accuracy_score(gt_cpu, pl_cpu) * 100.0
    qwk = cohen_kappa_score(gt_cpu,pl_cpu,labels=class_labels,weights="quadratic")
    f1_micro = f1_score(gt_cpu,pl_cpu,labels=class_labels,average="micro",zero_division=0)
    f1_macro = f1_score( gt_cpu, pl_cpu, labels=class_labels,average="macro",zero_division=0)
    precision = precision_score(gt_cpu, pl_cpu,labels=class_labels, average="macro",zero_division=0)
    recall = recall_score(gt_cpu,pl_cpu,labels=class_labels,average="macro", zero_division=0)
    sensitivity = recall
    mcc = matthews_corrcoef(gt_cpu, pl_cpu)

    cm = confusion_matrix(gt_cpu,pl_cpu,labels=class_labels)
    specificity_values = []

    for class_id in range(len(class_labels)):
        tp = cm[class_id, class_id]
        fn = cm[class_id, :].sum() - tp
        fp = cm[:, class_id].sum() - tp
        tn = cm.sum() - tp - fp - fn

        specificity_values.append(
            tn / (tn + fp) if (tn + fp) > 0 else 0.0
        )

    specificity = float(np.mean(specificity_values))
    auc = _safe_macro_ovr_auc(gt_cpu, probs, class_labels)

    return (
        np.round(acc, 4),
        np.round(bacc, 4),
        np.round(qwk, 4),
        np.round(f1_micro, 4),
        np.round(f1_macro, 4),
        np.round(auc, 4),
        np.round(precision, 4),
        np.round(recall, 4),
        np.round(sensitivity, 4),
        np.round(mcc, 4),
        np.round(specificity, 4),
    )


def getClasswiseScores(
    gt_cpu,
    pl_cpu,
    probs=None,
    num_classes=None,
    class_names=None,
):

    gt_cpu = np.asarray(gt_cpu, dtype=np.int64).reshape(-1)
    pl_cpu = np.asarray(pl_cpu, dtype=np.int64).reshape(-1)
    probs_array = None if probs is None else np.asarray(probs)

    class_labels = _resolve_class_labels(gt_cpu,pl_cpu,num_classes=num_classes,probs=probs_array)

    cm = confusion_matrix(gt_cpu,pl_cpu,labels=class_labels)
    total = int(cm.sum())
    rows = []

    for position, class_id in enumerate(class_labels):
        class_id = int(class_id)

        tp = int(cm[position, position])
        fn = int(cm[position, :].sum() - tp)
        fp = int(cm[:, position].sum() - tp)
        tn = int(total - tp - fn - fp)

        support = int(tp + fn)
        predicted = int(tp + fp)

        sensitivity = (
            float(tp / (tp + fn))
            if (tp + fn) > 0
            else 0.0
        )
        specificity = (
            float(tn / (tn + fp))
            if (tn + fp) > 0
            else 0.0
        )
        precision = (
            float(tp / (tp + fp))
            if (tp + fp) > 0
            else 0.0
        )
        f1_class = (
            float(2.0 * precision * sensitivity / (precision + sensitivity))
            if (precision + sensitivity) > 0
            else 0.0
        )

        class_accuracy = (
            float((tp + tn) / total) * 100.0
            if total > 0
            else 0.0
        )
        class_balanced_accuracy = (
            float((sensitivity + specificity) / 2.0) * 100.0
        )

        binary_true = (gt_cpu == class_id).astype(np.int64)
        binary_pred = (pl_cpu == class_id).astype(np.int64)

        f1_micro = f1_score(binary_true,binary_pred,average="micro",zero_division=0)
        f1_macro = f1_score(binary_true,binary_pred,average="macro",zero_division=0)
        mcc = matthews_corrcoef(binary_true, binary_pred)

        auc = np.nan
        if (
            probs_array is not None
            and probs_array.ndim == 2
            and class_id < probs_array.shape[1]
            and np.unique(binary_true).size == 2
        ):
            try:
                auc = roc_auc_score(
                    binary_true,
                    probs_array[:, class_id],
                )
            except ValueError:
                auc = np.nan

        if class_names is not None and position < len(class_names):
            class_name = str(class_names[position])
        else:
            class_name = str(class_id)

        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "support": support,
                "predicted": predicted,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "accuracy": np.round(class_accuracy, 4),
                "balanced_accuracy": np.round(
                    class_balanced_accuracy,
                    4,
                ),
                "f1_micro": np.round(f1_micro, 4),
                "f1_macro": np.round(f1_macro, 4),
                "auc": np.round(auc, 4),
                "precision": np.round(precision, 4),
                "recall": np.round(sensitivity, 4),
                "sensitivity": np.round(sensitivity, 4),
                "mcc": np.round(mcc, 4),
                "specificity": np.round(specificity, 4),
                "f1_class": np.round(f1_class, 4),
            }
        )

    return rows


def save_rows_to_csv(rows, output_path):
    if not rows:
        return

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            clean_row = {}
            for key, value in row.items():
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    clean_row[key] = ""
                else:
                    clean_row[key] = value
            writer.writerow(clean_row)


class DynamicThreshold:
    def __init__(
        self,
        init_threshold=0.8,
        min_t=0.5,
        max_t=0.99,
    ):
        self.threshold = torch.tensor(
            init_threshold,
            dtype=torch.float32,
        )
        self.min_t = min_t
        self.max_t = max_t
        self.initialized = False

    @torch.no_grad()
    def initialize(self, batch_confidences):
        self.threshold = (
            batch_confidences.mean()
            .detach()
            .float()
            .clamp(self.min_t, self.max_t)
        )
        self.initialized = True

    @torch.no_grad()
    def update(self, batch_confidences, momentum):
        batch_mean = batch_confidences.mean().detach().float()
        self.threshold = (
            momentum * self.threshold
            + (1.0 - momentum) * batch_mean
        ).clamp(self.min_t, self.max_t)
        return self.threshold.item()
