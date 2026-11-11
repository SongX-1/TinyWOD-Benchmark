"""Save actual loss/metric/PR/confidence/confusion figures using a headless backend."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from config import CLASS_NAMES
from .metrics import CONFIDENCES


def plot_evaluation(report):
    directory = Path(report["directory"])
    precision = report["precision_tensor"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for c, name in enumerate(CLASS_NAMES):
        if not report["present"][c]:
            continue
        values = precision[0, :, c, 0, -1]
        axes[0].plot(np.linspace(0, 1, len(values)), values, label=name)
        axes[1].plot(CONFIDENCES, report["f1_curve"][c], label=name)
        axes[2].plot(CONFIDENCES, report["p_curve"][c], label=name+" P")
        axes[2].plot(CONFIDENCES, report["r_curve"][c], linestyle="--", label=name+" R")
    for ax, title, xlabel, ylabel in zip(axes, ("COCO PR @ IoU 0.5", "F1-confidence", "Precision / Recall-confidence"),
                                       ("Recall", "Confidence", "Confidence"), ("Precision", "F1", "P / R")):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory/"PR_F1_confidence_curves.png", dpi=160)
    plt.close(fig)
    matrix = report["matrix"]
    for normalize in (False, True):
        data = matrix/np.maximum(matrix.sum(0, keepdims=True), 1) if normalize else matrix
        fig, ax = plt.subplots(figsize=(9, 8))
        image = ax.imshow(data, cmap="Blues")
        names = [*CLASS_NAMES, "background"]
        ax.set(xticks=range(2), yticks=range(2), xticklabels=names, yticklabels=names,
               xlabel="True", ylabel="Predicted", title="Confusion matrix"+(" (column normalized)" if normalize else ""))
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{data[i,j]:.2f}" if normalize else str(matrix[i, j]), ha="center", va="center",
                        color="white" if data[i, j] > data.max()*.5 else "black", fontsize=8)
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(directory/("confusion_matrix_normalized.png" if normalize else "confusion_matrix.png"), dpi=160)
        plt.close(fig)


def plot_training(results_csv, directory):
    with Path(results_csv).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    groups = [
        ("Loss", ["train/box_loss", "train/cls_loss", "train/giou_loss", "val/box_loss", "val/cls_loss", "val/giou_loss"]),
        ("Total objective", ["train/total_loss", "val/total_loss"]),
        ("COCO AP", ["AP50", "AP", "APsmall", "APmedium", "APlarge"]),
        ("Precision / Recall / F1", ["precision", "recall", "f1"]),
        ("Learning rate", [name for name in rows[0] if name.startswith("lr/pg")]),
        ("Time / inference", ["train_seconds", "val_seconds", "inference_ms"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (title, fields) in zip(axes.flat, groups):
        if title == "Time / inference":
            right = ax.twinx()
            for name in ("train_seconds", "val_seconds"):
                ax.plot(epochs, [float(row[name]) for row in rows], marker="o", markersize=3, label=name)
            right.plot(epochs, [float(row["inference_ms"]) for row in rows], color="black", linestyle="--",
                       marker="o", markersize=3, label="inference_ms")
            ax.set_ylabel("Seconds / epoch")
            right.set_ylabel("ms / image")
            left_handles, left_labels = ax.get_legend_handles_labels()
            right_handles, right_labels = right.get_legend_handles_labels()
            ax.legend(left_handles+right_handles, left_labels+right_labels, fontsize=7)
            ax.set(title=title, xlabel="Epoch", xlim=(.5, epochs[-1]+.5))
            if len(epochs) <= 10:
                ax.set_xticks(epochs)
            else:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=.25)
            continue
        for name in fields:
            values = [float(row[name]) if row.get(name) not in (None, "") else np.nan for row in rows]
            if np.isfinite(values).any():
                ax.plot(epochs, values, marker="o", markersize=3, label=name)
        ax.set(title=title, xlabel="Epoch", xlim=(.5, epochs[-1]+.5))
        if title in ("COCO AP", "Precision / Recall / F1"):
            ax.set_ylim(0, 1.02)
        if len(epochs) <= 10:
            ax.set_xticks(epochs)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend(fontsize=7)
    fig.tight_layout()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory/"training_curves.png", dpi=160)
    plt.close(fig)
