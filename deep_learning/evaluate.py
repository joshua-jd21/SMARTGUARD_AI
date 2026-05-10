# =============================================================
#  SmartGuard AI — CNN Evaluation Suite (FINAL)
#  File: deep_learning/evaluate.py
#
#  FINAL IMPROVEMENTS INCLUDED:
#  -------------------------------------------------------------
#  ✅ Raw + Normalized confusion matrices
#  ✅ ROC curves for all classes
#  ✅ Macro-average ROC AUC
#  ✅ Confidence distribution analysis
#  ✅ Per-class metrics visualization
#  ✅ JSON summary export
#  ✅ TXT classification report export
#  ✅ Safe normalization (no divide-by-zero)
#  ✅ Path-safe saving using pathlib
#  ✅ Production-ready standalone evaluation pipeline
# =============================================================

import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from loguru import logger

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support
)

from sklearn.preprocessing import label_binarize

# =============================================================
# PROJECT ROOT
# =============================================================

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

# =============================================================
# SETTINGS
# =============================================================

from config.settings import (
    PROCESSED_DATA_DIR,
    CNN_MODEL_PATH,
    MODEL_DIR,
    RISK_LABELS,
    RISK_COLORS
)

from deep_learning.cnn_model import load_cnn_model

# =============================================================
# PATHS
# =============================================================

PLOTS_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(PROCESSED_DATA_DIR)

MODEL_DIR_PATH = Path(MODEL_DIR)

MODEL_DIR_PATH.mkdir(
    parents=True,
    exist_ok=True
)

# =============================================================
# GLOBALS
# =============================================================

N_CLASSES = len(RISK_LABELS)

CLASS_NAMES = [
    RISK_LABELS[i]
    for i in range(N_CLASSES)
]

COLORS = [
    RISK_COLORS[label]
    for label in CLASS_NAMES
]

# =============================================================
# LOAD TEST DATA
# =============================================================

def load_test_data():

    logger.info(
        f"Loading test data from:\n{DATA_DIR}"
    )

    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_test shape: {y_test.shape}")

    unique, counts = np.unique(
        y_test,
        return_counts=True
    )

    logger.info("Test class distribution:")

    for cls, count in zip(unique, counts):
        logger.info(
            f"  {RISK_LABELS[int(cls)]:8s}: {count:,}"
        )

    return X_test, y_test


# =============================================================
# CONFUSION MATRIX
# =============================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    save_path: Path
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    # SAFE NORMALIZATION
    cm_norm = cm.astype(float) / (
        cm.sum(axis=1, keepdims=True) + 1e-8
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    # =========================================================
    # RAW COUNTS
    # =========================================================

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        ax=axes[0]
    )

    axes[0].set_title(
        "Confusion Matrix (Raw Counts)"
    )

    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # =========================================================
    # NORMALIZED
    # =========================================================

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        vmin=0,
        vmax=1,
        ax=axes[1]
    )

    axes[1].set_title(
        "Confusion Matrix (Normalized)"
    )

    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.suptitle(
        "SmartGuard AI — CNN Confusion Matrix",
        fontsize=13
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    logger.success(
        f"Confusion matrix saved:\n{save_path}"
    )


# =============================================================
# ROC CURVES + MACRO AUC
# =============================================================

def plot_roc_curves(
    y_true,
    y_pred_probs,
    save_path: Path
):

    y_bin = label_binarize(
        y_true,
        classes=list(range(N_CLASSES))
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    auc_scores = {}

    # =========================================================
    # PER-CLASS ROC
    # =========================================================

    for i, (cls_name, color) in enumerate(
        zip(CLASS_NAMES, COLORS)
    ):

        fpr, tpr, _ = roc_curve(
            y_bin[:, i],
            y_pred_probs[:, i]
        )

        roc_auc = auc(fpr, tpr)

        auc_scores[cls_name] = round(
            float(roc_auc),
            4
        )

        axes[0].plot(
            fpr,
            tpr,
            lw=2,
            color=color,
            label=f"{cls_name} (AUC = {roc_auc:.3f})"
        )

    axes[0].plot(
        [0, 1],
        [0, 1],
        "k--",
        lw=1,
        label="Random"
    )

    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.05])

    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")

    axes[0].set_title(
        "ROC Curves — Per Class"
    )

    axes[0].legend(
        loc="lower right",
        fontsize=8
    )

    axes[0].grid(alpha=0.3)

    # =========================================================
    # MACRO AUC
    # =========================================================

    macro_auc = roc_auc_score(
        y_bin,
        y_pred_probs,
        multi_class="ovr",
        average="macro"
    )

    auc_scores["MACRO_AVG"] = round(
        float(macro_auc),
        4
    )

    bars = axes[1].bar(
        CLASS_NAMES,
        [auc_scores[c] for c in CLASS_NAMES],
        color=COLORS
    )

    for bar, score in zip(
        bars,
        [auc_scores[c] for c in CLASS_NAMES]
    ):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            fontsize=10,
            fontweight="bold"
        )

    axes[1].axhline(
        y=0.9,
        color="green",
        linestyle="--",
        alpha=0.6,
        label="Excellent"
    )

    axes[1].set_ylim([0, 1.05])

    axes[1].set_ylabel("AUC Score")

    axes[1].set_title(
        f"AUC Scores\nMacro AUC = {macro_auc:.4f}"
    )

    axes[1].legend()

    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle(
        "SmartGuard AI — ROC & AUC Analysis",
        fontsize=13
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    logger.success(
        f"ROC curves saved:\n{save_path}"
    )

    return auc_scores


# =============================================================
# CONFIDENCE DISTRIBUTION
# =============================================================

def plot_confidence_distribution(
    y_true,
    y_pred,
    y_pred_probs,
    save_path: Path
):

    confidences = np.max(
        y_pred_probs,
        axis=1
    )

    correct_mask = (
        y_pred == y_true
    )

    correct_conf = confidences[correct_mask]

    incorrect_conf = confidences[
        ~correct_mask
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    # =========================================================
    # HISTOGRAM
    # =========================================================

    axes[0].hist(
        correct_conf,
        bins=30,
        alpha=0.7,
        color="#27AE60",
        label=f"Correct ({correct_mask.sum():,})"
    )

    axes[0].hist(
        incorrect_conf,
        bins=30,
        alpha=0.7,
        color="#E74C3C",
        label=f"Incorrect ({(~correct_mask).sum():,})"
    )

    axes[0].set_title(
        "Prediction Confidence Distribution"
    )

    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Predictions")

    axes[0].legend()

    axes[0].grid(alpha=0.3)

    # =========================================================
    # BOX PLOTS
    # =========================================================

    class_confidences = [
        confidences[y_true == cls]
        for cls in range(N_CLASSES)
    ]

    bp = axes[1].boxplot(
        class_confidences,
        labels=CLASS_NAMES,
        patch_artist=True,
        medianprops=dict(
            color="white",
            linewidth=2
        )
    )

    for patch, color in zip(
        bp["boxes"],
        COLORS
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    axes[1].set_title(
        "Confidence by Risk Class"
    )

    axes[1].set_ylabel("Confidence")

    axes[1].set_ylim([0, 1.05])

    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle(
        "SmartGuard AI — Confidence Analysis",
        fontsize=13
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    logger.success(
        f"Confidence distribution saved:\n{save_path}"
    )


# =============================================================
# PER-CLASS METRICS
# =============================================================

def plot_per_class_metrics(
    y_true,
    y_pred,
    save_path: Path
):

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(N_CLASSES)),
            zero_division=0
        )
    )

    x = np.arange(N_CLASSES)

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    bars1 = ax.bar(
        x - width,
        precision,
        width,
        label="Precision"
    )

    bars2 = ax.bar(
        x,
        recall,
        width,
        label="Recall"
    )

    bars3 = ax.bar(
        x + width,
        f1,
        width,
        label="F1"
    )

    for bars in [bars1, bars2, bars3]:

        for bar in bars:

            h = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.2f}",
                ha="center",
                fontsize=8
            )

    ax.set_xticks(x)

    ax.set_xticklabels(CLASS_NAMES)

    ax.set_ylim([0, 1.1])

    ax.set_ylabel("Score")

    ax.set_title(
        "Precision / Recall / F1 per Class"
    )

    ax.legend()

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    logger.success(
        f"Per-class metrics saved:\n{save_path}"
    )


# =============================================================
# SAVE REPORT TXT
# =============================================================

def save_classification_report_txt(
    y_true,
    y_pred,
    save_path: Path
):

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    with open(save_path, "w") as f:
        f.write(report)

    logger.success(
        f"Classification report saved:\n{save_path}"
    )


# =============================================================
# SAVE JSON SUMMARY
# =============================================================

def save_summary(
    y_true,
    y_pred,
    auc_scores,
    save_path: Path
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(N_CLASSES)),
            zero_division=0
        )
    )

    summary = {

        "model": "CNN_Risk_Classifier",

        "overall_accuracy": round(
            float(accuracy),
            4
        ),

        "macro_auc": auc_scores.get(
            "MACRO_AVG",
            0.0
        ),

        "per_class": {

            RISK_LABELS[i]: {

                "precision": round(
                    float(precision[i]),
                    4
                ),

                "recall": round(
                    float(recall[i]),
                    4
                ),

                "f1_score": round(
                    float(f1[i]),
                    4
                ),

                "support": int(
                    support[i]
                ),

                "auc": auc_scores.get(
                    RISK_LABELS[i],
                    0.0
                )

            }

            for i in range(N_CLASSES)
        }
    }

    with open(save_path, "w") as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    logger.success(
        f"Evaluation summary saved:\n{save_path}"
    )

    return summary


# =============================================================
# MAIN EVALUATION PIPELINE
# =============================================================

def run_evaluation():

    logger.info("=" * 60)
    logger.info(" SmartGuard AI — CNN Evaluation Suite")
    logger.info("=" * 60)

    # =========================================================
    # CHECK MODEL
    # =========================================================

    if not Path(CNN_MODEL_PATH).exists():

        logger.error(
            f"Model not found:\n{CNN_MODEL_PATH}"
        )

        logger.error(
            "Run cnn_model.py first."
        )

        return

    # =========================================================
    # LOAD MODEL + DATA
    # =========================================================

    model = load_cnn_model()

    X_test, y_test = load_test_data()

    # =========================================================
    # PREDICTIONS
    # =========================================================

    logger.info(
        "Running predictions..."
    )

    y_pred_probs = model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(
        y_pred_probs,
        axis=1
    )

    # =========================================================
    # PRINT REPORT
    # =========================================================

    print("\n" + "=" * 60)
    print(" CNN RISK CLASSIFIER — FULL EVALUATION")
    print("=" * 60)

    print(classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    # =========================================================
    # SAVE TXT REPORT
    # =========================================================

    save_classification_report_txt(
        y_test,
        y_pred,
        MODEL_DIR_PATH / "classification_report.txt"
    )

    # =========================================================
    # PLOTS
    # =========================================================

    plot_confusion_matrix(
        y_test,
        y_pred,
        PLOTS_DIR / "eval_confusion_matrix.png"
    )

    auc_scores = plot_roc_curves(
        y_test,
        y_pred_probs,
        PLOTS_DIR / "eval_roc_curves.png"
    )

    plot_confidence_distribution(
        y_test,
        y_pred,
        y_pred_probs,
        PLOTS_DIR / "eval_confidence_distribution.png"
    )

    plot_per_class_metrics(
        y_test,
        y_pred,
        PLOTS_DIR / "eval_per_class_metrics.png"
    )

    # =========================================================
    # SAVE SUMMARY JSON
    # =========================================================

    summary = save_summary(
        y_test,
        y_pred,
        auc_scores,
        MODEL_DIR_PATH / "cnn_eval_summary.json"
    )

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    print("\n" + "=" * 55)
    print(" EVALUATION SUMMARY")
    print("=" * 55)

    print(
        f" Overall Accuracy : "
        f"{summary['overall_accuracy']:.4f}"
    )

    print(
        f" Macro AUC        : "
        f"{summary['macro_auc']:.4f}"
    )

    print("=" * 55)

    logger.success(
        "\nEvaluation complete."
    )

    logger.success(
        f"Plots saved to:\n{PLOTS_DIR}"
    )

    logger.success(
        f"Reports saved to:\n{MODEL_DIR_PATH}"
    )

    print("\nGenerated files:")

    files = [
        "eval_confusion_matrix.png",
        "eval_roc_curves.png",
        "eval_confidence_distribution.png",
        "eval_per_class_metrics.png",
        "classification_report.txt",
        "cnn_eval_summary.json"
    ]

    for f in files:
        print(f"  • {f}")


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    run_evaluation()