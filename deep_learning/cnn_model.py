# =============================================================
#  SmartGuard AI — CNN Risk Classifier (FINAL)
#  File: deep_learning/cnn_model.py
#
#  FINAL IMPROVEMENTS INCLUDED:
#  -------------------------------------------------------------
#  ✅ Anti-overfitting architecture
#  ✅ Proper class weighting
#  ✅ Sparse categorical loss (memory efficient)
#  ✅ Flexible model loading
#  ✅ Safe plot saving using Path objects
#  ✅ Better kernel progression (3 → 5 → 9)
#  ✅ ELU activations (better gradient flow)
#  ✅ GlobalAveragePooling instead of Flatten
#  ✅ Production-safe prediction API
#  ✅ Cleaner evaluation pipeline
# =============================================================

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from loguru import logger

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import (
    Model,
    load_model
)

from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Dense,
    Dropout,
    BatchNormalization,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Activation
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from tensorflow.keras.optimizers import Adam

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
    SEQUENCE_LENGTH,
    N_FEATURES,
    BATCH_SIZE,
    EPOCHS_CNN,
    LEARNING_RATE,
    CNN_MODEL_PATH,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RISK_LABELS
)

# =============================================================
# REPRODUCIBILITY
# =============================================================

tf.random.set_seed(42)
np.random.seed(42)

# =============================================================
# GLOBALS
# =============================================================

N_CLASSES = len(RISK_LABELS)

MODEL_DIR = Path(MODEL_DIR)
PROCESSED_DATA_DIR = Path(PROCESSED_DATA_DIR)
CNN_MODEL_PATH = Path(CNN_MODEL_PATH)

PLOTS_DIR = Path(__file__).resolve().parent

# =============================================================
# BUILD CNN MODEL
# =============================================================

def build_cnn_classifier(
    sequence_length: int = SEQUENCE_LENGTH,
    n_features: int = N_FEATURES,
    n_classes: int = N_CLASSES,
    learning_rate: float = LEARNING_RATE
) -> Model:
    """
    Build 1D CNN for multi-class risk classification.

    Architecture:
        Conv1D(64, 3)
        Conv1D(128, 5)
        Conv1D(256, 9)
    """

    inputs = Input(
        shape=(sequence_length, n_features),
        name="vitals_input"
    )

    # =========================================================
    # BLOCK 1 — SHORT PATTERNS
    # =========================================================

    x = Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        name="conv1"
    )(inputs)

    x = BatchNormalization(name="bn1")(x)
    x = Activation("elu", name="elu1")(x)

    x = MaxPooling1D(
        pool_size=2,
        name="pool1"
    )(x)

    x = Dropout(
        0.2,
        name="drop1"
    )(x)

    # =========================================================
    # BLOCK 2 — MEDIUM PATTERNS
    # =========================================================

    x = Conv1D(
        filters=128,
        kernel_size=5,
        padding="same",
        name="conv2"
    )(x)

    x = BatchNormalization(name="bn2")(x)
    x = Activation("elu", name="elu2")(x)

    x = MaxPooling1D(
        pool_size=2,
        name="pool2"
    )(x)

    x = Dropout(
        0.25,
        name="drop2"
    )(x)

    # =========================================================
    # BLOCK 3 — LONG-RANGE TRENDS
    # =========================================================

    x = Conv1D(
        filters=256,
        kernel_size=9,
        padding="same",
        name="conv3"
    )(x)

    x = BatchNormalization(name="bn3")(x)
    x = Activation("elu", name="elu3")(x)

    x = Dropout(
        0.3,
        name="drop3"
    )(x)

    # =========================================================
    # GLOBAL POOLING
    # =========================================================

    x = GlobalAveragePooling1D(
        name="global_avg_pool"
    )(x)

    # =========================================================
    # DENSE HEAD
    # =========================================================

    x = Dense(
        128,
        activation="elu",
        name="dense1"
    )(x)

    x = Dropout(
        0.35,
        name="drop4"
    )(x)

    x = Dense(
        64,
        activation="elu",
        name="dense2"
    )(x)

    x = Dropout(
        0.25,
        name="drop5"
    )(x)

    # =========================================================
    # OUTPUT
    # =========================================================

    outputs = Dense(
        n_classes,
        activation="softmax",
        name="risk_output"
    )(x)

    # =========================================================
    # MODEL
    # =========================================================

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_Risk_Classifier"
    )

    # =========================================================
    # COMPILE
    # =========================================================

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),

        # Better memory efficiency than categorical_crossentropy
        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    logger.success(
        f"CNN built successfully "
        f"({model.count_params():,} params)"
    )

    return model


# =============================================================
# LOAD DATA
# =============================================================

def load_data():

    logger.info(
        f"Loading processed data from:\n"
        f"{PROCESSED_DATA_DIR}"
    )

    X_train = np.load(PROCESSED_DATA_DIR / "X_train.npy")
    X_val   = np.load(PROCESSED_DATA_DIR / "X_val.npy")
    X_test  = np.load(PROCESSED_DATA_DIR / "X_test.npy")

    y_train = np.load(PROCESSED_DATA_DIR / "y_train.npy")
    y_val   = np.load(PROCESSED_DATA_DIR / "y_val.npy")
    y_test  = np.load(PROCESSED_DATA_DIR / "y_test.npy")

    logger.info(f"X_train: {X_train.shape}")
    logger.info(f"X_val  : {X_val.shape}")
    logger.info(f"X_test : {X_test.shape}")

    unique, counts = np.unique(y_train, return_counts=True)

    logger.info("Training distribution:")

    for cls, count in zip(unique, counts):
        logger.info(
            f"  {RISK_LABELS[int(cls)]:8s}: {count:,}"
        )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# =============================================================
# CLASS WEIGHTS
# =============================================================

def compute_weights(y_train: np.ndarray) -> dict:

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    weight_dict = dict(
        zip(classes.tolist(), weights.tolist())
    )

    logger.info("Class weights:")

    for cls, weight in weight_dict.items():
        logger.info(
            f"  {RISK_LABELS[int(cls)]}: {weight:.4f}"
        )

    return weight_dict


# =============================================================
# CALLBACKS
# =============================================================

def get_callbacks(
    model_path: Path = CNN_MODEL_PATH
):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    return [

        EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            mode="max",
            verbose=1
        ),

        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]


# =============================================================
# EVALUATION
# =============================================================

def evaluate(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray
):

    logger.info("Evaluating CNN classifier...")

    y_pred_probs = model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(
        y_pred_probs,
        axis=1
    )

    label_names = [
        RISK_LABELS[i]
        for i in range(N_CLASSES)
    ]

    print("\n" + "=" * 60)
    print("  CNN RISK CLASSIFIER — EVALUATION")
    print("=" * 60)

    print(classification_report(
        y_test,
        y_pred,
        target_names=label_names,
        zero_division=0
    ))

    # =========================================================
    # CONFUSION MATRIX
    # =========================================================

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names
    )

    plt.title(
        "CNN Risk Classifier — Confusion Matrix"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    save_path = PLOTS_DIR / "cnn_confusion_matrix.png"

    plt.savefig(
        save_path,
        dpi=150
    )

    plt.close()

    logger.success(
        f"Confusion matrix saved:\n{save_path}"
    )

    return y_pred, y_pred_probs


# =============================================================
# PLOT TRAINING HISTORY
# =============================================================

def plot_history(history):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4)
    )

    # =========================================================
    # ACCURACY
    # =========================================================

    axes[0].plot(
        history.history["accuracy"],
        label="Train Accuracy"
    )

    axes[0].plot(
        history.history["val_accuracy"],
        label="Val Accuracy"
    )

    axes[0].set_title("CNN Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # =========================================================
    # LOSS
    # =========================================================

    axes[1].plot(
        history.history["loss"],
        label="Train Loss"
    )

    axes[1].plot(
        history.history["val_loss"],
        label="Val Loss"
    )

    axes[1].set_title("CNN Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()

    save_path = PLOTS_DIR / "cnn_training_history.png"

    plt.savefig(
        save_path,
        dpi=150
    )

    plt.close()

    logger.success(
        f"Training history saved:\n{save_path}"
    )


# =============================================================
# PUBLIC API
# =============================================================

def load_cnn_model(
    model_path: str = CNN_MODEL_PATH
) -> Model:

    logger.info(
        f"Loading CNN model from:\n{model_path}"
    )

    return load_model(model_path)


def predict_risk(
    model: Model,
    window: np.ndarray
) -> dict:
    """
    Predict patient risk for one sequence window.
    """

    if window.ndim == 2:
        window = window.reshape(
            1,
            window.shape[0],
            window.shape[1]
        )

    probs = model.predict(
        window,
        verbose=0
    )[0]

    risk_class = int(np.argmax(probs))

    return {
        "risk_label": RISK_LABELS[risk_class],
        "risk_class": risk_class,
        "confidence": round(float(probs[risk_class]), 4),

        "all_probabilities": {
            RISK_LABELS[i]: round(float(probs[i]), 4)
            for i in range(N_CLASSES)
        }
    }


# =============================================================
# TRAIN PIPELINE
# =============================================================

def train():

    logger.info("=" * 60)
    logger.info(" SmartGuard AI — CNN Training")
    logger.info("=" * 60)

    # =========================================================
    # LOAD DATA
    # =========================================================

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = load_data()

    # =========================================================
    # CLASS WEIGHTS
    # =========================================================

    class_weights = compute_weights(y_train)

    # =========================================================
    # BUILD MODEL
    # =========================================================

    model = build_cnn_classifier(
        sequence_length=X_train.shape[1],
        n_features=X_train.shape[2]
    )

    model.summary()

    # =========================================================
    # TRAIN
    # =========================================================

    logger.info(
        f"Training on {len(X_train):,} windows..."
    )

    history = model.fit(
        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS_CNN,
        batch_size=BATCH_SIZE,

        class_weight=class_weights,

        callbacks=get_callbacks(),

        shuffle=True,
        verbose=1
    )

    # =========================================================
    # PLOTS
    # =========================================================

    plot_history(history)

    # =========================================================
    # EVALUATION
    # =========================================================

    evaluate(
        model,
        X_test,
        y_test
    )

    logger.success("CNN training completed.")
    logger.success(f"Model saved:\n{CNN_MODEL_PATH}")

    return model


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    train()