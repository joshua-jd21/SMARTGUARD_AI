# =============================================================
#  SmartGuard AI — LSTM Autoencoder (Final Version)
#  File: deep_learning/lstm_model.py
#
#  FINAL FIXES APPLIED:
#  -------------------------------------------------------------
#  ✅ Flexible load_lstm_model(model_path=...)
#  ✅ Safe plot save paths using Path(__file__).resolve()
#  ✅ Anti-leakage compatible
#  ✅ Threshold optimization
#  ✅ Production-ready structure
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
    confusion_matrix,
    f1_score
)

from tensorflow.keras.models import (
    Model,
    load_model
)

from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    RepeatVector,
    TimeDistributed,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from tensorflow.keras.optimizers import Adam

# -------------------------------------------------------------
# PROJECT ROOT
# -------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT_DIR))

# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------

from config.settings import (
    SEQUENCE_LENGTH,
    BATCH_SIZE,
    EPOCHS_LSTM,
    LEARNING_RATE,
    LSTM_MODEL_PATH,
    MODEL_DIR,
    PROCESSED_DATA_DIR
)

# -------------------------------------------------------------
# REPRODUCIBILITY
# -------------------------------------------------------------

tf.random.set_seed(42)
np.random.seed(42)

# =============================================================
# BUILD MODEL
# =============================================================

def build_lstm_autoencoder(
    sequence_length: int,
    n_features: int,
    learning_rate: float = LEARNING_RATE
) -> Model:

    inputs = Input(
        shape=(sequence_length, n_features),
        name="input_sequence"
    )

    # ---------------------------------------------------------
    # ENCODER
    # ---------------------------------------------------------

    x = LSTM(
        128,
        activation="tanh",
        return_sequences=True,
        name="encoder_lstm_1"
    )(inputs)

    x = Dropout(0.2)(x)

    encoded = LSTM(
        64,
        activation="tanh",
        return_sequences=False,
        name="encoder_lstm_2"
    )(x)

    encoded = Dropout(0.2)(encoded)

    # ---------------------------------------------------------
    # BOTTLENECK
    # ---------------------------------------------------------

    x = RepeatVector(sequence_length)(encoded)

    # ---------------------------------------------------------
    # DECODER
    # ---------------------------------------------------------

    x = LSTM(
        64,
        activation="tanh",
        return_sequences=True,
        name="decoder_lstm_1"
    )(x)

    x = Dropout(0.2)(x)

    x = LSTM(
        128,
        activation="tanh",
        return_sequences=True,
        name="decoder_lstm_2"
    )(x)

    x = Dropout(0.2)(x)

    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------

    outputs = TimeDistributed(
        Dense(n_features),
        name="reconstruction_output"
    )(x)

    model = Model(
        inputs,
        outputs,
        name="LSTM_Autoencoder"
    )

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="mae",
        metrics=["mse"]
    )

    logger.success(
        f"LSTM Autoencoder built "
        f"({model.count_params():,} params)"
    )

    return model

# =============================================================
# LOAD DATA
# =============================================================

def load_data():

    data_dir = Path(PROCESSED_DATA_DIR)

    logger.info(
        f"Loading processed data from:\n{data_dir}"
    )

    X_train = np.load(data_dir / "X_train.npy")
    X_val   = np.load(data_dir / "X_val.npy")
    X_test  = np.load(data_dir / "X_test.npy")

    y_train = np.load(data_dir / "y_train.npy")
    y_val   = np.load(data_dir / "y_val.npy")
    y_test  = np.load(data_dir / "y_test.npy")

    # ---------------------------------------------------------
    # NORMAL DATA ONLY
    # ---------------------------------------------------------

    train_normal_mask = y_train <= 1
    val_normal_mask   = y_val <= 1

    X_train_normal = X_train[train_normal_mask]
    X_val_normal   = X_val[val_normal_mask]

    logger.info(f"Train Normal : {X_train_normal.shape}")
    logger.info(f"Val Normal   : {X_val_normal.shape}")
    logger.info(f"Test         : {X_test.shape}")

    return (
        X_train_normal,
        X_val_normal,
        X_val,
        y_val,
        X_test,
        y_test
    )

# =============================================================
# CALLBACKS
# =============================================================

def get_callbacks():

    os.makedirs(
        os.path.dirname(LSTM_MODEL_PATH),
        exist_ok=True
    )

    return [

        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        ModelCheckpoint(
            filepath=LSTM_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
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
# THRESHOLD OPTIMIZATION
# =============================================================

def optimize_threshold(
    model: Model,
    X_val: np.ndarray,
    y_val: np.ndarray
):

    logger.info("Optimizing anomaly threshold...")

    X_pred = model.predict(
        X_val,
        verbose=0
    )

    errors = np.mean(
        np.abs(X_val - X_pred),
        axis=(1, 2)
    )

    y_true = (y_val >= 2).astype(int)

    thresholds = np.linspace(
        errors.min(),
        errors.max(),
        100
    )

    best_threshold = None
    best_f1 = -1

    for threshold in thresholds:

        y_pred = (
            errors > threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    logger.success(
        f"Best Threshold: {best_threshold:.5f}"
    )

    logger.success(
        f"Best F1 Score : {best_f1:.4f}"
    )

    return float(best_threshold)

# =============================================================
# EVALUATION
# =============================================================

def evaluate(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float
):

    logger.info("Evaluating model...")

    X_pred = model.predict(
        X_test,
        verbose=0
    )

    errors = np.mean(
        np.abs(X_test - X_pred),
        axis=(1, 2)
    )

    y_true = (y_test >= 2).astype(int)

    y_pred = (
        errors > threshold
    ).astype(int)

    print("\n" + "=" * 60)
    print("  LSTM AUTOENCODER — EVALUATION")
    print("=" * 60)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Normal",
                "Anomaly"
            ],
            zero_division=0
        )
    )

    # ---------------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"]
    )

    plt.title("LSTM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    confusion_path = (
        PLOTS_DIR / "lstm_confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=150
    )

    plt.close()

    logger.success(
        f"Confusion matrix saved:\n{confusion_path}"
    )

# =============================================================
# TRAINING HISTORY PLOT
# =============================================================

def plot_history(history):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4)
    )

    # ---------------------------------------------------------
    # LOSS
    # ---------------------------------------------------------

    axes[0].plot(
        history.history["loss"],
        label="Train Loss"
    )

    axes[0].plot(
        history.history["val_loss"],
        label="Val Loss"
    )

    axes[0].set_title("MAE Loss")
    axes[0].legend()

    # ---------------------------------------------------------
    # MSE
    # ---------------------------------------------------------

    axes[1].plot(
        history.history["mse"],
        label="Train MSE"
    )

    axes[1].plot(
        history.history["val_mse"],
        label="Val MSE"
    )

    axes[1].set_title("MSE")
    axes[1].legend()

    plt.tight_layout()

    history_path = (
        PLOTS_DIR / "lstm_training_history.png"
    )

    plt.savefig(
        history_path,
        dpi=150
    )

    plt.close()

    logger.success(
        f"Training history saved:\n{history_path}"
    )

# =============================================================
# LOAD MODEL
# =============================================================

def load_lstm_model(
    model_path: str = LSTM_MODEL_PATH
):
    """
    Flexible model loader.

    Allows loading custom checkpoints.
    """

    logger.info(
        f"Loading LSTM model from:\n{model_path}"
    )

    return load_model(model_path)

# =============================================================
# SINGLE WINDOW PREDICTION
# =============================================================

def predict_anomaly(
    model: Model,
    window: np.ndarray,
    threshold: float
):

    if window.ndim == 2:
        window = np.expand_dims(window, axis=0)

    reconstruction = model.predict(
        window,
        verbose=0
    )

    error = float(
        np.mean(
            np.abs(window - reconstruction)
        )
    )

    is_anomaly = error > threshold

    anomaly_score = min(
        error / (threshold * 2),
        1.0
    )

    return {

        "reconstruction_error":
            round(error, 5),

        "is_anomaly":
            bool(is_anomaly),

        "anomaly_score":
            round(anomaly_score, 4),

        "threshold_used":
            round(threshold, 5)
    }

# =============================================================
# TRAINING PIPELINE
# =============================================================

def train():

    logger.info("=" * 60)
    logger.info(
        " SmartGuard AI — LSTM Autoencoder Training "
    )
    logger.info("=" * 60)

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    (
        X_train_normal,
        X_val_normal,
        X_val,
        y_val,
        X_test,
        y_test
    ) = load_data()

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    sequence_length = X_train_normal.shape[1]
    n_features      = X_train_normal.shape[2]

    model = build_lstm_autoencoder(
        sequence_length=sequence_length,
        n_features=n_features
    )

    model.summary()

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    history = model.fit(

        X_train_normal,
        X_train_normal,

        validation_data=(
            X_val_normal,
            X_val_normal
        ),

        epochs=EPOCHS_LSTM,
        batch_size=BATCH_SIZE,

        callbacks=get_callbacks(),

        shuffle=False,

        verbose=1
    )

    # ---------------------------------------------------------
    # PLOT HISTORY
    # ---------------------------------------------------------

    plot_history(history)

    # ---------------------------------------------------------
    # THRESHOLD
    # ---------------------------------------------------------

    threshold = optimize_threshold(
        model,
        X_val,
        y_val
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    threshold_path = (
        Path(MODEL_DIR)
        / "lstm_threshold.npy"
    )

    np.save(
        threshold_path,
        np.array([threshold])
    )

    logger.success(
        f"Threshold saved:\n{threshold_path}"
    )

    # ---------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------

    evaluate(
        model,
        X_test,
        y_test,
        threshold
    )

    logger.success(
        "LSTM training completed successfully."
    )

    return model, threshold

# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    train()