# =============================================================
#  SmartGuard AI — Data Preprocessing (Final Anti-Leakage Version)
#  File: deep_learning/preprocess.py
#
#  WHAT THIS FILE DOES:
#  -------------------------------------------------------------
#  1. Generates synthetic patient telemetry using SensorSimulator
#  2. Cleans physiologically impossible readings
#  3. Splits patients into Train / Validation / Test groups
#     BEFORE scaling (prevents data leakage)
#  4. Fits MinMaxScaler ONLY on training data
#  5. Applies same scaler to validation and test
#  6. Creates time-series windows for LSTM/CNN
#  7. Saves processed arrays + scaler to disk
#
#  WHY THIS VERSION IS IMPORTANT:
#  -------------------------------------------------------------
#  • Prevents scaler leakage
#  • Prevents patient leakage
#  • Maintains temporal causality
#  • Works for BOTH:
#       - LSTM Autoencoder
#       - CNN Risk Classifier
#  • Production-style preprocessing pipeline
# =============================================================

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------------------
# ADD PROJECT ROOT TO PATH
# -------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# -------------------------------------------------------------
# IMPORT SETTINGS
# -------------------------------------------------------------

try:
    from config.settings import (
        PROCESSED_DATA_DIR,
        SEQUENCE_LENGTH,
    )

    DATA_PROCESSED_DIR = Path(PROCESSED_DATA_DIR)

except ImportError:

    logger.warning(
        "config.settings not found — using fallback defaults."
    )

    DATA_PROCESSED_DIR = Path("./deep_learning/data/processed")
    SEQUENCE_LENGTH = 30

# -------------------------------------------------------------
# IMPORT SENSOR SIMULATOR
# -------------------------------------------------------------

from iot.sensor_simulator import (
    PatientState,
    SensorSimulator,
)

# -------------------------------------------------------------
# FEATURE + LABEL CONFIGURATION
# -------------------------------------------------------------

FEATURE_COLS = [
    "heart_rate",
    "spo2",
    "temperature",
]

LABEL_COL = "risk_label"

# -------------------------------------------------------------
# STATE → RISK LABEL MAPPING
#
# 0 = LOW
# 1 = MEDIUM
# 2 = HIGH
# 3 = CRITICAL
# -------------------------------------------------------------

STATE_TO_LABEL = {

    PatientState.RESTING: 0,
    PatientState.WALKING: 0,
    PatientState.SLEEPING: 0,
    PatientState.RUNNING: 0,

    PatientState.STRESSED: 1,
    PatientState.ANOMALY_MILD: 1,

    PatientState.ANOMALY_HIGH: 2,

    PatientState.CRITICAL: 3,
}

# =============================================================
# STEP 1 — SYNTHETIC DATA GENERATION
# =============================================================

def generate_synthetic_dataset(
    n_samples: int = 20_000,
    n_patients: int = 10
) -> pd.DataFrame:
    """
    Generate realistic synthetic telemetry data.

    Each patient gets their own SensorSimulator,
    producing independent physiological behavior.

    Returns:
        DataFrame with:
            patient_id
            heart_rate
            spo2
            temperature
            risk_label
    """

    logger.info(
        f"Generating {n_samples:,} synthetic readings "
        f"across {n_patients} patients..."
    )

    rows = []

    readings_per_patient = n_samples // n_patients

    for patient_id in range(n_patients):

        logger.info(
            f"Generating data for Patient {patient_id}"
        )

        simulator = SensorSimulator()

        for _ in range(readings_per_patient):

            # ---------------------------------------------
            # Capture CURRENT STATE before read()
            # because read() may transition internally
            # ---------------------------------------------

            current_state = simulator.state

            reading = simulator.read()

            rows.append({

                "patient_id": patient_id,

                "heart_rate":
                    reading["heart_rate"],

                "spo2":
                    reading["spo2"],

                "temperature":
                    reading["temperature"],

                "risk_label":
                    STATE_TO_LABEL[current_state]
            })

    df = pd.DataFrame(rows)

    logger.success(
        f"Synthetic dataset generated: {len(df):,} rows"
    )

    logger.info(
        f"\nRisk Distribution:\n"
        f"{df['risk_label'].value_counts().sort_index()}"
    )

    return df

# =============================================================
# STEP 2 — CLEAN DATA
# =============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove impossible physiological values.
    """

    before = len(df)

    df = df.dropna(
        subset=FEATURE_COLS + [LABEL_COL]
    )

    # ---------------------------------------------
    # Physiological range validation
    # ---------------------------------------------

    df = df[
        df["heart_rate"].between(30, 220)
    ]

    df = df[
        df["spo2"].between(70, 100)
    ]

    df = df[
        df["temperature"].between(90, 110)
    ]

    after = len(df)

    logger.info(
        f"Cleaning complete: "
        f"{before:,} → {after:,} rows"
    )

    return df.reset_index(drop=True)

# =============================================================
# STEP 3 — PATIENT-WISE SPLIT
#
# Prevents patient leakage.
#
# TRAIN PATIENTS:
#     0–6
#
# VALIDATION:
#     7
#
# TEST:
#     8–9
# =============================================================

def split_by_patient(df: pd.DataFrame):

    train_patients = [0, 1, 2, 3, 4, 5, 6]
    val_patients   = [7]
    test_patients  = [8, 9]

    train_df = df[
        df["patient_id"].isin(train_patients)
    ]

    val_df = df[
        df["patient_id"].isin(val_patients)
    ]

    test_df = df[
        df["patient_id"].isin(test_patients)
    ]

    logger.info(
        "\nPatient-wise split completed:"
    )

    logger.info(
        f"Train Patients: {train_patients}"
    )

    logger.info(
        f"Validation Patients: {val_patients}"
    )

    logger.info(
        f"Test Patients: {test_patients}"
    )

    logger.info(
        f"\nTrain Rows: {len(train_df):,}"
    )

    logger.info(
        f"Validation Rows: {len(val_df):,}"
    )

    logger.info(
        f"Test Rows: {len(test_df):,}"
    )

    return train_df, val_df, test_df

# =============================================================
# STEP 4 — SCALING
#
# IMPORTANT:
# Scaler is fitted ONLY on TRAIN DATA.
#
# This prevents:
#     DATA LEAKAGE
# =============================================================

def scale_data(
    train_df,
    val_df,
    test_df
):

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    # ---------------------------------------------
    # FIT ONLY ON TRAIN
    # ---------------------------------------------

    train_scaled = scaler.fit_transform(
        train_df[FEATURE_COLS]
    )

    # ---------------------------------------------
    # TRANSFORM ONLY
    # ---------------------------------------------

    val_scaled = scaler.transform(
        val_df[FEATURE_COLS]
    )

    test_scaled = scaler.transform(
        test_df[FEATURE_COLS]
    )

    logger.success(
        "Scaler fitted on TRAIN only."
    )

    return (
        train_scaled,
        val_scaled,
        test_scaled,
        scaler
    )

# =============================================================
# STEP 5 — CREATE TIME WINDOWS
#
# Converts flat readings into:
#
# (samples, sequence_length, features)
# =============================================================

def create_windows(
    features: np.ndarray,
    labels: np.ndarray,
    sequence_length: int
):

    X = []
    y = []

    for i in range(
        len(features) - sequence_length
    ):

        window = features[
            i : i + sequence_length
        ]

        label = labels[
            i + sequence_length - 1
        ]

        X.append(window)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y

# =============================================================
# STEP 6 — SAVE ARRAYS
# =============================================================

def save_processed_data(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    scaler
):

    DATA_PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------
    # SAVE NUMPY ARRAYS
    # ---------------------------------------------

    np.save(
        DATA_PROCESSED_DIR / "X_train.npy",
        X_train
    )

    np.save(
        DATA_PROCESSED_DIR / "X_val.npy",
        X_val
    )

    np.save(
        DATA_PROCESSED_DIR / "X_test.npy",
        X_test
    )

    np.save(
        DATA_PROCESSED_DIR / "y_train.npy",
        y_train
    )

    np.save(
        DATA_PROCESSED_DIR / "y_val.npy",
        y_val
    )

    np.save(
        DATA_PROCESSED_DIR / "y_test.npy",
        y_test
    )

    # ---------------------------------------------
    # SAVE SCALER
    # ---------------------------------------------

    joblib.dump(
        scaler,
        DATA_PROCESSED_DIR / "scaler.pkl"
    )

    logger.success(
        f"\nProcessed data saved to:\n"
        f"{DATA_PROCESSED_DIR}"
    )

# =============================================================
# STEP 7 — LOAD PROCESSED DATA
# =============================================================

def load_processed_data():

    X_train = np.load(
        DATA_PROCESSED_DIR / "X_train.npy"
    )

    X_val = np.load(
        DATA_PROCESSED_DIR / "X_val.npy"
    )

    X_test = np.load(
        DATA_PROCESSED_DIR / "X_test.npy"
    )

    y_train = np.load(
        DATA_PROCESSED_DIR / "y_train.npy"
    )

    y_val = np.load(
        DATA_PROCESSED_DIR / "y_val.npy"
    )

    y_test = np.load(
        DATA_PROCESSED_DIR / "y_test.npy"
    )

    scaler = joblib.load(
        DATA_PROCESSED_DIR / "scaler.pkl"
    )

    logger.success(
        "Processed data loaded successfully."
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler
    )

# =============================================================
# FULL PIPELINE
# =============================================================

def run_preprocessing(
    raw_csv: Path = None
):

    # ---------------------------------------------
    # LOAD / GENERATE DATA
    # ---------------------------------------------

    if raw_csv and Path(raw_csv).exists():

        logger.info(
            f"Loading CSV: {raw_csv}"
        )

        df = pd.read_csv(raw_csv)

    else:

        logger.warning(
            "No raw CSV found — generating synthetic data."
        )

        df = generate_synthetic_dataset()

    # ---------------------------------------------
    # CLEAN
    # ---------------------------------------------

    df = clean_data(df)

    # ---------------------------------------------
    # PATIENT-WISE SPLIT
    # ---------------------------------------------

    train_df, val_df, test_df = split_by_patient(df)

    # ---------------------------------------------
    # SCALE
    # ---------------------------------------------

    (
        train_scaled,
        val_scaled,
        test_scaled,
        scaler
    ) = scale_data(
        train_df,
        val_df,
        test_df
    )

    # ---------------------------------------------
    # WINDOWING
    # ---------------------------------------------

    X_train, y_train = create_windows(
        train_scaled,
        train_df[LABEL_COL].values,
        SEQUENCE_LENGTH
    )

    X_val, y_val = create_windows(
        val_scaled,
        val_df[LABEL_COL].values,
        SEQUENCE_LENGTH
    )

    X_test, y_test = create_windows(
        test_scaled,
        test_df[LABEL_COL].values,
        SEQUENCE_LENGTH
    )

    logger.info(
        f"\nWindow Shapes:\n"
        f"X_train: {X_train.shape}\n"
        f"X_val  : {X_val.shape}\n"
        f"X_test : {X_test.shape}"
    )

    # ---------------------------------------------
    # SAVE
    # ---------------------------------------------

    save_processed_data(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler
    )

    logger.success(
        "\nPreprocessing pipeline completed successfully."
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler
    )

# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    run_preprocessing()