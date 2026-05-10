# =============================================================
#  SmartGuard AI — Unified Prediction Interface
#  File: deep_learning/predict.py
#
#  PURPOSE:
#  -------------------------------------------------------------
#  Single inference interface combining:
#
#  1. CNN Risk Classifier
#     → Predicts LOW / MEDIUM / HIGH / CRITICAL
#
#  2. LSTM Autoencoder
#     → Detects anomalous physiological behaviour
#
#  FINAL FEATURES:
#  -------------------------------------------------------------
#  ✅ Unified prediction pipeline
#  ✅ Loads scaler + CNN + LSTM automatically
#  ✅ Thread-safe lazy asset loading
#  ✅ Feature-order protection
#  ✅ Safe padding using edge padding
#  ✅ Confidence-gated fail-safe system
#  ✅ Anomaly-aware recommendation engine
#  ✅ Production-ready structure
#  ✅ Demo runner included
#
#  COMMAND:
#      python deep_learning/predict.py
# =============================================================

import sys
import threading
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from loguru import logger

# -------------------------------------------------------------
# PROJECT ROOT
# -------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# -------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------

from config.settings import (
    SEQUENCE_LENGTH,
    N_FEATURES,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    RISK_LABELS,
    COVERAGE_RECOMMENDATIONS
)

# -------------------------------------------------------------
# MODEL IMPORTS
# -------------------------------------------------------------

from deep_learning.cnn_model import (
    load_cnn_model,
    predict_risk
)

from deep_learning.lstm_model import (
    load_lstm_model,
    predict_anomaly
)

# -------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------

FEATURE_COLS = [
    "heart_rate",
    "spo2",
    "temperature"
]

# Safety thresholds
CONFIDENCE_GATE = 0.40
HIGH_ANOMALY_GATE = 0.75

# -------------------------------------------------------------
# GLOBAL CACHE
# -------------------------------------------------------------

_ASSETS_LOADED = False
_ASSET_LOCK = threading.Lock()

CNN_MODEL = None
LSTM_MODEL = None
SCALER = None
ANOMALY_THRESHOLD = None


# =============================================================
# LOAD ASSETS
# =============================================================

def load_assets():
    """
    Thread-safe lazy loading of all inference assets.

    Loads:
    - CNN classifier
    - LSTM autoencoder
    - MinMaxScaler
    - LSTM anomaly threshold
    """

    global _ASSETS_LOADED
    global CNN_MODEL
    global LSTM_MODEL
    global SCALER
    global ANOMALY_THRESHOLD

    if _ASSETS_LOADED:
        return

    with _ASSET_LOCK:

        if _ASSETS_LOADED:
            return

        logger.info("=" * 60)
        logger.info("Loading SmartGuard AI inference assets...")
        logger.info("=" * 60)

        # -----------------------------------------------------
        # PATHS
        # -----------------------------------------------------

        processed_dir = Path(PROCESSED_DATA_DIR)
        model_dir = Path(MODEL_DIR)

        scaler_path = processed_dir / "scaler.pkl"
        threshold_path = model_dir / "lstm_threshold.npy"

        # -----------------------------------------------------
        # LOAD MODELS
        # -----------------------------------------------------

        logger.info("Loading LSTM autoencoder...")
        LSTM_MODEL = load_lstm_model()

        logger.info("Loading CNN classifier...")
        CNN_MODEL = load_cnn_model()

        # -----------------------------------------------------
        # LOAD SCALER
        # -----------------------------------------------------

        logger.info("Loading MinMaxScaler...")
        SCALER = joblib.load(scaler_path)

        # -----------------------------------------------------
        # LOAD THRESHOLD
        # -----------------------------------------------------

        logger.info("Loading anomaly threshold...")
        ANOMALY_THRESHOLD = float(
            np.load(threshold_path)[0]
        )

        _ASSETS_LOADED = True

        logger.success(
            "All inference assets loaded successfully."
        )


# =============================================================
# WINDOW PREPARATION
# =============================================================

def prepare_window(
    readings: List[Dict],
    scaler,
    sequence_length: int = SEQUENCE_LENGTH
) -> np.ndarray:
    """
    Convert raw sensor readings into a model-ready window.

    Steps:
    1. Extract features
    2. Scale using training scaler
    3. Pad if shorter than SEQUENCE_LENGTH
    4. Return shape:
         (1, sequence_length, N_FEATURES)

    Args:
        readings:
            List of sensor dictionaries

        scaler:
            Fitted MinMaxScaler

        sequence_length:
            Required window size

    Returns:
        np.ndarray
    """

    if len(readings) == 0:
        raise ValueError("No readings provided.")

    # ---------------------------------------------------------
    # FEATURE EXTRACTION
    # ---------------------------------------------------------

    features_array = np.array([
        [
            reading["heart_rate"],
            reading["spo2"],
            reading["temperature"]
        ]
        for reading in readings
    ], dtype=np.float32)

    # ---------------------------------------------------------
    # SCALE
    # Use DataFrame to preserve feature names/order
    # ---------------------------------------------------------

    scaled = scaler.transform(
        pd.DataFrame(
            features_array,
            columns=FEATURE_COLS
        )
    )

    # ---------------------------------------------------------
    # EDGE PADDING
    # Better than repeating zeros
    # ---------------------------------------------------------

    if len(scaled) < sequence_length:

        padding_count = (
            sequence_length - len(scaled)
        )

        scaled = np.pad(
            scaled,
            ((padding_count, 0), (0, 0)),
            mode="edge"
        )

    # ---------------------------------------------------------
    # TRIM TO LATEST WINDOW
    # ---------------------------------------------------------

    scaled = scaled[-sequence_length:]

    # ---------------------------------------------------------
    # FINAL SHAPE
    # ---------------------------------------------------------

    window = scaled.reshape(
        1,
        sequence_length,
        N_FEATURES
    ).astype(np.float32)

    return window


# =============================================================
# RECOMMENDATION ENGINE
# =============================================================

def generate_recommendation(
    risk_label: str,
    anomaly_score: float,
    is_anomaly: bool
) -> str:
    """
    Generate human-readable recommendation.

    Uses:
    - CNN risk class
    - LSTM anomaly score

    Higher anomaly score increases urgency.
    """

    # ---------------------------------------------------------
    # LOW
    # ---------------------------------------------------------

    if risk_label == "LOW":

        if is_anomaly:
            return (
                "Vitals mostly stable but minor irregular "
                "patterns detected. Continue monitoring."
            )

        return (
            "Patient vitals appear stable. "
            "No immediate action required."
        )

    # ---------------------------------------------------------
    # MEDIUM
    # ---------------------------------------------------------

    if risk_label == "MEDIUM":

        if anomaly_score >= 0.75:
            return (
                "Elevated risk with highly irregular patterns "
                "detected. Early clinical review recommended."
            )

        return (
            "Moderate physiological risk detected. "
            "Preventive health evaluation advised."
        )

    # ---------------------------------------------------------
    # HIGH
    # ---------------------------------------------------------

    if risk_label == "HIGH":

        if anomaly_score >= 0.75:
            return (
                "High-risk vitals with severe abnormal patterns "
                "detected. Immediate medical attention advised."
            )

        return (
            "High-risk vitals detected. "
            "Clinical attention advised as soon as possible."
        )

    # ---------------------------------------------------------
    # CRITICAL
    # ---------------------------------------------------------

    return (
        "Critical physiological instability detected. "
        "Emergency intervention strongly recommended."
    )


# =============================================================
# FAILSAFE LOGIC
# =============================================================

def apply_failsafe(
    cnn_result: Dict,
    anomaly_result: Dict
) -> Dict:
    """
    Safety override system.

    If:
    - CNN confidence is low
    AND
    - LSTM anomaly score is very high

    Then elevate risk level for safety.
    """

    confidence = cnn_result["confidence"]
    anomaly_score = anomaly_result["anomaly_score"]

    risk_class = cnn_result["risk_class"]
    risk_label = cnn_result["risk_label"]

    # ---------------------------------------------------------
    # FAILSAFE TRIGGER
    # ---------------------------------------------------------

    if (
        confidence < CONFIDENCE_GATE
        and anomaly_score > HIGH_ANOMALY_GATE
    ):

        logger.warning(
            "Failsafe triggered: "
            "low CNN confidence + high anomaly score."
        )

        elevated_class = min(risk_class + 1, 3)

        cnn_result["risk_class"] = elevated_class
        cnn_result["risk_label"] = (
            RISK_LABELS[elevated_class]
        )

        cnn_result["failsafe_triggered"] = True

    else:

        cnn_result["failsafe_triggered"] = False

    return cnn_result


# =============================================================
# MAIN PREDICTION PIPELINE
# =============================================================

def predict_patient_status(
    readings: List[Dict]
) -> Dict:
    """
    Unified prediction pipeline.

    Args:
        readings:
            List of recent sensor readings

    Returns:
        Dictionary containing:
        - CNN prediction
        - LSTM anomaly detection
        - Recommendation
    """

    load_assets()

    # ---------------------------------------------------------
    # PREPARE WINDOW
    # ---------------------------------------------------------

    window = prepare_window(
        readings,
        scaler=SCALER
    )

    # ---------------------------------------------------------
    # CNN RISK CLASSIFICATION
    # ---------------------------------------------------------

    cnn_result = predict_risk(
        CNN_MODEL,
        window
    )

    # ---------------------------------------------------------
    # LSTM ANOMALY DETECTION
    # ---------------------------------------------------------

    anomaly_result = predict_anomaly(
        LSTM_MODEL,
        window,
        threshold=ANOMALY_THRESHOLD
    )

    # ---------------------------------------------------------
    # FAILSAFE
    # ---------------------------------------------------------

    cnn_result = apply_failsafe(
        cnn_result,
        anomaly_result
    )

    # ---------------------------------------------------------
    # RECOMMENDATION
    # ---------------------------------------------------------

    recommendation = generate_recommendation(
        risk_label=cnn_result["risk_label"],
        anomaly_score=anomaly_result["anomaly_score"],
        is_anomaly=anomaly_result["is_anomaly"]
    )

    # ---------------------------------------------------------
    # INSURANCE RECOMMENDATIONS
    # ---------------------------------------------------------

    coverage = COVERAGE_RECOMMENDATIONS[
        cnn_result["risk_label"]
    ]

    # ---------------------------------------------------------
    # FINAL OUTPUT
    # ---------------------------------------------------------

    result = {

        "risk_prediction": cnn_result,

        "anomaly_detection": anomaly_result,

        "recommendation": recommendation,

        "coverage_recommendations": coverage
    }

    return result


# =============================================================
# DEMO
# =============================================================

def run_demo():
    """
    Demo prediction pipeline using synthetic readings.
    """

    logger.info("=" * 60)
    logger.info("  SmartGuard AI — Unified Prediction Demo")
    logger.info("=" * 60)

    # ---------------------------------------------------------
    # SAMPLE HIGH-RISK READINGS
    # ---------------------------------------------------------

    sample_readings = [

        {
            "heart_rate": 130,
            "spo2": 89,
            "temperature": 101.8
        }

        for _ in range(30)
    ]

    # ---------------------------------------------------------
    # PREDICT
    # ---------------------------------------------------------

    result = predict_patient_status(
        sample_readings
    )

    cnn = result["risk_prediction"]
    anomaly = result["anomaly_detection"]

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("  SMARTGUARD AI — PREDICTION RESULT")
    print("=" * 60)

    print("\n[ CNN Risk Classification ]")
    print(f"Risk Level : {cnn['risk_label']}")
    print(f"Confidence : {cnn['confidence']:.4f}")

    print("\n[ LSTM Anomaly Detection ]")
    print(f"Anomaly Detected : {anomaly['is_anomaly']}")
    print(f"Anomaly Score    : {anomaly['anomaly_score']:.4f}")

    print("\n[ Recommendation ]")
    print(result["recommendation"])

    print("\n[ Coverage Recommendations ]")
    for rec in result["coverage_recommendations"]:
        print(f"• {rec}")

    print("\n" + "=" * 60)


# =============================================================
# PUBLIC API
# =============================================================

__all__ = [
    "predict_patient_status",
    "prepare_window",
    "load_assets"
]


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    run_demo()