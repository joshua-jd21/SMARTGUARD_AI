# =============================================================
#  SmartGuard AI — Central Configuration
#  File: config/settings.py
#
#  PURPOSE:
#  Single source of truth for the entire project.
#  Every other file imports from here.
#  Never hardcode any value anywhere else.
#
#  SETUP:
#  Create a file called ".env" in your project root:
#
#     GOOGLE_API_KEY=your-gemini-key-here
#     SERPAPI_API_KEY=your-serpapi-key-here
#     OPENAI_API_KEY=sk-your-key-here          (optional)
#     ANTHROPIC_API_KEY=your-anthropic-key-here (optional)
#
# =============================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env before any os.getenv() call
load_dotenv()

# =============================================================
#  SECTION 1 — PROJECT PATHS
# =============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Deep learning data directories
DATA_RAW_DIR       = BASE_DIR / "deep_learning" / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "deep_learning" / "data" / "processed"
PROCESSED_DATA_DIR = DATA_PROCESSED_DIR

# Saved model files
MODELS_DIR      = BASE_DIR / "models"
MODEL_DIR       = MODELS_DIR
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.keras"
CNN_MODEL_PATH  = MODELS_DIR / "cnn_model.keras"
RAW_DATA_DIR    = DATA_RAW_DIR

# =============================================================
#  SECTION 2 — API KEYS
# =============================================================

GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
SERPAPI_API_KEY   = os.getenv("SERPAPI_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Alias for compatibility
SERP_API_KEY = SERPAPI_API_KEY

# Startup warnings
if not GOOGLE_API_KEY:
    print("[WARNING] GOOGLE_API_KEY not found in .env — Gemini agents will fail")
if not SERPAPI_API_KEY:
    print("[WARNING] SERPAPI_API_KEY not found in .env — SerpAPI will use fallback results")
if not OPENAI_API_KEY:
    print("[INFO] OPENAI_API_KEY not set (optional — not used by default)")
if not ANTHROPIC_API_KEY:
    print("[INFO] ANTHROPIC_API_KEY not set (optional)")

# =============================================================
#  SECTION 3 — LLM CONFIGURATION
# =============================================================

# PRIMARY LLM: Gemini (used by all 3 agents)
# Default: gemini-2.5-flash — gemini-2.0-flash often hits free-tier quota limit: 0 on new keys/projects.
# Avoid bare "gemini-1.5-flash" (404 on many keys); prefer ids from AI Studio → Rate limits for your tier.
# Override in .env, e.g. GEMINI_MODEL=gemini-2.5-flash-lite
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def _normalize_gemini_model(raw: str) -> str:
    name = (raw or "").strip()
    if name.startswith("models/"):
        name = name[7:]
    return name or _DEFAULT_GEMINI_MODEL


GEMINI_MODEL = _normalize_gemini_model(os.getenv("GEMINI_MODEL", _DEFAULT_GEMINI_MODEL))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# Legacy alias — kept for backward compatibility but NOT passed to Gemini
LLM_MODEL    = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")

AGENT_MAX_ITERATIONS = 5
AGENT_VERBOSE        = True
AGENT_TYPE           = "zero-shot-react-description"

SERP_MAX_RESULTS = 3
MAX_RESULT_CHARS = 2500

# =============================================================
#  SECTION 4 — MQTT CONFIGURATION
# =============================================================

MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "") or os.getenv("MQTT_BROKER", "broker.hivemq.com")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "") or os.getenv("MQTT_PORT", "1883"))
MQTT_USERNAME    = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD    = os.getenv("MQTT_PASSWORD", "")
MQTT_CLIENT_ID   = "smartguard_subscriber"

MQTT_BROKER = MQTT_BROKER_HOST
MQTT_PORT   = MQTT_BROKER_PORT

MQTT_TOPIC         = os.getenv("MQTT_TOPIC", "smartguard/patient/vitals")
MQTT_TOPIC_SENSORS = MQTT_TOPIC
MQTT_TOPIC_RISK    = "smartguard/risk_alert"
MQTT_TOPIC_REPORT  = "smartguard/insurance_report"
MQTT_QOS           = 1
MQTT_RETAIN        = False

# =============================================================
#  SECTION 5 — PATIENT CONFIGURATION
# =============================================================

PATIENT_ID   = os.getenv("PATIENT_ID",   "PATIENT_001")
PATIENT_NAME = os.getenv("PATIENT_NAME", "Rajan")
PATIENT_AGE  = int(os.getenv("PATIENT_AGE", "45"))

BASELINE_HEART_RATE  = 72
BASELINE_SPO2        = 98
BASELINE_TEMPERATURE = 98.6

# =============================================================
#  SECTION 6 — SENSOR SIMULATION
# =============================================================

SENSOR_PUBLISH_INTERVAL_SEC = 1
SENSOR_INTERVAL_SECONDS     = SENSOR_PUBLISH_INTERVAL_SEC

NORMAL_HEART_RATE_RANGE  = (60,   90)
NORMAL_SPO2_RANGE        = (95,  100)
NORMAL_TEMPERATURE_RANGE = (97.5, 99.0)
ACTIVITY_LEVELS          = ["rest", "walk", "run"]

SENSOR_NORMAL_RANGES = {
    "heart_rate":  NORMAL_HEART_RATE_RANGE,
    "spo2":        NORMAL_SPO2_RANGE,
    "temperature": NORMAL_TEMPERATURE_RANGE,
    "activity":    ACTIVITY_LEVELS,
}

ANOMALY_HEART_RATE_RANGE  = (110, 140)
ANOMALY_SPO2_RANGE        = (88,   94)
ANOMALY_TEMPERATURE_RANGE = (99.5, 103.0)
ANOMALY_INJECTION_RATE    = 0.25

# =============================================================
#  SECTION 7 — DEEP LEARNING CONFIGURATION
# =============================================================

SEQUENCE_LENGTH = 30
N_FEATURES      = 3

TRAIN_TEST_SPLIT = 0.2
BATCH_SIZE       = 32
EPOCHS_LSTM      = 30
EPOCHS_CNN       = 30
EPOCHS           = 30
LEARNING_RATE    = 0.001
VALIDATION_SPLIT = 0.2

LSTM_UNITS  = 64
CNN_FILTERS = 32

RISK_LABELS = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH",
    3: "CRITICAL",
}

RISK_COLORS = {
    0:          "#27AE60",
    1:          "#F39C12",
    2:          "#E74C3C",
    3:          "#7B241C",
    "LOW":      "#27AE60",
    "MEDIUM":   "#F39C12",
    "HIGH":     "#E74C3C",
    "CRITICAL": "#7B241C",
}

ANOMALY_THRESHOLD = 0.05

MITBIH_RECORDS = [
    "100", "101", "103", "105", "106",
    "108", "109", "111", "112", "113",
]

# =============================================================
#  SECTION 8 — CLINICAL THRESHOLDS
# =============================================================

THRESHOLDS = {
    "heart_rate": {
        "low":      (30,   60),
        "normal":   (60,  100),
        "elevated": (100, 120),
        "critical": (120, 300),
    },
    "spo2": {
        "normal":   (95, 100),
        "low":      (90,  95),
        "critical": (0,   90),
    },
    "temperature": {
        "normal":   (97.0,  99.5),
        "fever":    (99.5, 103.0),
        "critical": (103.0, 115.0),
    },
}

# =============================================================
#  SECTION 9 — INSURANCE BUSINESS RULES
# =============================================================

PREMIUM_ADJUSTMENTS = {
    "LOW":      0,
    "MEDIUM":   8,
    "HIGH":     18,
    "CRITICAL": 30,
}

CLAIM_PROBABILITIES = {
    "LOW":      5,
    "MEDIUM":   15,
    "HIGH":     30,
    "CRITICAL": 55,
}

COVERAGE_RECOMMENDATIONS = {
    "LOW":      ["Standard coverage — no changes needed"],
    "MEDIUM":   ["Add preventive health check-up rider"],
    "HIGH":     ["Add cardiac cover rider", "Add critical illness rider"],
    "CRITICAL": [
        "Add cardiac cover rider",
        "Add critical illness rider",
        "Add ICU/hospitalisation cover",
        "Enroll in wellness program",
    ],
}

URGENCY_FLOOR = {
    "LOW":      1,
    "MEDIUM":   2,
    "HIGH":     3,
    "CRITICAL": 4,
}

# =============================================================
#  SECTION 10 — AGENT RESEARCH SAFETY
# =============================================================

TRUSTED_SOURCES = {
    "AHA":       5,
    "WHO":       5,
    "CDC":       5,
    "NIH":       5,
    "NEJM":      5,
    "JAMA":      5,
    "Lancet":    5,
    "ESC":       4,
    "ACC":       4,
    "PhysioNet": 4,
}

# =============================================================
#  SECTION 11 — DASHBOARD CONFIGURATION
# =============================================================

DASHBOARD_TITLE               = "SmartGuard AI — Health Insurance Risk Monitor"
DASHBOARD_REFRESH_SEC         = 5
DASHBOARD_REFRESH_INTERVAL_MS = DASHBOARD_REFRESH_SEC * 1000
DASHBOARD_HISTORY_POINTS      = 50
MAX_CHART_POINTS              = DASHBOARD_HISTORY_POINTS
DASHBOARD_PORT                = 8501

# =============================================================
#  CONFIGURATION CHECK  (run directly to verify setup)
# =============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  SmartGuard AI — Configuration Check")
    print("=" * 55)
    print(f"  Patient        : {PATIENT_NAME} (ID: {PATIENT_ID})")
    print(f"  MQTT Broker    : {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
    print(f"  MQTT Topic     : {MQTT_TOPIC}")
    print(f"  Sensor Interval: every {SENSOR_PUBLISH_INTERVAL_SEC}s")
    print(f"  Sequence Length: {SEQUENCE_LENGTH} time steps")
    print(f"  Risk Labels    : {list(RISK_LABELS.values())}")
    print(f"  LLM Provider   : {LLM_PROVIDER}")
    print(f"  Gemini Model   : {GEMINI_MODEL}")
    print(f"  LLM Temperature: {LLM_TEMPERATURE}")
    print(f"  LSTM Path      : {LSTM_MODEL_PATH}")
    print(f"  CNN Path       : {CNN_MODEL_PATH}")
    print(f"  Data Dir       : {DATA_PROCESSED_DIR}")
    print("-" * 55)
    print(f"  Google API Key : {'✅ Found' if GOOGLE_API_KEY else '❌ MISSING'}")
    print(f"  SerpAPI Key    : {'✅ Found' if SERPAPI_API_KEY else '❌ MISSING'}")
    print(f"  OpenAI Key     : {'✅ Found' if OPENAI_API_KEY else '— not set (optional)'}")
    print(f"  Anthropic Key  : {'✅ Found' if ANTHROPIC_API_KEY else '— not set (optional)'}")
    print("=" * 55)