#!/usr/bin/env python3
"""
SmartGuard AI — Setup Verification Script
==========================================
Run this before first startup to catch environment issues early.

Usage:
    python check_setup.py
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Created by scripts/bootstrap_macos_arm64_venv.sh when only Rosetta/x86 Python exists.
UI_ONLY_VENV = (ROOT_DIR / ".venv" / "SMARTGUARD_NO_TF").exists()

PASS  = "  ✅"
WARN  = "  ⚠️ "
FAIL  = "  ❌"
INFO  = "  ℹ️ "

issues  = []
warnings = []


def _homebrew_arm64_python310() -> str:
    """Return path to an arm64 Homebrew CPython 3.10, or empty string."""
    for rel in (
        "/opt/homebrew/opt/python@3.10/bin/python3.10",
        "/opt/homebrew/bin/python3.10",
    ):
        p = Path(rel)
        if not p.is_file() or not os.access(p, os.X_OK):
            continue
        try:
            proc = subprocess.run(
                [str(p), "-c", "import platform; print(platform.machine())"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip() == "arm64":
                return str(p)
        except (OSError, subprocess.TimeoutExpired):
            continue
    return ""


def check(label: str, ok: bool, detail: str = "", fatal: bool = False):
    if ok:
        print(f"{PASS} {label}")
    elif fatal:
        print(f"{FAIL} {label}" + (f"\n       {detail}" if detail else ""))
        issues.append(label)
    else:
        print(f"{WARN} {label}" + (f"\n       {detail}" if detail else ""))
        warnings.append(label)


print("=" * 60)
print("  SmartGuard AI — Setup Check")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# 1. Python version
# ──────────────────────────────────────────────────────────────

print("\n[1] Python")
major, minor = sys.version_info[:2]
check(
    f"Python {major}.{minor}",
    (major == 3 and minor == 10),
    "This project is validated on Python 3.10 (see requirements.txt).",
    fatal=True,
)

machine = platform.machine().lower()
if sys.platform == "darwin" and machine == "x86_64":
    try:
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        brand = ""
    if "Apple" in brand and not UI_ONLY_VENV:
        brew_py = _homebrew_arm64_python310()
        venv_lines = (
            f"         rm -rf .venv\n"
            f"         {brew_py or '/opt/homebrew/opt/python@3.10/bin/python3.10'}"
            f" -m venv .venv\n"
            f"         source .venv/bin/activate && pip install -r requirements.txt\n"
        )
        if not brew_py:
            venv_lines = (
                "         brew install python@3.10\n"
                "         rm -rf .venv\n"
                "         /opt/homebrew/opt/python@3.10/bin/python3.10 -m venv .venv\n"
                "         source .venv/bin/activate && pip install -r requirements.txt\n"
            )
        print(
            f"\n{FAIL} This Python is x86_64-only (Rosetta). Apple Silicon needs arm64 Python.\n"
            f"       `arch -arm64 python3.10` fails if python3.10 is Intel-only (Bad CPU type).\n"
            f"       TensorFlow would install the wrong wheel and crash (AVX).\n"
            f"\n"
            f"       Fix — install Homebrew arm64 Python 3.10, then create the venv with THAT binary:\n"
            + (f"       (detected arm64 3.10: {brew_py})\n" if brew_py else "")
            + f"\n"
            f"{venv_lines}"
            f"       Or run:  bash scripts/bootstrap_macos_arm64_venv.sh\n"
        )
        issues.append("Python arch (use arm64, not Rosetta x86_64)")
        print("\n" + "=" * 60)
        print(f"  RESULT: {len(issues)} fatal issue(s) — system cannot start")
        print("=" * 60)
        sys.exit(1)

if sys.platform == "darwin" and machine == "x86_64" and UI_ONLY_VENV:
    print(
        f"\n{WARN} Rosetta/x86_64 Python with UI-only venv (.venv/SMARTGUARD_NO_TF).\n"
        f"       Agents + Streamlit work; deep_learning/* needs arm64 Python + requirements.txt."
    )

# ──────────────────────────────────────────────────────────────
# 2. .env file
# ──────────────────────────────────────────────────────────────

print("\n[2] Environment Variables")

env_path = ROOT_DIR / ".env"
check(".env file exists", env_path.exists(), f"Expected at {env_path}", fatal=True)

# Load .env manually (avoid importing settings which prints warnings)
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)

google_key = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
serp_key   = os.getenv("SERPAPI_API_KEY", "")

check(
    "GOOGLE_API_KEY (or GEMINI_API_KEY) is set",
    bool(google_key),
    "Required for Gemini agents. Add to .env",
    fatal=True,
)
check(
    "SERPAPI_API_KEY is set",
    bool(serp_key),
    "Optional — fallback mode works without it",
    fatal=False,
)

# ──────────────────────────────────────────────────────────────
# 3. Required packages
# ──────────────────────────────────────────────────────────────

print("\n[3] Python Packages")

packages = [
    ("langchain_google_genai", "langchain-google-genai", True),
    (
        "tensorflow",
        "tensorflow / tensorflow-macos",
        not UI_ONLY_VENV,
    ),
    ("streamlit",              "streamlit",              True),
    ("langchain",              "langchain",              True),
    ("langchain_community",    "langchain-community",    True),
    ("paho.mqtt",              "paho-mqtt",              True),
    ("sklearn",                "scikit-learn",           True),
    ("numpy",                  "numpy",                  True),
    ("pandas",                 "pandas",                 True),
    ("loguru",                 "loguru",                 True),
    ("joblib",                 "joblib",                 True),
    ("dotenv",                 "python-dotenv",          True),
    ("serpapi",                "google-search-results",  False),
]

for module, pip_name, fatal in packages:
    try:
        if module == "tensorflow":
            if UI_ONLY_VENV:
                print(
                    f"{INFO} tensorflow (skipped — UI-only venv; use arm64 Python + requirements.txt for DL)"
                )
                continue
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import tensorflow as tf; print(tf.__version__)",
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if proc.returncode != 0:
                err_str = (proc.stderr or "") + (proc.stdout or "")
                raise RuntimeError(err_str.strip() or f"exit {proc.returncode}")
            check(f"{pip_name}", True)
            continue
        __import__(module)
        check(f"{pip_name}", True)
    except (ImportError, Exception) as e:
        err_str = str(e)
        if "AVX" in err_str and sys.platform == "darwin":
            hint = (
                "TensorFlow x86 wheel needs AVX (often fails under Rosetta). "
                "Use arm64 Python 3.10 and recreate .venv (see README)."
            )
        elif "numpy" in err_str.lower() or "binary incompatib" in err_str.lower():
            hint = "NumPy version conflict — run: pip install -r requirements.txt"
        else:
            hint = f"Install with: pip install {pip_name}"
        check(f"{pip_name}", False, hint, fatal=fatal)

# ──────────────────────────────────────────────────────────────
# 4. Model files
# ──────────────────────────────────────────────────────────────

print("\n[4] Model Files")

models_dir = ROOT_DIR / "models"
data_dir   = ROOT_DIR / "deep_learning" / "data" / "processed"

model_files = [
    models_dir / "lstm_model.keras",
    models_dir / "cnn_model.keras",
    models_dir / "lstm_threshold.npy",
    data_dir   / "scaler.pkl",
    data_dir   / "X_train.npy",
]

all_models_ok = True
for f in model_files:
    exists = f.exists()
    if not exists:
        all_models_ok = False
    check(str(f.relative_to(ROOT_DIR)), exists, fatal=False)

if not all_models_ok:
    print(f"\n{INFO} To generate missing model files, run in order:")
    print("       1. python deep_learning/preprocess.py")
    print("       2. python deep_learning/lstm_model.py")
    print("       3. python deep_learning/cnn_model.py")

# ──────────────────────────────────────────────────────────────
# 5. MQTT broker connectivity (optional)
# ──────────────────────────────────────────────────────────────

print("\n[5] MQTT Connectivity")

try:
    import socket
    broker = os.getenv("MQTT_BROKER_HOST", "") or os.getenv("MQTT_BROKER", "broker.hivemq.com")
    port   = int(os.getenv("MQTT_BROKER_PORT", "") or os.getenv("MQTT_PORT", "1883"))
    s = socket.create_connection((broker, port), timeout=5)
    s.close()
    check(f"Reachable: {broker}:{port}", True)
except Exception as e:
    check(
        f"MQTT broker reachable",
        False,
        f"{e} — IoT simulation won't work without network access",
        fatal=False,
    )

# ──────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 60)

if issues:
    print(f"  RESULT: {len(issues)} fatal issue(s) — system cannot start")
    print(f"  Fix these before running: {', '.join(issues)}")
elif warnings:
    print(f"  RESULT: Ready (with {len(warnings)} optional warning(s))")
else:
    print("  RESULT: All checks passed — system is ready to run!")

print("=" * 60)
sys.exit(1 if issues else 0)
