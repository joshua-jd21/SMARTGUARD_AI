# =============================================================
#  SmartGuard AI — Master Runner
#  File: main.py
#
#  PURPOSE:
#  Single entry point for the entire SmartGuard AI system.
#
#  USAGE:
#
#    Run everything (IoT + Dashboard):
#      python main.py
#
#    Run pipeline only:
#      python main.py --mode pipeline
#
#    Run dashboard only:
#      python main.py --mode dashboard
#
#    Run IoT simulation only:
#      python main.py --mode iot
#
#    Test pipeline with mock data:
#      python main.py --mode test
#
# =============================================================

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

from loguru import logger

# ─── Logging configuration ────────────────────────────────────
# Remove default handler and add structured output:
#   - Stderr (INFO+): human-readable during development
#   - File (DEBUG+): full trace with rotation for debugging
logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True)
logger.add(
    "smartguard.log",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    enqueue=True,   # thread-safe async writes
)
# ──────────────────────────────────────────────────────────────

# =============================================================
# PROJECT ROOT
# =============================================================

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# =============================================================
# SETTINGS
# =============================================================

from config.settings import (
    DASHBOARD_PORT,
)

# =============================================================
# CONFIG CHECK
# =============================================================

def _check_config() -> bool:
    """
    Verify critical environment variables before startup.
    """

    try:

        from config.settings import (
            GOOGLE_API_KEY,
            SERPAPI_API_KEY,
            GEMINI_MODEL,
            MQTT_BROKER_HOST,
            MQTT_BROKER_PORT,
            PATIENT_NAME,
        )

        logger.info("=" * 60)
        logger.info("  SmartGuard AI — Startup Configuration")
        logger.info("=" * 60)

        logger.info(f"  Patient        : {PATIENT_NAME}")

        logger.info(
            f"  MQTT Broker    : "
            f"{MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}"
        )

        logger.info(f"  Gemini Model   : {GEMINI_MODEL}")

        logger.info("-" * 60)

        logger.info(
            f"  Google API Key : "
            f"{'FOUND' if GOOGLE_API_KEY else 'MISSING'}"
        )

        logger.info(
            f"  SerpAPI Key    : "
            f"{'FOUND' if SERPAPI_API_KEY else 'MISSING (fallback mode)'}"
        )

        logger.info("=" * 60)

        if not GOOGLE_API_KEY:

            logger.error(
                "GOOGLE_API_KEY missing in .env. "
                "Gemini agents cannot start."
            )

            return False

        return True

    except Exception as e:

        logger.error(
            f"Configuration check failed: {e}"
        )

        return False


# =============================================================
# MODE: TEST
# =============================================================

def run_test() -> None:
    """
    Runs full pipeline tests across all risk levels.
    """

    logger.info("=" * 60)
    logger.info("  SmartGuard AI — TEST MODE")
    logger.info("=" * 60)

    from agents.agent_pipeline import (
        SmartGuardPipeline,
        _mock_prediction,
    )

    pipeline = SmartGuardPipeline()

    for risk_label in [

        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    ]:

        logger.info("")
        logger.info(
            f"Testing risk level: {risk_label}"
        )

        result = pipeline.run(
            _mock_prediction(risk_label)
        )

        if result.success:

            logger.success(
                f"[{risk_label}] SUCCESS | "
                f"time={result.execution_time_sec}s | "
                f"urgency={result.risk_context.urgency_level}/5 | "
                f"premium=+{result.insurance_report.premium_adjustment}%"
            )

        else:

            logger.error(
                f"[{risk_label}] FAILED: {result.error}"
            )

    logger.info("")
    logger.info("=" * 60)
    logger.info("  TEST COMPLETE")
    logger.info("=" * 60)


# =============================================================
# MODE: PIPELINE
# =============================================================

def run_pipeline_once() -> None:
    """
    Runs the pipeline once using mock data.
    """

    logger.info(
        "Running SmartGuard AI pipeline..."
    )

    from agents.agent_pipeline import (
        SmartGuardPipeline,
        _mock_prediction,
    )

    pipeline = SmartGuardPipeline()

    result = pipeline.run(
        _mock_prediction("CRITICAL")
    )

    if result.success:

        print("\n")
        print(result.insurance_report.report_text)

    else:

        logger.error(
            f"Pipeline failed: {result.error}"
        )


# =============================================================
# MODE: IOT
# =============================================================

def run_iot() -> None:
    """
    Starts MQTT sensor simulator.
    """

    logger.info(
        "Starting IoT sensor simulation..."
    )

    iot_script = (
        ROOT_DIR
        / "iot"
        / "mqtt_publisher.py"
    )

    if not iot_script.exists():

        logger.error(
            f"IoT script not found:\n{iot_script}"
        )

        return

    try:

        subprocess.run(
            [sys.executable, str(iot_script)],
            check=True,
        )

    except KeyboardInterrupt:

        logger.info(
            "IoT simulation stopped."
        )

    except subprocess.CalledProcessError as e:

        logger.error(
            f"IoT simulation crashed: {e}"
        )


# =============================================================
# MODE: DASHBOARD
# =============================================================

def run_dashboard() -> None:
    """
    Launch Streamlit dashboard.
    """

    logger.info(
        "Launching Streamlit dashboard..."
    )

    dashboard_script = (
        ROOT_DIR
        / "dashboard"
        / "app.py"
    )

    if not dashboard_script.exists():

        logger.error(
            f"Dashboard script not found:\n"
            f"{dashboard_script}"
        )

        return

    try:

        subprocess.run(

            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(dashboard_script),

                "--server.port",
                str(DASHBOARD_PORT),

                "--server.headless=true",
            ],

            check=True,
        )

    except KeyboardInterrupt:

        logger.info(
            "Dashboard stopped."
        )

    except subprocess.CalledProcessError as e:

        logger.error(
            f"Dashboard crashed: {e}"
        )


# =============================================================
# MODE: FULL SYSTEM
# =============================================================

def run_full() -> None:
    """
    Runs:
      - IoT simulator (background)
      - Dashboard (foreground)
    """

    logger.info("=" * 60)
    logger.info("  SmartGuard AI — FULL SYSTEM MODE")
    logger.info("  IoT + Dashboard")
    logger.info("=" * 60)

    iot_script = (
        ROOT_DIR
        / "iot"
        / "mqtt_publisher.py"
    )

    dashboard_script = (
        ROOT_DIR
        / "dashboard"
        / "app.py"
    )

    missing = []

    if not iot_script.exists():
        missing.append(str(iot_script))

    if not dashboard_script.exists():
        missing.append(str(dashboard_script))

    if missing:

        logger.error(
            "Cannot start full system.\n"
            "Missing files:\n"
            + "\n".join(missing)
        )

        return

    # =========================================================
    # IOT BACKGROUND THREAD
    # =========================================================

    def _iot_thread():

        try:

            subprocess.run(
                [sys.executable, str(iot_script)],
                check=True,
            )

        except Exception as e:

            logger.warning(
                f"IoT thread exited: {e}"
            )

    iot = threading.Thread(
        target=_iot_thread,
        daemon=True,
    )

    iot.start()

    logger.info(
        "IoT simulator started in background."
    )

    logger.info(
        "Waiting 3 seconds before dashboard launch..."
    )

    time.sleep(3)

    # =========================================================
    # DASHBOARD FOREGROUND
    # =========================================================

    try:

        run_dashboard()

    except KeyboardInterrupt:

        logger.info(
            "Full system shutdown requested."
        )


# =============================================================
# ARGUMENT PARSER
# =============================================================

def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(

        prog="SmartGuard AI",

        description=(
            "SmartGuard AI — Real-Time "
            "Health Insurance Risk Assessment"
        ),
    )

    parser.add_argument(

        "--mode",

        choices=[
            "full",
            "pipeline",
            "dashboard",
            "iot",
            "test",
        ],

        default="full",

        help=(
            "full      = IoT + Dashboard\n"
            "pipeline  = Single pipeline run\n"
            "dashboard = Dashboard only\n"
            "iot       = IoT simulator only\n"
            "test      = Test all risk levels\n"
        ),
    )

    return parser.parse_args()


# =============================================================
# MAIN
# =============================================================

def main() -> None:

    args = _parse_args()

    logger.info("=" * 60)
    logger.info("  SmartGuard AI")
    logger.info(f"  MODE: {args.mode.upper()}")
    logger.info("=" * 60)

    # =========================================================
    # CONFIG CHECK
    # Skip for pure IoT mode
    # =========================================================

    if args.mode != "iot":

        if not _check_config():

            logger.error(
                "Startup aborted due to configuration errors."
            )

            sys.exit(1)

    # =========================================================
    # DISPATCH
    # =========================================================

    if args.mode == "test":

        run_test()

    elif args.mode == "pipeline":

        run_pipeline_once()

    elif args.mode == "iot":

        run_iot()

    elif args.mode == "dashboard":

        run_dashboard()

    else:

        run_full()


# =============================================================
# ENTRY
# =============================================================

if __name__ == "__main__":

    main()