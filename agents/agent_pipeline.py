# =============================================================
# SmartGuard AI — Multi-Agent Pipeline (FINAL PRODUCTION)
# File: agents/agent_pipeline.py
#
# PURPOSE:
# -------------------------------------------------------------
# Chains all 3 agents together:
#
#   Agent 1 → RiskReaderAgent
#   Agent 2 → WebResearcherAgent
#   Agent 3 → PolicyAdvisorAgent
#
# INPUT:
#   Deep learning prediction dictionary
#
# OUTPUT:
#   Final InsuranceReport
#
# DESIGN:
# -------------------------------------------------------------
# ✅ Full end-to-end orchestration
# ✅ Structured dataclass chaining
# ✅ Production logging
# ✅ Timing metrics
# ✅ Graceful failure handling
# ✅ Defensive prediction validation
# ✅ Gemini-compatible
# ✅ Compatible with fallback-only mode
# ✅ Ready for MQTT + Streamlit integration
# ✅ JSON serialisation support
# =============================================================

import hashlib
import json
import sys
import time

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

from loguru import logger

# ============================================================
# QUOTA PROTECTION — in-memory response cache + rate limiter
# ============================================================

_response_cache: Dict[str, Tuple] = {}

# Cache entries expire after this many seconds
CACHE_TTL_SECONDS = 300  # 5 minutes

# Minimum wall-clock gap between pipeline runs (per instance)
MIN_RUN_INTERVAL_SECONDS = 10


def _cache_key(prediction: Dict) -> str:
    """Stable hash of the prediction payload for cache lookups."""
    serialised = json.dumps(prediction, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()

# =============================================================
# PROJECT ROOT
# =============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# =============================================================
# AGENTS
# =============================================================

from agents.risk_reader_agent import (
    RiskReaderAgent,
    RiskContext,
)

from agents.web_researcher_agent import (
    WebResearcherAgent,
    ResearchResult,
)

from agents.policy_advisor_agent import (
    PolicyAdvisorAgent,
    InsuranceReport,
)

# =============================================================
# PIPELINE RESULT
# =============================================================

@dataclass
class PipelineResult:
    """
    Full pipeline output bundle.
    """

    risk_context: Optional[RiskContext]
    research_result: Optional[ResearchResult]
    insurance_report: Optional[InsuranceReport]

    success: bool
    status: str

    execution_time_sec: float

    created_at: str

    pipeline_version: str = "1.0.0"

    error: str = ""

    # =========================================================
    # SERIALISATION
    # =========================================================

    def to_dict(self) -> Dict:

        return {

            "risk_context":
                self.risk_context.to_dict()
                if self.risk_context else None,

            "research_result":
                self.research_result.to_dict()
                if self.research_result else None,

            "insurance_report":
                self.insurance_report.to_dict()
                if self.insurance_report else None,

            "success":
                self.success,

            "status":
                self.status,

            "execution_time_sec":
                self.execution_time_sec,

            "pipeline_version":
                self.pipeline_version,

            "created_at":
                self.created_at,

            "error":
                self.error,
        }

    def to_json(self) -> str:

        return json.dumps(
            self.to_dict(),
            indent=2,
        )


# =============================================================
# SMARTGUARD PIPELINE
# =============================================================

class SmartGuardPipeline:
    """
    Main orchestration pipeline.
    """

    def __init__(self):

        logger.info("=" * 60)
        logger.info(" Initialising SmartGuard AI Pipeline")
        logger.info("=" * 60)

        try:

            self.risk_reader = (
                RiskReaderAgent()
            )

            self.web_researcher = (
                WebResearcherAgent()
            )

            self.policy_advisor = (
                PolicyAdvisorAgent()
            )

        except Exception as e:

            logger.exception(
                f"Agent initialisation failed: {e}"
            )

            raise

        logger.success(
            "All agents initialised successfully."
        )

        # Rate-limiter state — tracks last run time per instance
        self._last_run_time: float = 0.0

    # =========================================================
    # VALIDATION
    # =========================================================

    def _validate_prediction(
        self,
        prediction: Dict,
    ) -> None:
        """
        Defensive validation before pipeline execution.
        """

        required_keys = [

            "risk_prediction",

            "anomaly_detection",
        ]

        for key in required_keys:

            if key not in prediction:

                raise ValueError(
                    f"Missing required prediction key: {key}"
                )

        # =====================================================
        # RISK STRUCTURE VALIDATION
        # =====================================================

        risk_required = [

            "risk_label",

            "risk_class",

            "confidence",

            "all_probabilities",
        ]

        for key in risk_required:

            if key not in prediction[
                "risk_prediction"
            ]:

                raise ValueError(
                    f"Missing risk_prediction field: {key}"
                )

        # =====================================================
        # VALID RISK LABELS
        # =====================================================

        valid_risks = {

            "LOW",

            "MEDIUM",

            "HIGH",

            "CRITICAL",
        }

        risk_label = str(
            prediction["risk_prediction"]["risk_label"]
        ).strip().upper()

        if risk_label not in valid_risks:

            raise ValueError(
                f"Invalid risk label: {risk_label}"
            )

        # =====================================================
        # ANOMALY STRUCTURE VALIDATION
        # =====================================================

        anomaly_required = [

            "is_anomaly",

            "anomaly_score",

            "reconstruction_error",
        ]

        for key in anomaly_required:

            if key not in prediction[
                "anomaly_detection"
            ]:

                raise ValueError(
                    f"Missing anomaly_detection field: {key}"
                )

    # =========================================================
    # MAIN PIPELINE
    # =========================================================

    def run(
        self,
        prediction: Dict,
    ) -> PipelineResult:

        start_time = time.time()

        logger.info("=" * 60)
        logger.info(" SMARTGUARD AI PIPELINE STARTED")
        logger.info("=" * 60)

        # =====================================================
        # CACHE CHECK — return early on duplicate input
        # =====================================================

        key = _cache_key(prediction)
        now = time.time()
        if key in _response_cache:
            cached_result, cached_at = _response_cache[key]
            age = now - cached_at
            if age < CACHE_TTL_SECONDS:
                logger.info(
                    f"Cache hit — returning cached result "
                    f"(age={age:.0f}s, ttl={CACHE_TTL_SECONDS}s)"
                )
                return cached_result
            else:
                del _response_cache[key]

        # =====================================================
        # RATE LIMITER — enforce minimum run interval
        # =====================================================

        since_last = now - self._last_run_time
        if since_last < MIN_RUN_INTERVAL_SECONDS:
            wait = MIN_RUN_INTERVAL_SECONDS - since_last
            logger.info(
                f"Rate limit: waiting {wait:.1f}s before next run"
            )
            time.sleep(wait)

        self._last_run_time = time.time()

        risk_context = None
        research_result = None
        insurance_report = None

        try:

            # =================================================
            # VALIDATE INPUT
            # =================================================

            self._validate_prediction(
                prediction
            )

            logger.success(
                "Prediction payload validation successful."
            )

            # =================================================
            # AGENT 1 — RISK READER
            # =================================================

            logger.info("")
            logger.info(
                "STEP 1/3 — RiskReaderAgent"
            )

            risk_context = (
                self.risk_reader.run(
                    prediction
                )
            )

            if not risk_context:

                raise RuntimeError(
                    "RiskReaderAgent returned None."
                )

            logger.success(
                f"RiskReaderAgent complete | "
                f"risk={risk_context.risk_label} | "
                f"urgency={risk_context.urgency_level}/5"
            )

            # =================================================
            # AGENT 2 — WEB RESEARCHER
            # =================================================

            logger.info("")
            logger.info(
                "STEP 2/3 — WebResearcherAgent"
            )

            research_result = (
                self.web_researcher.run(
                    risk_context
                )
            )

            if not research_result:

                raise RuntimeError(
                    "WebResearcherAgent returned None."
                )

            logger.success(
                f"WebResearcherAgent complete | "
                f"sources={len(research_result.sources_cited)}"
            )

            # =================================================
            # AGENT 3 — POLICY ADVISOR
            # =================================================

            logger.info("")
            logger.info(
                "STEP 3/3 — PolicyAdvisorAgent"
            )

            insurance_report = (
                self.policy_advisor.run(
                    research_result
                )
            )

            if not insurance_report:

                raise RuntimeError(
                    "PolicyAdvisorAgent returned None."
                )

            logger.success(
                f"PolicyAdvisorAgent complete | "
                f"premium=+{insurance_report.premium_adjustment}%"
            )

            # =================================================
            # COMPLETE
            # =================================================

            elapsed = round(
                time.time() - start_time,
                2
            )

            logger.success("=" * 60)

            logger.success(
                f" PIPELINE COMPLETE | "
                f"time={elapsed}s"
            )

            logger.success("=" * 60)

            pipeline_result = PipelineResult(

                risk_context=risk_context,

                research_result=research_result,

                insurance_report=insurance_report,

                success=True,

                status="SUCCESS",

                execution_time_sec=elapsed,

                created_at=datetime.now(
                    timezone.utc
                ).isoformat(),
            )

            # Store in cache for deduplication
            _response_cache[key] = (pipeline_result, time.time())

            return pipeline_result

        except Exception as e:

            elapsed = round(
                time.time() - start_time,
                2
            )

            logger.exception(
                f"Pipeline failed: {e}"
            )

            return PipelineResult(

                risk_context=risk_context,

                research_result=research_result,

                insurance_report=insurance_report,

                success=False,

                status="FAILED",

                execution_time_sec=elapsed,

                created_at=datetime.now(
                    timezone.utc
                ).isoformat(),

                error=str(e),
            )

# =============================================================
# MOCK PREDICTION
# =============================================================

def _mock_prediction(
    risk_label: str = "CRITICAL"
) -> Dict:
    """
    Simulates real deep learning model output.
    """

    class_map = {

        "LOW": 0,

        "MEDIUM": 1,

        "HIGH": 2,

        "CRITICAL": 3,
    }

    # =========================================================
    # REALISTIC PROBABILITIES
    # =========================================================

    probs = {

        "LOW": 0.004,

        "MEDIUM": 0.004,

        "HIGH": 0.004,

        "CRITICAL": 0.988,
    }

    if risk_label != "CRITICAL":

        probs = {

            label: 0.004
            for label in class_map
        }

        probs[risk_label] = 0.988

    return {

        "patient": {

            "name": "Rajan",

            "age": 45,
        },

        "risk_prediction": {

            "risk_label":
                risk_label,

            "risk_class":
                class_map[risk_label],

            "confidence":
                0.9875,

            "failsafe_triggered":
                False,

            "all_probabilities":
                probs,
        },

        "anomaly_detection": {

            "is_anomaly":
                True,

            "anomaly_score":
                0.8732,

            "reconstruction_error":
                0.09214,

            "threshold_used":
                0.05120,
        },
    }

# =============================================================
# MAIN TEST
# =============================================================

if __name__ == "__main__":

    logger.info(
        "Testing SmartGuardPipeline..."
    )

    pipeline = SmartGuardPipeline()

    result = pipeline.run(
        _mock_prediction("CRITICAL")
    )

    print("\n")

    if result.success:

        print("=" * 62)
        print(" SMARTGUARD AI — FINAL REPORT")
        print("=" * 62)

        print(
            result.insurance_report.report_text
        )

        print("\nPIPELINE METRICS")
        print("-" * 62)

        print(
            f"Execution Time : "
            f"{result.execution_time_sec}s"
        )

        print(
            f"Pipeline Status: "
            f"{result.status}"
        )

        print(
            f"Pipeline Ver. : "
            f"{result.pipeline_version}"
        )

        print(
            f"Created At     : "
            f"{result.created_at}"
        )

        print("=" * 62)

    else:

        print("=" * 62)
        print(" PIPELINE FAILED")
        print("=" * 62)

        print(f"Error: {result.error}")

        print(
            f"\nExecution Time: "
            f"{result.execution_time_sec}s"
        )

        print("=" * 62)