# =============================================================
#  SmartGuard AI — Risk Reader Agent
#  File: agents/risk_reader_agent.py
#
#  AGENT 1 of 3
#
#  INPUT:
#      Deep learning prediction dictionary
#      {risk_prediction, anomaly_detection, patient}
#
#  OUTPUT:
#      RiskContext dataclass
#
#  FIXES APPLIED:
#  ✅ Uses GEMINI_MODEL from settings (not hardcoded)
#  ✅ cnn_confidence stored as float (not formatted string)
#  ✅ anomaly_score stored as float (not formatted string)
#  ✅ Retry logic
#  ✅ Deterministic urgency floor
#  ✅ Failsafe escalation
#  ✅ Robust parser
#  ✅ UTC timestamps
# =============================================================

import sys

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from loguru import logger

# =============================================================
#  PROJECT ROOT
# =============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# =============================================================
#  LANGCHAIN
# =============================================================

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# =============================================================
#  SETTINGS
# =============================================================

from config.settings import (
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    LLM_TEMPERATURE,
    PREMIUM_ADJUSTMENTS,
    CLAIM_PROBABILITIES,
    COVERAGE_RECOMMENDATIONS,
    URGENCY_FLOOR,
)

# =============================================================
#  RISK CONTEXT DATACLASS
# =============================================================

@dataclass
class RiskContext:
    """
    Structured output of Agent 1.
    All numeric fields stored as raw Python types (float/int).
    Formatting is done at display/report time only.
    """

    patient_name:      str
    patient_age:       int

    risk_label:        str
    risk_class:        int

    # Stored as float [0.0 – 1.0] — format at display time
    cnn_confidence:    float

    is_anomaly:        bool

    # Stored as float [0.0 – 1.0] — format at display time
    anomaly_score:     float
    reconstruction_err: float

    failsafe_triggered: bool

    all_probabilities:  Dict[str, float]

    medical_summary:    str = ""
    search_queries:     List[str] = field(default_factory=list)

    urgency_level:      int = 1
    premium_adjustment: int = 0
    claim_probability:  int = 0
    coverage_notes:     List[str] = field(default_factory=list)

    created_at:         str = ""

    def to_dict(self) -> Dict:
        return self.__dict__


# =============================================================
#  SYSTEM PROMPT
# =============================================================

SYSTEM_PROMPT = """
You are a senior medical AI analyst working for SmartGuard AI,
a health insurance risk assessment platform.

Your role is to interpret physiological sensor data and deep
learning model outputs, and produce structured medical context
for the insurance underwriting team.

IMPORTANT RULES:

1. If failsafe_triggered is True, the system detected a severe
   physiological anomaly despite classifier uncertainty.
   Treat this as elevated urgency regardless of risk label.

2. URGENCY_LEVEL must always follow:
   LOW=1, MEDIUM=2, HIGH=3, CRITICAL=4 or 5
   Never return URGENCY_LEVEL: 2 for a CRITICAL patient.

3. Never diagnose diseases.
   Describe physiological patterns only.

4. Be concise, clinical, and professional.

5. Consider patient age and anomaly severity.

You MUST respond in this EXACT format (values on the same line):

MEDICAL_SUMMARY: <2-3 sentence interpretation>
URGENCY_LEVEL: <integer 1-5>
SEARCH_QUERY_1: <medical guideline query>
SEARCH_QUERY_2: <insurance underwriting query>
SEARCH_QUERY_3: <treatment/prevention query>
"""

# =============================================================
#  HUMAN PROMPT
# =============================================================

HUMAN_PROMPT = """
Interpret this patient risk assessment.

PATIENT:
Name: {patient_name}
Age: {patient_age}

CNN RISK CLASSIFICATION:
Risk Label: {risk_label}
Risk Class: {risk_class}
Confidence: {cnn_confidence_pct}
All Probabilities: {all_probabilities}
Failsafe Triggered: {failsafe_triggered}

LSTM ANOMALY DETECTION:
Anomaly Detected: {is_anomaly}
Anomaly Score: {anomaly_score_fmt}
Reconstruction Error: {reconstruction_err_fmt}

INSURANCE CONTEXT:
Premium Adjustment: +{premium_adjustment}%
Claim Probability: {claim_probability}%
Coverage Notes: {coverage_notes}

Respond exactly as instructed.
URGENCY_LEVEL must be >= {urgency_floor}.
"""

# =============================================================
#  RESPONSE PARSER
# =============================================================

def _parse_llm_response(response_text: str, risk_label: str) -> Dict:
    """
    Parses the LLM response.
    All keys expected on the same line as their value.
    Falls back gracefully if any field is missing.
    """

    floor = URGENCY_FLOOR.get(risk_label, 1)

    result = {
        "medical_summary": "",
        "urgency_level":   floor,
        "search_queries":  [],
    }

    for raw_line in response_text.strip().splitlines():

        line = (
            raw_line
            .strip()
            .replace("**", "")
            .replace("*", "")
            .replace("#", "")
            .strip()
        )

        if line.startswith("MEDICAL_SUMMARY:"):
            result["medical_summary"] = (
                line.split("MEDICAL_SUMMARY:", 1)[-1].strip()
            )

        elif line.startswith("URGENCY_LEVEL:"):
            try:
                raw_urgency = int(
                    line.split("URGENCY_LEVEL:", 1)[-1].strip()
                )
                raw_urgency = max(1, min(5, raw_urgency))
                result["urgency_level"] = max(raw_urgency, floor)
            except Exception:
                result["urgency_level"] = floor

        elif line.startswith("SEARCH_QUERY_"):
            query = line.split(":", 1)[-1].strip()
            if query:
                result["search_queries"].append(query)

    # ---------------------------------------------------------
    #  FALLBACK: medical summary
    # ---------------------------------------------------------
    if not result["medical_summary"]:
        severity_map = {
            "LOW":      "within normal physiological parameters",
            "MEDIUM":   "showing mildly elevated physiological readings requiring monitoring",
            "HIGH":     "showing significant physiological instability requiring clinical review",
            "CRITICAL": "flagged with CRITICAL physiological instability requiring immediate clinical interpretation",
        }
        phrase = severity_map.get(risk_label, "requiring clinical review")
        result["medical_summary"] = f"Patient {phrase}."

    # ---------------------------------------------------------
    #  FALLBACK: search queries
    # ---------------------------------------------------------
    if not result["search_queries"]:
        risk = risk_label.lower()
        result["search_queries"] = [
            f"{risk} cardiovascular risk management guidelines 2024",
            f"health insurance {risk} physiological risk underwriting assessment",
            f"preventive cardiology monitoring {risk} risk recommendations",
        ]

    return result


# =============================================================
#  RISK READER AGENT
# =============================================================

class RiskReaderAgent:

    def __init__(self):

        if not GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY is not set. Check your .env file.")
            raise ValueError("GOOGLE_API_KEY is not set.")

        # -----------------------------------------------------
        #  GEMINI LLM — uses GEMINI_MODEL from settings
        # -----------------------------------------------------
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=LLM_TEMPERATURE,
            google_api_key=GOOGLE_API_KEY,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(HUMAN_PROMPT),
        ])

        # LCEL chain
        self.chain = self.prompt | self.llm

        logger.success(
            f"RiskReaderAgent initialised | model={GEMINI_MODEL}"
        )

    # =========================================================
    #  MAIN RUN
    # =========================================================

    def run(self, prediction: Dict) -> RiskContext:

        logger.info("=" * 60)
        logger.info("  Agent 1 — Risk Reader")
        logger.info("=" * 60)

        # =====================================================
        #  EXTRACT SECTIONS
        # =====================================================

        cnn     = prediction.get("risk_prediction", {})
        anomaly = prediction.get("anomaly_detection", {})

        # =====================================================
        #  VALIDATE CNN FIELDS
        # =====================================================

        for f in ["risk_label", "risk_class", "confidence", "all_probabilities"]:
            if f not in cnn:
                raise ValueError(f"Missing CNN field: {f}")

        # =====================================================
        #  VALIDATE ANOMALY FIELDS
        # =====================================================

        for f in ["is_anomaly", "anomaly_score", "reconstruction_error"]:
            if f not in anomaly:
                raise ValueError(f"Missing anomaly field: {f}")

        # =====================================================
        #  PATIENT
        # =====================================================

        patient      = prediction.get("patient", {})
        patient_name = patient.get("name", "Patient")

        try:
            patient_age = int(patient.get("age", 0))
        except (TypeError, ValueError):
            raise ValueError("Invalid patient age in prediction payload.")

        # =====================================================
        #  CNN VALUES — stored as raw types
        # =====================================================

        risk_label  = str(cnn["risk_label"]).upper()
        risk_class  = int(cnn["risk_class"])
        confidence  = float(cnn["confidence"])          # ← float, NOT string
        all_probs   = dict(cnn["all_probabilities"])
        failsafe    = bool(cnn.get("failsafe_triggered", False))

        # =====================================================
        #  ANOMALY VALUES — stored as raw floats
        # =====================================================

        is_anomaly       = bool(anomaly["is_anomaly"])
        anomaly_score    = float(anomaly["anomaly_score"])      # ← float
        anomaly_score    = max(0.0, min(1.0, anomaly_score))
        reconstruction_err = float(anomaly["reconstruction_error"])  # ← float

        # =====================================================
        #  INSURANCE VALUES
        # =====================================================

        premium_adj    = PREMIUM_ADJUSTMENTS.get(risk_label, 0)
        claim_prob     = CLAIM_PROBABILITIES.get(risk_label, 0)
        coverage_notes = COVERAGE_RECOMMENDATIONS.get(risk_label, [])

        # =====================================================
        #  URGENCY FLOOR
        # =====================================================

        urgency_floor = URGENCY_FLOOR.get(risk_label, 1)

        if failsafe and urgency_floor < 4:
            urgency_floor = 4

        # =====================================================
        #  PROMPT PAYLOAD
        #  Format floats here for display in the prompt only
        # =====================================================

        payload = {
            "patient_name":        patient_name,
            "patient_age":         patient_age,
            "risk_label":          risk_label,
            "risk_class":          risk_class,
            "cnn_confidence_pct":  f"{confidence:.2%}",
            "all_probabilities":   all_probs,
            "failsafe_triggered":  failsafe,
            "is_anomaly":          is_anomaly,
            "anomaly_score_fmt":   f"{anomaly_score:.4f}",
            "reconstruction_err_fmt": f"{reconstruction_err:.5f}",
            "premium_adjustment":  premium_adj,
            "claim_probability":   claim_prob,
            "coverage_notes":      coverage_notes,
            "urgency_floor":       urgency_floor,
        }

        # =====================================================
        #  LLM CALL WITH RETRIES
        # =====================================================

        raw_text    = ""
        llm_success = False

        for attempt in range(1, 4):
            try:
                response    = self.chain.invoke(payload)
                raw_text    = response.content
                llm_success = True
                # Log token usage if available
                usage = getattr(response, "response_metadata", {}).get(
                    "usage_metadata", {}
                )
                if usage:
                    logger.debug(
                        f"RiskReaderAgent tokens: "
                        f"input={usage.get('prompt_token_count', '?')} "
                        f"output={usage.get('candidates_token_count', '?')}"
                    )
                logger.debug(f"LLM succeeded on attempt {attempt}")
                break
            except Exception as e:
                logger.warning(f"LLM attempt {attempt}/3 failed: {e}")

        if not llm_success:
            logger.error("All LLM attempts failed. Using deterministic fallback.")

        # =====================================================
        #  PARSE RESPONSE
        # =====================================================

        parsed = _parse_llm_response(raw_text, risk_label)

        # =====================================================
        #  BUILD RISK CONTEXT
        #  All numeric values stored as raw floats/ints
        # =====================================================

        context = RiskContext(
            patient_name=patient_name,
            patient_age=patient_age,
            risk_label=risk_label,
            risk_class=risk_class,
            cnn_confidence=confidence,            # ← raw float
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,          # ← raw float
            reconstruction_err=reconstruction_err, # ← raw float
            failsafe_triggered=failsafe,
            all_probabilities=all_probs,
            medical_summary=parsed["medical_summary"],
            search_queries=parsed["search_queries"],
            urgency_level=max(parsed["urgency_level"], urgency_floor),
            premium_adjustment=premium_adj,
            claim_probability=claim_prob,
            coverage_notes=coverage_notes,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        logger.success(
            f"RiskReaderAgent complete | "
            f"risk={context.risk_label} | "
            f"urgency={context.urgency_level}/5 | "
            f"llm={'OK' if llm_success else 'FALLBACK'}"
        )

        return context


# =============================================================
#  MOCK TEST
# =============================================================

def _mock_prediction() -> Dict:
    return {
        "patient": {"name": "Rajan", "age": 45},
        "risk_prediction": {
            "risk_label":        "CRITICAL",
            "risk_class":        3,
            "confidence":        0.9875,
            "failsafe_triggered": False,
            "all_probabilities": {
                "LOW": 0.001, "MEDIUM": 0.004,
                "HIGH": 0.007, "CRITICAL": 0.988,
            },
        },
        "anomaly_detection": {
            "is_anomaly":         True,
            "anomaly_score":      0.8732,
            "reconstruction_error": 0.09214,
            "threshold_used":     0.05120,
        },
    }


# =============================================================
#  MAIN
# =============================================================

if __name__ == "__main__":

    logger.info("Testing RiskReaderAgent...")

    agent   = RiskReaderAgent()
    context = agent.run(_mock_prediction())

    print("\n" + "=" * 60)
    print("  SMARTGUARD AI — RISK CONTEXT")
    print("=" * 60)
    print(f"Patient         : {context.patient_name}")
    print(f"Age             : {context.patient_age}")
    print(f"Risk Level      : {context.risk_label}")
    print(f"CNN Confidence  : {context.cnn_confidence:.2%}")
    print(f"Anomaly Score   : {context.anomaly_score:.4f}")
    print(f"Urgency Level   : {context.urgency_level}/5")
    print(f"\nMedical Summary :\n{context.medical_summary}")
    print("\nSearch Queries:")
    for i, q in enumerate(context.search_queries, 1):
        print(f"  {i}. {q}")
    print("\nCoverage Notes:")
    for note in context.coverage_notes:
        print(f"  • {note}")
    print(f"\nCreated At: {context.created_at}")
    print("=" * 60)