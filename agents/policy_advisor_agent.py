# ============================================================
# SmartGuard AI — Policy Advisor Agent
# File: agents/policy_advisor_agent.py
#
# AGENT 3 of 3
#
# INPUT:
#     ResearchResult (from WebResearcherAgent)
#
# OUTPUT:
#     InsuranceReport
#
# FIXES APPLIED:
# ------------------------------------------------------------
# ✅ Uses GEMINI_MODEL from settings (NOT LLM_MODEL)
# ✅ Gemini integration corrected
# ✅ Retry logic
# ✅ Deterministic insurance business rules
# ✅ UTC timestamps
# ✅ Robust parser
# ✅ Structured dataclass output
# ✅ LCEL syntax
# ✅ Production-safe logging
# ============================================================

import sys

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from loguru import logger

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# ============================================================
# LANGCHAIN
# ============================================================

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ============================================================
# SETTINGS
# ============================================================

from config.settings import (
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    LLM_TEMPERATURE,
    PREMIUM_ADJUSTMENTS,
    CLAIM_PROBABILITIES,
    COVERAGE_RECOMMENDATIONS,
    URGENCY_FLOOR,
)

# ============================================================
# AGENT 2 OUTPUT
# ============================================================

from agents.web_researcher_agent import ResearchResult

# ============================================================
# OUTPUT DATACLASS
# ============================================================

@dataclass
class InsuranceReport:

    patient_name: str
    patient_age: int

    risk_level: str
    urgency_level: int

    premium_adjustment: int
    claim_probability: int

    coverage_changes: List[str] = field(default_factory=list)

    agent_finding: str = ""

    immediate_actions: List[str] = field(default_factory=list)

    policy_rationale: str = ""

    clinical_referral: bool = False

    wellness_program: str = ""

    evidence_strength: str = "MEDIUM"

    report_text: str = ""

    created_at: str = ""

    def to_dict(self) -> Dict:

        return self.__dict__


# ============================================================
# FALLBACKS
# ============================================================

_WELLNESS_PROGRAMS = {

    "LOW":
        "Standard annual wellness program",

    "MEDIUM":
        "30-day preventive cardiac wellness program",

    "HIGH":
        "60-day monitored cardiac wellness program",

    "CRITICAL":
        "90-day intensive cardiac monitoring and wellness program",
}

_EVIDENCE_STRENGTH = {

    "LOW":
        "LOW",

    "MEDIUM":
        "MEDIUM",

    "HIGH":
        "HIGH",

    "CRITICAL":
        "HIGH",
}

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a senior insurance policy advisor working for SmartGuard AI.

You receive:
1. Physiological risk analysis
2. Web-researched medical guidelines
3. Insurance underwriting context

Your job is to produce:
- an insurance justification
- immediate insurance actions
- a concise policy rationale

RULES:
1. Never contradict the assigned risk level.
2. If risk is CRITICAL, urgency must remain HIGH.
3. Never diagnose diseases.
4. Use professional insurance language.
5. Be concise and factual.

Respond EXACTLY in this format:

AGENT_FINDING: <single paragraph>

IMMEDIATE_ACTION_1: <action>

IMMEDIATE_ACTION_2: <action>

IMMEDIATE_ACTION_3: <action>

POLICY_RATIONALE: <2-3 sentence rationale>
"""

# ============================================================
# HUMAN PROMPT
# ============================================================

HUMAN_PROMPT = """
PATIENT:
Name: {patient_name}
Age: {patient_age}

RISK CONTEXT:
Risk Level: {risk_level}
Urgency Level: {urgency_level}/5

Medical Summary:
{medical_summary}

WEB RESEARCH SYNTHESIS:
{synthesised_finding}

KEY GUIDELINES:
{guidelines}

CLINICAL ACTIONS:
{clinical_actions}

INSURANCE IMPLICATIONS:
{insurance_implications}

INSURANCE RULES:
Premium Adjustment: +{premium_adjustment}%
Claim Probability: {claim_probability}%
Coverage Changes: {coverage_changes}
Wellness Program: {wellness_program}

Generate the structured insurance advisory.
"""

# ============================================================
# RESPONSE PARSER
# ============================================================

def _parse_response(text: str) -> Dict:

    result = {

        "agent_finding": "",

        "immediate_actions": [],

        "policy_rationale": "",
    }

    for raw_line in text.strip().splitlines():

        line = raw_line.strip().lstrip("*-# ").strip()

        # =====================================================
        # AGENT FINDING
        # =====================================================

        if line.startswith("AGENT_FINDING:"):

            result["agent_finding"] = (
                line.split("AGENT_FINDING:", 1)[-1]
                .strip()
            )

        # =====================================================
        # IMMEDIATE ACTIONS
        # =====================================================

        elif line.startswith("IMMEDIATE_ACTION_"):

            action = line.split(":", 1)[-1].strip()

            if action:

                result["immediate_actions"].append(
                    action
                )

        # =====================================================
        # POLICY RATIONALE
        # =====================================================

        elif line.startswith("POLICY_RATIONALE:"):

            result["policy_rationale"] = (
                line.split("POLICY_RATIONALE:", 1)[-1]
                .strip()
            )

    return result


# ============================================================
# POLICY ADVISOR AGENT
# ============================================================

class PolicyAdvisorAgent:

    def __init__(self):

        # =====================================================
        # GOOGLE API VALIDATION
        # =====================================================

        if not GOOGLE_API_KEY:

            raise ValueError(
                "GOOGLE_API_KEY missing in .env"
            )

        # =====================================================
        # GEMINI LLM
        # =====================================================

        self.llm = ChatGoogleGenerativeAI(

            model=GEMINI_MODEL,

            temperature=LLM_TEMPERATURE,

            google_api_key=GOOGLE_API_KEY,
        )

        # =====================================================
        # PROMPT
        # =====================================================

        self.prompt = ChatPromptTemplate.from_messages([

            SystemMessagePromptTemplate.from_template(
                SYSTEM_PROMPT
            ),

            HumanMessagePromptTemplate.from_template(
                HUMAN_PROMPT
            ),
        ])

        # =====================================================
        # LCEL CHAIN
        # =====================================================

        self.chain = self.prompt | self.llm

        logger.success(
            f"PolicyAdvisorAgent initialised | model={GEMINI_MODEL}"
        )

    # ========================================================
    # MAIN RUN
    # ========================================================

    def run(
        self,
        research_result: ResearchResult
    ) -> InsuranceReport:

        logger.info("=" * 60)

        logger.info("  Agent 3 — Policy Advisor")

        logger.info("=" * 60)

        # ====================================================
        # RISK CONTEXT
        # ====================================================

        rc = research_result.risk_context

        risk = rc.risk_label.upper()

        # ====================================================
        # BUSINESS RULES
        # ====================================================

        premium_adjustment = PREMIUM_ADJUSTMENTS.get(
            risk,
            0
        )

        claim_probability = CLAIM_PROBABILITIES.get(
            risk,
            0
        )

        urgency_level = max(

            rc.urgency_level,

            URGENCY_FLOOR.get(risk, 1)
        )

        coverage_changes = (

            rc.coverage_notes

            or COVERAGE_RECOMMENDATIONS.get(
                risk,
                []
            )
        )

        wellness_program = _WELLNESS_PROGRAMS.get(

            risk,

            "Standard wellness program"
        )

        clinical_referral = risk in (

            "HIGH",
            "CRITICAL"
        )

        # ====================================================
        # PAYLOAD
        # ====================================================

        payload = {

            "patient_name":
                rc.patient_name,

            "patient_age":
                rc.patient_age,

            "risk_level":
                risk,

            "urgency_level":
                urgency_level,

            "medical_summary":
                rc.medical_summary,

            "synthesised_finding":
                research_result.synthesised_finding,

            "guidelines":
                "\n".join(
                    research_result.key_guidelines
                ),

            "clinical_actions":
                "\n".join(
                    research_result.clinical_actions
                ),

            "insurance_implications":
                "\n".join(
                    research_result.insurance_implications
                ),

            "premium_adjustment":
                premium_adjustment,

            "claim_probability":
                claim_probability,

            "coverage_changes":
                ", ".join(coverage_changes),

            "wellness_program":
                wellness_program,
        }

        # ====================================================
        # LLM CALL WITH RETRIES
        # ====================================================

        raw_text = ""

        llm_ok = False

        for attempt in range(1, 4):

            try:

                response = self.chain.invoke(
                    payload
                )

                raw_text = response.content

                llm_ok = True

                logger.debug(
                    f"LLM succeeded on attempt {attempt}"
                )

                break

            except Exception as e:

                logger.warning(
                    f"LLM attempt {attempt}/3 failed: {e}"
                )

        # ====================================================
        # FALLBACK LOGGING
        # ====================================================

        if not llm_ok:

            logger.error(
                "All LLM attempts failed. "
                "Using deterministic fallback."
            )

        # ====================================================
        # PARSE RESPONSE
        # ====================================================

        parsed = (
            _parse_response(raw_text)
            if raw_text
            else {}
        )

        # ====================================================
        # FALLBACKS
        # ====================================================

        if not parsed.get("agent_finding"):

            parsed["agent_finding"] = (

                f"The patient's {risk} physiological risk "

                f"profile requires structured insurance review "

                f"according to current cardiovascular monitoring "

                f"and underwriting guidance."
            )

        if not parsed.get("immediate_actions"):

            parsed["immediate_actions"] = [

                "Review policy underwriting profile immediately",

                "Initiate preventive cardiac monitoring review",

                "Notify insurance case management team",

                "Evaluate additional rider eligibility",
            ]

        if not parsed.get("policy_rationale"):

            parsed["policy_rationale"] = (

                f"The patient's {risk} physiological risk level "

                f"indicates elevated future healthcare utilisation "

                f"probability. Preventive monitoring and adjusted "

                f"coverage are recommended to reduce long-term "

                f"claim exposure."
            )

        # ====================================================
        # REPORT TEXT
        # ====================================================

        report_text = self._format_report(

            patient_name=rc.patient_name,

            patient_age=rc.patient_age,

            risk=risk,

            urgency=urgency_level,

            premium=premium_adjustment,

            claim_probability=claim_probability,

            coverage_changes=coverage_changes,

            wellness_program=wellness_program,

            clinical_referral=clinical_referral,

            finding=parsed["agent_finding"],

            actions=parsed["immediate_actions"],

            rationale=parsed["policy_rationale"],
        )

        # ====================================================
        # FINAL REPORT OBJECT
        # ====================================================

        report = InsuranceReport(

            patient_name=rc.patient_name,

            patient_age=rc.patient_age,

            risk_level=risk,

            urgency_level=urgency_level,

            premium_adjustment=premium_adjustment,

            claim_probability=claim_probability,

            coverage_changes=coverage_changes,

            agent_finding=parsed["agent_finding"],

            immediate_actions=parsed["immediate_actions"],

            policy_rationale=parsed["policy_rationale"],

            clinical_referral=clinical_referral,

            wellness_program=wellness_program,

            evidence_strength=_EVIDENCE_STRENGTH.get(
                risk,
                "MEDIUM"
            ),

            report_text=report_text,

            created_at=datetime.now(
                timezone.utc
            ).isoformat(),
        )

        logger.success(

            f"PolicyAdvisorAgent complete | "

            f"risk={risk} | "

            f"urgency={urgency_level}/5 | "

            f"llm={'OK' if llm_ok else 'FALLBACK'}"
        )

        return report

    # ========================================================
    # REPORT FORMATTER
    # ========================================================

    def _format_report(

        self,

        patient_name,

        patient_age,

        risk,

        urgency,

        premium,

        claim_probability,

        coverage_changes,

        wellness_program,

        clinical_referral,

        finding,

        actions,

        rationale,
    ) -> str:

        lines = [

            "=" * 62,

            " SMARTGUARD AI — INSURANCE POLICY REPORT",

            "=" * 62,

            f"Patient Name       : {patient_name}",

            f"Patient Age        : {patient_age}",

            f"Risk Level         : {risk}",

            f"Urgency Level      : {urgency}/5",

            f"Premium Adjustment : +{premium}%",

            f"Claim Probability  : {claim_probability}%",

            f"Clinical Referral  : "
            f"{'YES' if clinical_referral else 'NO'}",

            "",

            "AGENT FINDING:",

            finding,

            "",

            "IMMEDIATE ACTIONS:",
        ]

        for i, action in enumerate(actions, 1):

            lines.append(
                f"{i}. {action}"
            )

        lines += [

            "",

            "COVERAGE CHANGES:",
        ]

        for item in coverage_changes:

            lines.append(
                f"• {item}"
            )

        lines += [

            "",

            "WELLNESS PROGRAM:",

            wellness_program,

            "",

            "POLICY RATIONALE:",

            rationale,

            "",

            "=" * 62,
        ]

        return "\n".join(lines)