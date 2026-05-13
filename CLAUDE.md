# CLAUDE.md

## Project Overview

SmartGuard AI is an AI-powered health insurance risk assessment and underwriting assistance system.

The system continuously monitors physiological signals using IoT sensor streams, processes them through deep learning models, and generates evidence-backed insurance recommendations using AI agents.

Pipeline:

IoT Sensors → MQTT Streaming → Deep Learning → AI Agents → Insurance Insights

Core components:
- IoT simulation using MQTT
- LSTM anomaly detection
- CNN risk classification
- LangChain multi-agent pipeline
- Live medical web research via SerpAPI
- Streamlit dashboard

This project does NOT autonomously approve or reject insurance claims or policies. It acts as a decision-support and underwriting intelligence system.

---

## Architecture

Folders:
- `iot/` → sensor simulation + MQTT pipeline
- `deep_learning/` → preprocessing, training, inference
- `agents/` → LangChain agent orchestration
- `dashboard/` → Streamlit frontend
- `models/` → saved `.keras` models
- `config/` → settings and environment configs

Main entry:
- `main.py`

---

## Development Principles

Claude should:

- Preserve modular architecture
- Maintain end-to-end pipeline integrity
- Keep components loosely coupled
- Prefer explainable systems over black-box behavior
- Never introduce fake medical certainty
- Treat all recommendations as assistive, not authoritative
- Avoid hardcoding sensitive logic
- Keep the system deterministic where possible

---

## AI System Rules

LLMs are used for:
- interpretation
- orchestration
- summarization
- recommendation generation

LLMs are NOT used for:
- raw medical diagnosis
- replacing clinical judgment
- unsupported claim decisions

All recommendations should be grounded in:
- model outputs
- retrieved evidence
- configurable business rules

---

## Coding Guidelines

- Keep functions small and isolated
- Avoid unnecessary abstractions
- Prefer readability over cleverness
- Add comments only where logic is non-obvious
- Keep naming explicit and consistent
- Use environment variables for secrets/API keys
- Do not break MQTT or agent orchestration flow

---

## Expected Workflow

1. Sensor data generated
2. MQTT publishes stream
3. Subscriber receives data
4. Data preprocessing
5. LSTM anomaly analysis
6. CNN risk classification
7. Agent pipeline executes
8. Insurance insights generated
9. Dashboard updated

---

## Important Constraints

- Project currently uses simulated IoT data
- Clinical recommendations are informational
- Real deployment would require:
  - validated datasets
  - compliance review
  - secure infrastructure
  - medical oversight

Keep all future improvements aligned with real-world deployability.