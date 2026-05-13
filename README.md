---
title: SmartGuard AI
emoji: 🩺
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: AI-assisted health risk + insurance intelligence (cloud demo)
---

# SmartGuard AI

AI-powered **health risk telemetry and underwriting decision-support** built on a
hybrid edge + cloud architecture. SmartGuard AI continuously monitors
physiological signals, runs deep-learning risk classification on the edge, and
synthesises evidence-backed insurance recommendations using a LangChain
multi-agent pipeline.

> SmartGuard AI is **decision-support only**. It never auto-approves or
> auto-rejects insurance claims or policies. All recommendations require a
> human reviewer.

---

## Two deployment surfaces

This repository ships in two complementary forms:

| Surface | Where it lives | What it does | Runtime cost |
|---|---|---|---|
| **Local edge layer** | `deep_learning/`, `agents/`, `iot/` | Real-time MQTT ingestion, LSTM anomaly detection, CNN risk classification, LangChain agents (Gemini + SerpAPI) | Heavy (TensorFlow, LLM API quota) |
| **Cloud presentation layer** | `app.py`, `demo/` (this branch) | Lightweight SaaS-style dashboard, monitoring rollups, report visualization | Free-tier compatible |

The cloud layer is intentionally decoupled from the heavy compute stack so it
can run on **Hugging Face Spaces** and **Streamlit Community Cloud** with zero
secrets and zero runtime API calls. The local edge layer is untouched and
continues to run the full pipeline as documented below.

---

## Cloud demo — quick deploy

The `cloud-demo` branch is purpose-built for free hosting tiers.

### Hugging Face Spaces (Docker SDK)

1. Create a new Space → **SDK: Docker**.
2. Push or link this branch as the Space repo. The remote is conventionally:
   ```bash
   git remote add hf https://huggingface.co/spaces/<user>/<space-name>
   git push hf cloud-demo:main
   ```
3. The YAML front-matter at the top of this README already configures Spaces:
   - `sdk: docker`
   - `app_port: 7860`
4. HF Spaces builds the included `Dockerfile`, which:
   - Uses `python:3.10-slim` (small base image, fast cold start).
   - Installs only the 5 cloud requirements.
   - Copies the repo (minus everything excluded by `.dockerignore` — i.e. local
     heavy modules like `agents/`, `deep_learning/`, `iot/`, `models/`,
     `dashboard/`, `scripts/`, `main.py`).
   - Starts Streamlit on `0.0.0.0:7860`.
5. No secrets, no environment variables required — the demo runs end-to-end on
   the bundled `demo/` assets.

### Local Docker build

```bash
docker build -t smartguard-ai:cloud-demo .
docker run --rm -p 7860:7860 smartguard-ai:cloud-demo
# open http://localhost:7860
```

The built image contains **only** the cloud presentation layer — the local
edge stack (`agents/`, `deep_learning/`, `iot/`, etc.) is excluded by
`.dockerignore` so production images stay small and free of dual-use
PHI-handling code paths.

### Streamlit Community Cloud

1. Connect this repository on [streamlit.io/cloud](https://streamlit.io/cloud).
2. Select branch `cloud-demo`.
3. Main file path: `app.py`.
4. Deploy. The app boots in seconds with no external dependencies.

### Local sanity check

```bash
pip install -r requirements.txt
streamlit run app.py
```

The cloud `requirements.txt` is the **lightweight set** (Streamlit, pandas,
numpy, plotly, python-dotenv). The full local stack is preserved in
`requirements-local.txt`.

---

## What the cloud dashboard renders

`app.py` reads pre-computed snapshots from `demo/`:

- `demo/sample_prediction.json` — multi-patient deep-learning predictions
  (LOW / MEDIUM / HIGH / CRITICAL) with anomaly scores and vitals snapshots.
- `demo/sample_report.json` — multi-agent insurance decision-support reports
  (claim probability, premium adjustment, coverage recommendations, sources).
- `demo/sample_vitals.csv` — 2-hour telemetry window per patient for live-looking
  charts.

Tabs available in the cloud dashboard:

1. **Overview** — portfolio risk distribution, fleet snapshot KPIs.
2. **Live Monitoring** — per-patient telemetry with anomaly markers.
3. **Insurance Reports** — multi-agent decision-support reports with sources.
4. **Architecture** — hybrid edge + cloud topology.
5. **AI Workflow** — three-agent pipeline explanation.
6. **Deployment** — build metadata + hosting recipes.

The cloud layer:

- Does **not** import TensorFlow, Keras, LangChain, or `paho-mqtt`.
- Does **not** call Gemini, SerpAPI, or any external service.
- Does **not** open MQTT loops or background threads.
- Loads only `streamlit`, `pandas`, `numpy`, `plotly`, `python-dotenv`.

This is the explicit cloud-vs-local execution contract — see *Architecture*
below.

---

## Local full pipeline — preserved as-is

The full local execution path is unchanged and remains the source of truth.
For complete local instructions (macOS Apple Silicon, TensorFlow + Gemini +
SerpAPI + MQTT setup), see [`README-local.md`](#local-setup-detailed) section
below.

### Local setup (detailed)

**Prerequisites**

- macOS **Apple Silicon** (arm64) or Linux, Python **3.10.x**.
- Google AI Studio API key (Gemini). Optional: SerpAPI key.

**Important on Apple Silicon:** the interpreter must be **arm64**.
Check with `python3.10 -c "import platform; print(platform.machine())"` → expect `arm64`.

```bash
brew install python@3.10
bash scripts/bootstrap_macos_arm64_venv.sh
source .venv/bin/activate

# Install the FULL local stack (NOT the cloud-lite requirements.txt):
pip install -r requirements-local.txt

cp .env.example .env
# Edit .env: set GOOGLE_API_KEY (and optionally SERPAPI_API_KEY)
python check_setup.py
```

**Generate deep-learning assets (first time)**

```bash
python deep_learning/preprocess.py
python deep_learning/lstm_model.py
python deep_learning/cnn_model.py
```

**Run end-to-end**

```bash
# Optional MQTT subscriber:
python iot/mqtt_subscriber.py

# Optional MQTT publisher:
python iot/mqtt_publisher.py

# Full app:
python main.py --mode full
# other modes: --mode dashboard, --mode iot, --mode pipeline, --mode test
```

**Deep-learning inference smoke test**

```bash
python deep_learning/predict.py
```

**Agents only**

```bash
python agents/agent_pipeline.py
```

---

## Architecture

```
                          ┌───────────────────────────────────────┐
                          │            LOCAL EDGE LAYER           │
                          │  (heavy compute, customer-owned infra)│
                          │                                       │
   IoT sensors ─MQTT──►  preprocess → LSTM (anomaly) → CNN (risk) │
                          │                 │                     │
                          │                 ▼                     │
                          │  LangChain agents (RiskReader →       │
                          │  WebResearcher → PolicyAdvisor)       │
                          │  Gemini + SerpAPI                     │
                          └───────────────┬───────────────────────┘
                                          │ decision-support payload
                                          ▼
                          ┌───────────────────────────────────────┐
                          │          CLOUD PRESENTATION           │
                          │   (lightweight, free-tier hostable)   │
                          │                                       │
                          │   Streamlit dashboard (app.py)        │
                          │   demo/*.json + demo/*.csv snapshots  │
                          └───────────────────────────────────────┘
```

**Folders**

- `iot/` — sensor simulation + MQTT pipeline (local-only).
- `deep_learning/` — preprocessing, LSTM autoencoder, CNN classifier, inference (local-only).
- `agents/` — LangChain multi-agent orchestration (local-only).
- `dashboard/app.py` — original full-feature Streamlit dashboard that drives the local pipeline.
- `app.py` — **cloud entrypoint** (this branch) — lightweight, demo-only.
- `demo/` — static snapshots that power the cloud dashboard.
- `config/`, `models/`, `data/`, `scripts/` — unchanged.

**Why two layers?**

- **Compliance** — PHI stays on the edge; only de-identified decision-support payloads cross the boundary.
- **Cost** — GPU/inference cost bounded to edge nodes; the cloud layer scales cheaply.
- **Reliability** — the dashboard stays up even when the edge pipeline is offline.
- **Hackathon-friendly** — instantly deployable presentation without standing up the full compute stack.

---

## Cloud-vs-local execution contract

| Capability | Local (`main` branch / `requirements-local.txt`) | Cloud (`cloud-demo` branch / `requirements.txt`) |
|---|---|---|
| TensorFlow / Keras inference | ✅ live | ❌ never imported |
| MQTT streaming loop | ✅ live | ❌ never opened |
| LangChain agent execution | ✅ live | ❌ never invoked |
| Gemini API calls | ✅ live | ❌ no API key required |
| SerpAPI calls | ✅ optional | ❌ never invoked |
| Streamlit dashboard | ✅ `dashboard/app.py` | ✅ `app.py` (this file) |
| Free-tier deployable | ❌ (requires GPU + secrets) | ✅ HF Spaces / Streamlit Cloud |
| Startup time | ~seconds–minutes (TF init) | <2s |

---

## Hackathon-readiness checklist

- ✅ Fast cold start (no model loading, no API warmup).
- ✅ Zero environment variables required for cloud deployment.
- ✅ Zero external network calls at runtime.
- ✅ Deterministic, demo-friendly output (no LLM variance).
- ✅ Original architecture preserved — local pipeline untouched.
- ✅ Single-command deploy: `streamlit run app.py`.

---

## Gemini model names (local only)

Default model is `gemini-2.5-flash` (see `config/settings.py`). If you hit a
429 with `generate_content_free_tier_requests` and `limit: 0`, that model has
no free-tier quota on your project — switch to one listed in
[AI Studio rate limits](https://aistudio.google.com/app/rate-limit) (e.g.
`gemini-2.5-flash-lite`). Set `GEMINI_MODEL` in `.env` accordingly.

---

## Validation checklist

| Step | Command | Expected |
|------|---------|----------|
| Cloud app | `streamlit run app.py` | Dashboard renders in <2s, no API calls |
| Cloud deps | `pip install -r requirements.txt` | Installs only Streamlit + pandas + numpy + plotly + dotenv |
| Local setup | `python check_setup.py` | Exit 0; optional warnings for SerpAPI / MQTT |
| Local pipeline | `python main.py --mode test` | Four `SUCCESS` lines with report metrics |
| Local predict | `python deep_learning/predict.py` | Risk / anomaly summary |

---

## License

MIT — see project root.
