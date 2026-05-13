# SmartGuard AI

Real-time health risk signals, deep learning (CNN + LSTM), multi-agent LangChain + Gemini analysis, MQTT simulation, and a Streamlit dashboard.

## Prerequisites

- macOS **Apple Silicon** (arm64), Python **3.10.x**

**Important:** The interpreter itself must be **arm64**, not x86_64 under Rosetta. Check with:

`python3.10 -c "import platform; print(platform.machine())"` → expect **`arm64`**.

If you see **`x86_64`** on an M-series Mac, your `python3.10` is **Intel-only** (Rosetta). `pip` then skips `tensorflow-macos` and TensorFlow crashes (AVX). **`arch -arm64 python3.10`** does **not** fix that — it fails with *Bad CPU type* because the binary cannot run as arm64.

**Fix:** install Apple Silicon Homebrew Python 3.10, then create the venv with that binary:

```bash
brew install python@3.10
bash scripts/bootstrap_macos_arm64_venv.sh
```

Or manually: `/opt/homebrew/opt/python@3.10/bin/python3.10 -m venv .venv` then `pip install -r requirements.txt`.

- Google AI Studio API key (Gemini)
- Optional: SerpAPI key (web research; otherwise fallback text is used)

## Environment setup

```bash
cd SMARTGUARD_AI
bash scripts/bootstrap_macos_arm64_venv.sh
source .venv/bin/activate
```

The script prefers **arm64** Python from `/opt/homebrew` and installs **`requirements.txt`**. If that Python is missing (common before you install Apple Silicon Homebrew), it falls back to **`requirements-agents-ui.txt`** (agents + Streamlit + MQTT, **no TensorFlow**) and creates **`.venv/SMARTGUARD_NO_TF`**. `python check_setup.py` skips the TensorFlow check in that mode.

Installing Homebrew to `/opt/homebrew` needs a normal Terminal run (may prompt for **sudo**): see [brew.sh](https://brew.sh). Automated installs cannot supply your password.

On Linux or Intel Mac, use `python3.10 -m venv .venv` and `pip install -r requirements.txt` as usual.

**TensorFlow on Apple Silicon:** `requirements.txt` installs `tensorflow-macos` and `tensorflow-metal` only on `darwin` + `arm64`. On other platforms it installs standard `tensorflow==2.15.0`.

**Reproducible installs:** after a successful install, refresh the lock with `pip freeze > requirements.lock.txt` and restore the three `#` comment lines at the top (see the committed file). Recreate the same tree with: `pip install -r requirements.lock.txt`.

```bash
cp .env.example .env
# Edit .env: set GOOGLE_API_KEY and optionally SERPAPI_API_KEY
python check_setup.py
```

## Generate deep learning assets (first time)

If `check_setup.py` reports missing models or processed data:

```bash
python deep_learning/preprocess.py
python deep_learning/lstm_model.py
python deep_learning/cnn_model.py
```

## Run order (end-to-end)

1. **Optional — MQTT subscriber** (writes CSV under `deep_learning/data/processed/`):

   ```bash
   python iot/mqtt_subscriber.py
   ```

2. **Optional — MQTT publisher** (simulated vitals):

   ```bash
   python iot/mqtt_publisher.py
   ```

3. **Full app (subscriber not required):**

   ```bash
   python main.py --mode full
   ```

   Other modes: `--mode dashboard`, `--mode iot`, `--mode pipeline`, `--mode test`.

4. **Deep learning inference smoke test:**

   ```bash
   python deep_learning/predict.py
   ```

5. **Agents only:**

   ```bash
   python agents/agent_pipeline.py
   ```

## Gemini model names

Default model is `gemini-2.5-flash` (see `config/settings.py`). If you see **429** with `generate_content_free_tier_requests` and **limit: 0** for `gemini-2.0-flash`, that model has no free-tier quota on your project—use a model that appears under [AI Studio rate limits](https://aistudio.google.com/app/rate-limit) for your tier (e.g. `gemini-2.5-flash-lite`). In `.env` set `GEMINI_MODEL` to an id your key supports. Bare `gemini-1.5-flash` often returns **404** on newer API keys; the `models/` prefix is stripped automatically if you paste a full resource name.

## Validation checklist

| Step | Command | Expected |
|------|---------|----------|
| Setup | `python check_setup.py` | Exit 0; optional warnings for SerpAPI / MQTT |
| Config | `python config/settings.py` | Prints paths and key status |
| LangChain | `python -c "from langchain_google_genai import ChatGoogleGenerativeAI; print('ok')"` | Prints `ok` |
| TensorFlow | `python -c "import tensorflow as tf; print(tf.__version__)"` | Version `2.15.0` (may take a few seconds) |
| Pipeline | `python main.py --mode test` | Four `SUCCESS` lines with report metrics |
| Predict | `python deep_learning/predict.py` | Risk / anomaly summary (requires trained assets) |

## Architecture (short)

- **`config/settings.py`** — paths, API keys, MQTT, Gemini model name.
- **`deep_learning/`** — preprocess, LSTM autoencoder, CNN classifier, `predict.py` unified inference.
- **`agents/`** — Risk reader → Web researcher (SerpAPI + Gemini) → Policy advisor; `agent_pipeline.py` orchestrates.
- **`iot/`** — HiveMQ-compatible publisher/subscriber and sensor simulator.
- **`dashboard/app.py`** — Streamlit UI calling `SmartGuardPipeline`.
