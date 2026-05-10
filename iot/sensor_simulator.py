# =============================================================
#  SmartGuard AI — Sensor Simulator (Upgraded v2)
#  File: iot/sensor_simulator.py
#
#  WHAT'S NEW vs v1:
#  ✅ PatientState enum — 8 distinct health states
#  ✅ SensorSimulator class — stateful, realistic transitions
#  ✅ Markov-style state machine — states flow naturally
#  ✅ All v1 functions kept — mqtt_publisher.py still works
# =============================================================

import random
import numpy as np
from datetime import datetime
from enum import Enum, auto
import sys, os, json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    PATIENT_ID, PATIENT_NAME,
    NORMAL_HEART_RATE_RANGE, NORMAL_SPO2_RANGE, NORMAL_TEMPERATURE_RANGE,
    ANOMALY_HEART_RATE_RANGE, ANOMALY_SPO2_RANGE, ANOMALY_TEMPERATURE_RANGE,
    ACTIVITY_LEVELS, ANOMALY_INJECTION_RATE,
    BASELINE_HEART_RATE, BASELINE_SPO2, BASELINE_TEMPERATURE
)


# ──────────────────────────────────────────────────────────────
#  PATIENT STATE ENUM
#  Risk mapping:
#  RESTING, WALKING, SLEEPING, RUNNING → label 0 (LOW)
#  STRESSED, ANOMALY_MILD              → label 1 (MEDIUM)
#  ANOMALY_HIGH                        → label 2 (HIGH)
#  CRITICAL                            → label 3 (CRITICAL)
# ──────────────────────────────────────────────────────────────

class PatientState(Enum):
    RESTING      = auto()
    WALKING      = auto()
    SLEEPING     = auto()
    RUNNING      = auto()
    STRESSED     = auto()
    ANOMALY_MILD = auto()
    ANOMALY_HIGH = auto()
    CRITICAL     = auto()


# Vital sign ranges per state: (heart_rate, spo2, temperature)
STATE_VITALS = {
    PatientState.RESTING:      ((60,  75),  (96, 99),  (97.5, 98.5)),
    PatientState.WALKING:      ((75,  100), (95, 98),  (98.0, 99.0)),
    PatientState.SLEEPING:     ((48,  64),  (95, 98),  (97.0, 98.2)),
    PatientState.RUNNING:      ((120, 160), (94, 97),  (99.0, 101.0)),
    PatientState.STRESSED:     ((90,  115), (94, 97),  (98.5, 99.5)),
    PatientState.ANOMALY_MILD: ((100, 122), (91, 94),  (99.0, 100.5)),
    PatientState.ANOMALY_HIGH: ((115, 138), (88, 92),  (100.0, 102.0)),
    PatientState.CRITICAL:     ((130, 165), (82, 89),  (101.5, 104.0)),
}

STATE_ACTIVITY = {
    PatientState.RESTING:      "rest",
    PatientState.WALKING:      "walk",
    PatientState.SLEEPING:     "rest",
    PatientState.RUNNING:      "run",
    PatientState.STRESSED:     "rest",
    PatientState.ANOMALY_MILD: "rest",
    PatientState.ANOMALY_HIGH: "rest",
    PatientState.CRITICAL:     "rest",
}

# Markov transition matrix — probabilities must sum to 1.0 per row
STATE_TRANSITIONS = {
    PatientState.RESTING: [
        (PatientState.RESTING,      0.50),
        (PatientState.WALKING,      0.20),
        (PatientState.SLEEPING,     0.15),
        (PatientState.STRESSED,     0.10),
        (PatientState.ANOMALY_MILD, 0.05),
    ],
    PatientState.WALKING: [
        (PatientState.WALKING,      0.40),
        (PatientState.RESTING,      0.30),
        (PatientState.RUNNING,      0.15),
        (PatientState.STRESSED,     0.10),
        (PatientState.ANOMALY_MILD, 0.05),
    ],
    PatientState.SLEEPING: [
        (PatientState.SLEEPING,     0.60),
        (PatientState.RESTING,      0.25),
        (PatientState.ANOMALY_MILD, 0.10),
        (PatientState.CRITICAL,     0.05),
    ],
    PatientState.RUNNING: [
        (PatientState.RUNNING,      0.40),
        (PatientState.WALKING,      0.35),
        (PatientState.RESTING,      0.20),
        (PatientState.ANOMALY_MILD, 0.05),
    ],
    PatientState.STRESSED: [
        (PatientState.STRESSED,     0.35),
        (PatientState.RESTING,      0.25),
        (PatientState.ANOMALY_MILD, 0.25),
        (PatientState.WALKING,      0.10),
        (PatientState.ANOMALY_HIGH, 0.05),
    ],
    PatientState.ANOMALY_MILD: [
        (PatientState.ANOMALY_MILD, 0.30),
        (PatientState.RESTING,      0.25),
        (PatientState.ANOMALY_HIGH, 0.25),
        (PatientState.STRESSED,     0.15),
        (PatientState.CRITICAL,     0.05),
    ],
    PatientState.ANOMALY_HIGH: [
        (PatientState.ANOMALY_HIGH, 0.35),
        (PatientState.ANOMALY_MILD, 0.20),
        (PatientState.CRITICAL,     0.25),
        (PatientState.STRESSED,     0.10),
        (PatientState.RESTING,      0.10),
    ],
    PatientState.CRITICAL: [
        (PatientState.CRITICAL,     0.50),
        (PatientState.ANOMALY_HIGH, 0.30),
        (PatientState.ANOMALY_MILD, 0.15),
        (PatientState.RESTING,      0.05),
    ],
}


# ──────────────────────────────────────────────────────────────
#  SENSOR SIMULATOR CLASS
# ──────────────────────────────────────────────────────────────

class SensorSimulator:
    """
    Stateful IoT sensor simulator.
    Tracks patient state and transitions naturally over time.

    Usage:
        sim = SensorSimulator()
        reading = sim.read()    # get one reading dict
        state   = sim.state     # check current PatientState
    """

    def __init__(self, initial_state: PatientState = None):
        if initial_state is None:
            self._state = random.choice([
                PatientState.RESTING,
                PatientState.WALKING,
                PatientState.SLEEPING,
            ])
        else:
            self._state = initial_state

    @property
    def state(self) -> PatientState:
        return self._state

    def _transition(self):
        """Move to next state via Markov transition probabilities."""
        transitions = STATE_TRANSITIONS[self._state]
        states = [t[0] for t in transitions]
        probs  = [t[1] for t in transitions]
        self._state = random.choices(states, weights=probs, k=1)[0]

    def _generate_vitals(self) -> dict:
        """Generate vitals for current state with Gaussian noise."""
        hr_range, spo2_range, temp_range = STATE_VITALS[self._state]
        heart_rate  = round(random.uniform(*hr_range)   + np.random.normal(0, 1.5),  1)
        spo2        = round(min(random.uniform(*spo2_range) + np.random.normal(0, 0.3), 100.0), 1)
        temperature = round(random.uniform(*temp_range) + np.random.normal(0, 0.15), 1)
        return {
            "heart_rate" : heart_rate,
            "spo2"       : spo2,
            "temperature": temperature,
            "activity"   : STATE_ACTIVITY[self._state],
        }

    def read(self) -> dict:
        """
        Generate one reading for current state, then transition.
        Returns a full reading dict with all vital signs.
        """
        vitals     = self._generate_vitals()
        is_anomaly = self._state in {
            PatientState.ANOMALY_MILD,
            PatientState.ANOMALY_HIGH,
            PatientState.CRITICAL,
        }
        reading = {
            "patient_id"  : PATIENT_ID,
            "patient_name": PATIENT_NAME,
            "timestamp"   : datetime.now().isoformat(),
            "heart_rate"  : vitals["heart_rate"],
            "spo2"        : vitals["spo2"],
            "temperature" : vitals["temperature"],
            "activity"    : vitals["activity"],
            "state"       : self._state.name,
            "is_anomaly"  : is_anomaly,
        }
        self._transition()   # move to next state after reading
        return reading


# ──────────────────────────────────────────────────────────────
#  V1 COMPATIBILITY — mqtt_publisher.py still works unchanged
# ──────────────────────────────────────────────────────────────

def generate_reading(force_anomaly=False):
    sim = SensorSimulator()
    if force_anomaly:
        sim._state = PatientState.CRITICAL
    return sim.read()

def display_reading(reading):
    status = "🚨 ANOMALY" if reading["is_anomaly"] else "✅ NORMAL"
    ts     = reading["timestamp"][11:19]
    print(f"\n[SmartGuard IoT]  {ts}  |  Patient: {reading['patient_name']}")
    print(f"  State       : {reading.get('state', 'UNKNOWN')}")
    print(f"  Heart Rate  : {reading['heart_rate']} bpm")
    print(f"  SpO2        : {reading['spo2']} %")
    print(f"  Temperature : {reading['temperature']} °F")
    print(f"  Activity    : {reading['activity']}")
    print(f"  Status      : {status}")

def stream_readings(force_anomaly=False):
    sim = SensorSimulator()
    if force_anomaly:
        sim._state = PatientState.CRITICAL
    while True:
        yield sim.read()

def generate_batch(n_readings=1000):
    sim = SensorSimulator()
    return [sim.read() for _ in range(n_readings)]


# ──────────────────────────────────────────────────────────────
#  STANDALONE TEST — python iot/sensor_simulator.py
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("  SmartGuard AI — Upgraded Sensor Simulator v2")
    print("=" * 60)

    print("\n--- 10 stateful readings (watch state evolve) ---")
    sim = SensorSimulator(initial_state=PatientState.RESTING)
    for i in range(10):
        r = sim.read()
        display_reading(r)
        time.sleep(0.2)

    print("\n--- All 8 states sampled ---")
    for state in PatientState:
        sim2 = SensorSimulator(initial_state=state)
        r = sim2.read()
        print(
            f"  {state.name:15s} | "
            f"HR: {r['heart_rate']:6.1f} | "
            f"SpO2: {r['spo2']:5.1f}% | "
            f"Temp: {r['temperature']:5.1f}°F | "
            f"Anomaly: {r['is_anomaly']}"
        )

    print("\n✅ Simulator v2 ready. preprocess.py imports will now work.")