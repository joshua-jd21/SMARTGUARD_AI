# =============================================================
#  SmartGuard AI — MQTT Subscriber
#  File: iot/mqtt_subscriber.py
#
#  PURPOSE:
#  Listens to the MQTT broker and receives every reading
#  published by the publisher. Stores readings in memory
#  and writes them to a CSV file for the DL module to use.
#
#  Think of this as the "cloud receiver" — in production
#  this would run on a server, not the patient's device.
#
#  Run this in a SEPARATE terminal from the publisher.
# =============================================================

import json
import csv
import os
import sys
import time
from datetime import datetime
from collections import deque
import paho.mqtt.client as mqtt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, MQTT_CLIENT_ID,
    ANOMALY_THRESHOLD, PROCESSED_DATA_DIR
)


# ──────────────────────────────────────────────────────────────
#  STORAGE SETUP
#  We store received readings in two places:
#  1. In-memory deque  → fast access for the DL pipeline
#  2. CSV file         → persistent storage for training data
# ──────────────────────────────────────────────────────────────

# In-memory buffer: stores the last 200 readings
# deque with maxlen automatically drops oldest when full
readings_buffer = deque(maxlen=200)

# CSV file path for persistent storage
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
CSV_FILE = os.path.join(PROCESSED_DATA_DIR, "sensor_readings.csv")

# CSV column headers
CSV_HEADERS = [
    "timestamp", "patient_id", "patient_name",
    "heart_rate", "spo2", "temperature", "activity", "is_anomaly"
]

# Track total readings received
readings_count = 0


# ──────────────────────────────────────────────────────────────
#  CSV WRITER
#  Appends one reading to the CSV file.
#  Creates the file with headers if it doesn't exist yet.
# ──────────────────────────────────────────────────────────────

def write_to_csv(reading):
    """Append a reading dictionary to the CSV file."""
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)

        # Write header row only when creating the file fresh
        if not file_exists:
            writer.writeheader()

        # Write only the columns we care about (ignore extra keys)
        row = {key: reading[key] for key in CSV_HEADERS if key in reading}
        writer.writerow(row)


# ──────────────────────────────────────────────────────────────
#  ALERT HANDLER
#  Called whenever an anomaly reading is received.
#  In production this would trigger a notification to the
#  insurance system or a clinical alert.
# ──────────────────────────────────────────────────────────────

def handle_alert(reading):
    """Handle an anomalous reading — log it and raise alert."""
    ts = reading["timestamp"][11:19]
    print(f"\n{'!'*55}")
    print(f"  🚨 ALERT DETECTED — {ts}")
    print(f"  Patient    : {reading['patient_name']} ({reading['patient_id']})")
    print(f"  Heart Rate : {reading['heart_rate']} bpm  ← ELEVATED")
    print(f"  SpO2       : {reading['spo2']} %  ← LOW")
    print(f"  Temperature: {reading['temperature']} °F")
    print(f"  Activity   : {reading['activity']}")
    print(f"  → Forwarding to Deep Learning pipeline...")
    print(f"{'!'*55}\n")


# ──────────────────────────────────────────────────────────────
#  MQTT CALLBACKS
# ──────────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, rc):
    """Called when subscriber connects to broker."""
    if rc == 0:
        print(f"[MQTT Subscriber] ✅ Connected to {MQTT_BROKER}")
        print(f"[MQTT Subscriber] 👂 Listening on topic: {MQTT_TOPIC}\n")
        print("-" * 55)
        # Subscribe to our topic after connecting
        # QoS 1 = at least once delivery
        client.subscribe(MQTT_TOPIC, qos=1)
    else:
        print(f"[MQTT Subscriber] ❌ Connection failed. Code: {rc}")


def on_message(client, userdata, msg):
    """
    Called automatically every time a new message arrives.
    This is the heart of the subscriber — processes each reading.
    """
    global readings_count

    try:
        # Decode JSON payload back to Python dictionary
        payload = msg.payload.decode("utf-8")
        reading = json.loads(payload)

        readings_count += 1

        # Store in memory buffer
        readings_buffer.append(reading)

        # Store in CSV file
        write_to_csv(reading)

        # Display status in terminal
        ts = reading["timestamp"][11:19]
        status = "🚨 ANOMALY" if reading["is_anomaly"] else "✅ OK"

        print(
            f"[{ts}] #{readings_count:04d} | "
            f"HR: {reading['heart_rate']:5.1f} bpm | "
            f"SpO2: {reading['spo2']:5.1f}% | "
            f"Temp: {reading['temperature']:5.1f}°F | "
            f"Activity: {reading['activity']:4s} | "
            f"{status}"
        )

        # Trigger alert pipeline if anomaly detected
        if reading["is_anomaly"]:
            handle_alert(reading)

    except json.JSONDecodeError as e:
        print(f"[MQTT Subscriber] ❌ Bad JSON received: {e}")
    except KeyError as e:
        print(f"[MQTT Subscriber] ❌ Missing field in reading: {e}")


def on_disconnect(client, userdata, rc):
    """Called when subscriber disconnects."""
    print(f"\n[MQTT Subscriber] Disconnected. Total received: {readings_count}")


# ──────────────────────────────────────────────────────────────
#  PUBLIC API
#  Functions used by other modules (e.g. dashboard, DL module)
#  to access the received data.
# ──────────────────────────────────────────────────────────────

def get_latest_reading():
    """Returns the most recently received reading, or None."""
    if readings_buffer:
        return readings_buffer[-1]
    return None


def get_recent_readings(n=30):
    """Returns the last N readings as a list."""
    return list(readings_buffer)[-n:]


def get_all_readings():
    """Returns all readings currently in the buffer."""
    return list(readings_buffer)


# ──────────────────────────────────────────────────────────────
#  SUBSCRIBER CLASS
# ──────────────────────────────────────────────────────────────

class SensorSubscriber:
    """
    Connects to MQTT broker and listens for sensor readings.
    Stores everything received for downstream processing.
    """

    def __init__(self):
        self.client = mqtt.Client(client_id=MQTT_CLIENT_ID)
        self.client.on_connect    = on_connect
        self.client.on_message    = on_message
        self.client.on_disconnect = on_disconnect

    def start(self):
        """Connect and start listening — runs forever until Ctrl+C."""
        print("=" * 55)
        print("  SmartGuard AI — MQTT Subscriber")
        print(f"  Saving data to: {CSV_FILE}")
        print("  Press Ctrl+C to stop.")
        print("=" * 55 + "\n")

        try:
            self.client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            # loop_forever() blocks here and handles all MQTT traffic
            # It automatically reconnects if connection drops
            self.client.loop_forever()

        except KeyboardInterrupt:
            print(f"\n[MQTT Subscriber] Stopped. {readings_count} readings saved to {CSV_FILE}")
            self.client.disconnect()

        except Exception as e:
            print(f"[MQTT Subscriber] ❌ Error: {e}")
            self.client.disconnect()


# ──────────────────────────────────────────────────────────────
#  STANDALONE RUN
#  Open a NEW terminal and run: python iot/mqtt_subscriber.py
#  Keep mqtt_publisher.py running in the other terminal.
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    subscriber = SensorSubscriber()
    subscriber.start()