# =============================================================
#  SmartGuard AI — MQTT Subscriber
#  File: iot/mqtt_subscriber.py
#
#  PURPOSE:
#  Listens to the MQTT broker and receives every reading
#  published by the publisher. Stores readings in memory
#  and writes them to a CSV file for the DL module to use.
#
#  Run this in a SEPARATE terminal from the publisher.
#
#  IMPROVEMENTS:
#  ✅ Thread-safe readings_count with Lock
#  ✅ Batched CSV writes (flush every 10 messages)
#     — avoids per-message file I/O overhead
# =============================================================

import csv
import json
import os
import sys
import threading
from collections import deque
from datetime import datetime

import paho.mqtt.client as mqtt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    MQTT_BROKER,
    MQTT_CLIENT_ID,
    MQTT_PORT,
    MQTT_TOPIC,
    PROCESSED_DATA_DIR,
)


# ──────────────────────────────────────────────────────────────
#  STORAGE SETUP
# ──────────────────────────────────────────────────────────────

# In-memory buffer: stores the last 200 readings
readings_buffer = deque(maxlen=200)

# CSV file path for persistent storage
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
CSV_FILE = os.path.join(PROCESSED_DATA_DIR, "sensor_readings.csv")

# CSV column headers
CSV_HEADERS = [
    "timestamp", "patient_id", "patient_name",
    "heart_rate", "spo2", "temperature", "activity", "is_anomaly"
]

# ──────────────────────────────────────────────────────────────
#  THREAD-SAFE COUNTER
# ──────────────────────────────────────────────────────────────

readings_count = 0
_counter_lock = threading.Lock()

# ──────────────────────────────────────────────────────────────
#  BATCHED CSV WRITER
#  Accumulates readings and flushes every _FLUSH_INTERVAL msgs.
#  Reduces file I/O from N writes → N/10 writes.
# ──────────────────────────────────────────────────────────────

_CSV_FLUSH_INTERVAL = 10
_write_buffer: list = []
_write_lock = threading.Lock()


def _flush_csv():
    """Write accumulated readings to CSV and clear the buffer."""
    global _write_buffer
    with _write_lock:
        if not _write_buffer:
            return
        file_exists = os.path.isfile(CSV_FILE)
        try:
            with open(CSV_FILE, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                if not file_exists:
                    writer.writeheader()
                for reading in _write_buffer:
                    row = {
                        key: reading[key]
                        for key in CSV_HEADERS
                        if key in reading
                    }
                    writer.writerow(row)
        except OSError as e:
            print(f"[MQTT Subscriber] ❌ CSV write error: {e}")
        _write_buffer = []


def queue_csv_write(reading):
    """Add reading to write buffer; flush when batch is full."""
    with _write_lock:
        _write_buffer.append(reading)
        should_flush = len(_write_buffer) >= _CSV_FLUSH_INTERVAL

    if should_flush:
        _flush_csv()


# ──────────────────────────────────────────────────────────────
#  ALERT HANDLER
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
        client.subscribe(MQTT_TOPIC, qos=1)
    else:
        print(f"[MQTT Subscriber] ❌ Connection failed. Code: {rc}")


def on_message(client, userdata, msg):
    """
    Called automatically every time a new message arrives.
    Thread-safe counter + batched CSV writes.
    """
    global readings_count

    try:
        payload = msg.payload.decode("utf-8")
        reading = json.loads(payload)

        # Thread-safe counter increment
        with _counter_lock:
            readings_count += 1
            local_count = readings_count

        # Store in memory buffer
        readings_buffer.append(reading)

        # Batched CSV write (flushes every _CSV_FLUSH_INTERVAL msgs)
        queue_csv_write(reading)

        # Display status in terminal
        ts = reading["timestamp"][11:19]
        status = "🚨 ANOMALY" if reading["is_anomaly"] else "✅ OK"

        print(
            f"[{ts}] #{local_count:04d} | "
            f"HR: {reading['heart_rate']:5.1f} bpm | "
            f"SpO2: {reading['spo2']:5.1f}% | "
            f"Temp: {reading['temperature']:5.1f}°F | "
            f"Activity: {reading['activity']:4s} | "
            f"{status}"
        )

        if reading["is_anomaly"]:
            handle_alert(reading)

    except json.JSONDecodeError as e:
        print(f"[MQTT Subscriber] ❌ Bad JSON received: {e}")
    except KeyError as e:
        print(f"[MQTT Subscriber] ❌ Missing field in reading: {e}")


def on_disconnect(client, userdata, rc):
    """Flush remaining buffered writes on disconnect."""
    _flush_csv()
    print(f"\n[MQTT Subscriber] Disconnected. Total received: {readings_count}")


# ──────────────────────────────────────────────────────────────
#  PUBLIC API
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
            self.client.loop_forever()

        except KeyboardInterrupt:
            _flush_csv()  # Flush any remaining buffered writes
            print(
                f"\n[MQTT Subscriber] Stopped. "
                f"{readings_count} readings saved to {CSV_FILE}"
            )
            self.client.disconnect()

        except Exception as e:
            _flush_csv()
            print(f"[MQTT Subscriber] ❌ Error: {e}")
            self.client.disconnect()


# ──────────────────────────────────────────────────────────────
#  STANDALONE RUN
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    subscriber = SensorSubscriber()
    subscriber.start()
