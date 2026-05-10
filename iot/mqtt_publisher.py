# =============================================================
#  SmartGuard AI — MQTT Publisher
#  File: iot/mqtt_publisher.py
#
#  PURPOSE:
#  Connects to the MQTT broker and publishes sensor readings
#  every 5 seconds. Think of this as the "sender" side.
#
#  In a real IoT system, the ESP32 microcontroller runs this
#  logic in C/MicroPython. Here we simulate it in Python —
#  the broker receives identical packets either way.
#
#  MQTT Concept:
#  Publisher  →  [MQTT Broker]  →  Subscriber
#  (this file)   (HiveMQ cloud)    (mqtt_subscriber.py)
# =============================================================

import json
import time
import sys
import os
import paho.mqtt.client as mqtt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    MQTT_BROKER, MQTT_PORT, MQTT_TOPIC,
    SENSOR_INTERVAL_SECONDS, PATIENT_NAME
)
from iot.sensor_simulator import generate_reading, display_reading


# ──────────────────────────────────────────────────────────────
#  MQTT CALLBACKS
#  These functions are called automatically by the MQTT client
#  when specific events happen (connect, disconnect, publish).
# ──────────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, rc):
    """
    Called automatically when publisher connects to broker.
    rc = return code: 0 means success, anything else is an error.
    """
    if rc == 0:
        print(f"[MQTT Publisher] ✅ Connected to broker: {MQTT_BROKER}")
        print(f"[MQTT Publisher] 📡 Publishing to topic: {MQTT_TOPIC}")
        print(f"[MQTT Publisher] ⏱  Interval: every {SENSOR_INTERVAL_SECONDS}s\n")
    else:
        print(f"[MQTT Publisher] ❌ Connection failed. Return code: {rc}")


def on_disconnect(client, userdata, rc):
    """Called when publisher disconnects from broker."""
    if rc == 0:
        print("[MQTT Publisher] Disconnected cleanly.")
    else:
        print(f"[MQTT Publisher] Unexpected disconnect. Code: {rc}")


def on_publish(client, userdata, mid):
    """
    Called when a message is successfully sent to the broker.
    'mid' is the message ID — confirms delivery.
    """
    print(f"[MQTT Publisher] 📤 Message delivered (ID: {mid})")


# ──────────────────────────────────────────────────────────────
#  PUBLISHER CLASS
#  Wraps the MQTT client with our project-specific logic.
# ──────────────────────────────────────────────────────────────

class SensorPublisher:
    """
    Connects to MQTT broker and continuously publishes
    sensor readings from the simulator.
    """

    def __init__(self):
        # Create MQTT client instance
        self.client = mqtt.Client(client_id="smartguard_publisher")

        # Attach our callback functions
        self.client.on_connect    = on_connect
        self.client.on_disconnect = on_disconnect
        self.client.on_publish    = on_publish

        self.is_connected = False

    def connect(self):
        """Connect to the MQTT broker."""
        print(f"[MQTT Publisher] Connecting to {MQTT_BROKER}:{MQTT_PORT}...")
        try:
            self.client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            # Start background thread that handles MQTT network traffic
            self.client.loop_start()
            time.sleep(1)  # Give it a moment to establish connection
            self.is_connected = True
        except Exception as e:
            print(f"[MQTT Publisher] ❌ Could not connect: {e}")
            self.is_connected = False

    def publish_reading(self, reading):
        """
        Publish one sensor reading to the MQTT topic.
        The reading dict is serialised to JSON string before sending.
        JSON is the universal format for IoT data exchange.
        """
        payload = json.dumps(reading)  # dict → JSON string

        result = self.client.publish(
            topic   = MQTT_TOPIC,
            payload = payload,
            qos     = 1,        # QoS 1 = at least once delivery guarantee
            retain  = False     # Don't retain last message on broker
        )
        return result

    def run(self, force_anomaly=False, max_readings=None):
        """
        Main loop — generates and publishes readings continuously.

        Args:
            force_anomaly (bool): Force every reading to be anomalous.
                                  Useful for demo/testing.
            max_readings (int):   Stop after N readings. None = run forever.
        """
        if not self.is_connected:
            print("[MQTT Publisher] Not connected. Call connect() first.")
            return

        count = 0
        print(f"[MQTT Publisher] 🚀 Starting stream for patient: {PATIENT_NAME}")
        print("[MQTT Publisher] Press Ctrl+C to stop.\n")
        print("-" * 55)

        try:
            while True:
                # Generate one reading from the simulator
                reading = generate_reading(force_anomaly=force_anomaly)

                # Print to terminal so we can watch it live
                display_reading(reading)

                # Publish to MQTT broker
                self.publish_reading(reading)

                count += 1

                # Stop if we've hit the max (used in testing)
                if max_readings and count >= max_readings:
                    print(f"\n[MQTT Publisher] Reached {max_readings} readings. Stopping.")
                    break

                # Wait before next reading
                time.sleep(SENSOR_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print(f"\n[MQTT Publisher] Stopped by user. Total readings sent: {count}")

        finally:
            self.disconnect()

    def disconnect(self):
        """Cleanly disconnect from broker."""
        self.client.loop_stop()
        self.client.disconnect()


# ──────────────────────────────────────────────────────────────
#  STANDALONE RUN
#  Command: python iot/mqtt_publisher.py
#  This starts the publisher and streams readings to the broker.
#  Keep this running and open mqtt_subscriber.py in a new terminal.
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  SmartGuard AI — MQTT Publisher")
    print("=" * 55)

    publisher = SensorPublisher()
    publisher.connect()

    # Check if demo mode requested
    # Run: python iot/mqtt_publisher.py demo
    # This forces all anomaly readings for demo purposes
    force_anomaly = len(sys.argv) > 1 and sys.argv[1] == "demo"

    if force_anomaly:
        print("[MQTT Publisher] 🎭 DEMO MODE — All readings will be anomalies\n")

    publisher.run(force_anomaly=force_anomaly)