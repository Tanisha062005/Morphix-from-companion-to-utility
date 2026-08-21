"""
Morphix Desktop Companion Robot - System Health & Telemetry Test Suite
======================================================================
File: tests/test_system_health_plugin.py
Description:
    Unit tests for SystemHealthService and SystemHealthIntentPlugin matching.
"""

import unittest
from src.intent.intent_classifier import IntentClassifier, IntentType
from src.intent.system_health_plugin import SystemHealthIntentPlugin, SystemHealthService


class TestSystemHealthPlugin(unittest.TestCase):
    """Verifies telemetry collection, thermal alert evaluation, and intent matching."""

    def setUp(self) -> None:
        self.service = SystemHealthService()
        self.plugin = SystemHealthIntentPlugin(service=self.service)

    def test_telemetry_gathering(self) -> None:
        data = self.service.get_telemetry()
        self.assertIn("cpu", data)
        self.assertIn("ram", data)
        self.assertIn("battery", data)
        self.assertIn("temperature", data)
        self.assertIn("is_thermal_alert", data)

    def test_thermal_alert_speech_formatting(self) -> None:
        alert_data = {
            "cpu": 80.0,
            "ram": 75.0,
            "battery": 40,
            "temperature": 72.0,
            "is_thermal_alert": True,
        }
        res_hi = self.service.format_speech_response(alert_data, is_hindi=True)
        self.assertIn("garam", res_hi)
        self.assertIn("72", res_hi)

        res_en = self.service.format_speech_response(alert_data, is_hindi=False)
        self.assertIn("warm", res_en)
        self.assertIn("72", res_en)

    def test_nominal_speech_formatting(self) -> None:
        nominal_data = {
            "cpu": 25.0,
            "ram": 45.0,
            "battery": 90,
            "temperature": 42.0,
            "is_thermal_alert": False,
        }
        res_hi = self.service.format_speech_response(nominal_data, is_hindi=True)
        self.assertIn("CPU", res_hi)
        self.assertIn("25", res_hi)

    def test_intent_matching(self) -> None:
        queries = [
            "system status",
            "how are you feeling",
            "morphix health",
            "battery level",
            "battery kitni hai",
            "cpu status",
            "health check"
        ]
        for q in queries:
            res = self.plugin.match(q)
            self.assertIsNotNone(res, f"Failed to match query: {q}")
            self.assertEqual(res.intent, IntentType.GET_SYSTEM_HEALTH)

    def test_classifier_integration(self) -> None:
        classifier = IntentClassifier()
        res = classifier.classify("what is your battery level")
        self.assertEqual(res.intent, IntentType.GET_SYSTEM_HEALTH)


if __name__ == "__main__":
    unittest.main()
