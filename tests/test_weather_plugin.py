"""
Morphix Desktop Companion Robot - Weather Plugin Test Suite
============================================================
File: tests/test_weather_plugin.py
Description:
    Unit tests for WeatherService, WeatherIntentPlugin, extract_city, and IntentClassifier integration.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.intent.intent_classifier import IntentClassifier, IntentType
from src.intent.weather_plugin import WeatherIntentPlugin, WeatherService, extract_city


class TestWeatherPlugin(unittest.TestCase):
    """Verifies location detection, city extraction, weather fetching, WMO code mapping, and intent matching."""

    def setUp(self) -> None:
        self.weather_service = WeatherService()
        self.plugin = WeatherIntentPlugin(service=self.weather_service)

    def test_wmo_code_mapping(self) -> None:
        self.assertEqual(self.weather_service._map_wmo_code(0), ("sunny", "clear sky"))
        self.assertEqual(self.weather_service._map_wmo_code(2), ("cloudy", "partly cloudy"))
        self.assertEqual(self.weather_service._map_wmo_code(3), ("cloudy", "overcast"))
        self.assertEqual(self.weather_service._map_wmo_code(61), ("rainy", "rain showers"))
        self.assertEqual(self.weather_service._map_wmo_code(71), ("snowy", "snowfall"))
        self.assertEqual(self.weather_service._map_wmo_code(95), ("stormy", "thunderstorm"))

    def test_speech_formatting_english(self) -> None:
        data = {
            "city": "Nashik",
            "temperature": 24,
            "condition": "cloudy",
            "description": "partly cloudy",
            "wind_speed": 12.5,
            "humidity": 70
        }
        res = self.weather_service.format_speech_response(data, is_hindi=False)
        self.assertIn("24 degrees Celsius", res)
        self.assertIn("Nashik", res)
        self.assertIn("partly cloudy", res)

    def test_speech_formatting_hindi(self) -> None:
        data = {
            "city": "Delhi",
            "temperature": 28,
            "condition": "sunny",
            "description": "clear sky",
            "wind_speed": 10.0,
            "humidity": 45
        }
        res = self.weather_service.format_speech_response(data, is_hindi=True)
        self.assertIn("Delhi", res)
        self.assertIn("28 degree Celsius", res)

    def test_city_extraction(self) -> None:
        city_queries = [
            ("What is the weather in Delhi?", "Delhi"),
            ("Nagpur ka weather kaisa hai?", "Nagpur"),
            ("Weather of Mumbai", "Mumbai"),
            ("Temperature for Bangalore", "Bangalore"),
            ("How is the weather?", None),
        ]
        for q, expected in city_queries:
            with self.subTest(q=q):
                city = extract_city(q)
                self.assertEqual(city, expected)
                res = self.plugin.match(q)
                self.assertIsNotNone(res)
                self.assertEqual(res.parameters.get("city"), expected)

    def test_intent_matching_english(self) -> None:
        queries = [
            "what's the weather",
            "how's the weather today",
            "show me the weather",
            "tell me the weather",
            "what is the temperature",
            "current temperature"
        ]
        for q in queries:
            res = self.plugin.match(q)
            self.assertIsNotNone(res, f"Failed to match: {q}")
            self.assertEqual(res.intent, IntentType.GET_WEATHER)
            self.assertFalse(res.parameters.get("is_hindi", False))

    def test_intent_matching_hindi(self) -> None:
        queries = [
            "aaj ka mausam kaisa hai",
            "temperature kitna hai",
            "weather kaisa hai",
            "mausam ka haal batao"
        ]
        for q in queries:
            res = self.plugin.match(q)
            self.assertIsNotNone(res, f"Failed to match: {q}")
            self.assertEqual(res.intent, IntentType.GET_WEATHER)
            self.assertTrue(res.parameters.get("is_hindi", False))

    def test_intent_classifier_integration(self) -> None:
        classifier = IntentClassifier()
        res = classifier.classify("what is the weather today")
        self.assertEqual(res.intent, IntentType.GET_WEATHER)

        res_hi = classifier.classify("aaj ka mausam kaisa hai")
        self.assertEqual(res_hi.intent, IntentType.GET_WEATHER)


if __name__ == "__main__":
    unittest.main()
