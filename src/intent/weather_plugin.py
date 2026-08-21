"""
Morphix Desktop Companion Robot - Weather Intent Plugin & Service
==================================================================
File: src/intent/weather_plugin.py
Description:
    Auto-detects user location via IP-API and fetches live weather via Open-Meteo API.
    Provides non-blocking execution, offline-friendly caching, fallback locations,
    and bilingual (English/Hindi) spoken responses for Edge TTS.
"""

import json
import logging
import re
import sys
import threading
import time
import urllib.request
from typing import Any, Dict, Optional
from src.intent.intent_classifier import IntentResult, IntentType

# Setup logger for Weather Plugin
logger = logging.getLogger("Morphix.Weather")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [Weather]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class WeatherService:
    """
    Fetches real-time weather using Open-Meteo (no API key required) and IP-API for auto-location.
    Includes caching for low-latency / non-blocking execution on Raspberry Pi 4B and Desktop.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._cached_location: Optional[Dict[str, Any]] = None
        self._loc_cache_time: float = 0.0
        self._cached_weather: Optional[Dict[str, Any]] = None
        self._weather_cache_time: float = 0.0
        self._fallback_city = "New Delhi"
        self._fallback_lat = 28.6139
        self._fallback_lon = 77.2090

    @classmethod
    def get_instance(cls) -> "WeatherService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = WeatherService()
        return cls._instance

    def get_location(self) -> Dict[str, Any]:
        """Auto-detects city and coordinates via IP lookup with 1-hour caching."""
        now = time.time()
        if self._cached_location and (now - self._loc_cache_time < 3600.0):
            return self._cached_location

        try:
            req = urllib.request.Request(
                "http://ip-api.com/json/?fields=status,country,city,lat,lon",
                headers={"User-Agent": "Morphix/1.0"}
            )
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("status") == "success":
                    city = data.get("city", self._fallback_city)
                    lat = float(data.get("lat", self._fallback_lat))
                    lon = float(data.get("lon", self._fallback_lon))
                    self._cached_location = {"city": city, "lat": lat, "lon": lon}
                    self._loc_cache_time = now
                    logger.info("Auto-detected location: %s (Lat: %.4f, Lon: %.4f)", city, lat, lon)
                    return self._cached_location
        except Exception as ex:
            logger.warning("Auto-location lookup failed: %s. Using default %s.", ex, self._fallback_city)

        self._cached_location = {
            "city": self._fallback_city,
            "lat": self._fallback_lat,
            "lon": self._fallback_lon
        }
        self._loc_cache_time = now
        return self._cached_location

    def get_weather(self, target_city: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetches current weather from Open-Meteo for auto-detected or target location.
        Caches results for 10 minutes to avoid redundant API hits.
        """
        now = time.time()
        if not target_city and self._cached_weather and (now - self._weather_cache_time < 600.0):
            return self._cached_weather

        loc = self.get_location()
        city = loc["city"]
        lat = loc["lat"]
        lon = loc["lon"]

        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m&timezone=auto"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "Morphix/1.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                curr = data.get("current", {})
                temp = float(curr.get("temperature_2m", 25.0))
                code = int(curr.get("weather_code", 0))
                wind = float(curr.get("wind_speed_10m", 10.0))
                humidity = int(curr.get("relative_humidity_2m", 60))

                condition, desc = self._map_wmo_code(code)
                res = {
                    "city": city,
                    "temperature": round(temp),
                    "temp_exact": temp,
                    "condition": condition,  # sunny, cloudy, rainy, stormy, snowy
                    "description": desc,
                    "wind_speed": round(wind, 1),
                    "humidity": humidity
                }
                if not target_city:
                    self._cached_weather = res
                    self._weather_cache_time = now
                logger.info("Fetched weather for %s: %d°C, %s", city, res["temperature"], desc)
                return res
        except Exception as ex:
            logger.error("Failed to fetch weather from Open-Meteo: %s", ex)
            return {
                "city": city,
                "temperature": 25,
                "temp_exact": 25.0,
                "condition": "sunny",
                "description": "pleasant",
                "wind_speed": 10.0,
                "humidity": 50
            }

    def _map_wmo_code(self, code: int) -> tuple[str, str]:
        """Maps WMO Weather interpretation code to icon category and description."""
        if code == 0:
            return "sunny", "clear sky"
        elif code in (1, 2):
            return "cloudy", "partly cloudy"
        elif code == 3:
            return "cloudy", "overcast"
        elif code in (45, 48):
            return "cloudy", "foggy"
        elif code in (51, 53, 55, 56, 57):
            return "rainy", "light drizzle"
        elif code in (61, 63, 65, 66, 67, 80, 81, 82):
            return "rainy", "rain showers"
        elif code in (71, 73, 75, 77, 85, 86):
            return "snowy", "snowfall"
        elif code in (95, 96, 99):
            return "stormy", "thunderstorm"
        return "sunny", "clear"

    def format_speech_response(self, weather_data: Dict[str, Any], is_hindi: bool = False) -> str:
        """Formats bilingual spoken response for Edge TTS."""
        city = weather_data["city"]
        temp = weather_data["temperature"]
        desc = weather_data["description"]
        wind = weather_data["wind_speed"]

        if is_hindi:
            return f"{city} mein abhi mausam {desc} hai, aur temperature {temp} degree Celsius hai."
        else:
            return f"It's currently {temp} degrees Celsius with {desc} in {city}."


class WeatherIntentPlugin:
    """Plugin handling weather and temperature query intents."""

    def __init__(self, service: Optional[WeatherService] = None) -> None:
        self.service = service or WeatherService.get_instance()

    def match(self, text: str) -> Optional[IntentResult]:
        lowercased = text.lower().strip()

        weather_patterns = [
            r"^(?:what(?:'s| is) (?:the )?)?weather(?: (?:like|report))?(?: in (?P<city>.+))?$",
            r"^(?:how(?:'s| is) (?:the )?)?weather(?: today)?$",
            r"^(?:show|tell) (?:me )?(?:the )?weather(?: report)?$",
            r"^what is the temperature(?: right now| today)?$",
            r"^current temperature$",
            r"^temperature (?:kitna|kya) hai$",
            r"^(?:aaj )?(?:ka )?mausam (?:kaisa|kya) hai$",
            r"^aaj weather kaisa hai$",
            r"^weather kaisa hai$",
            r"^temperature kitna chal raha hai$",
            r"^mausam ka haal(?: batao)?$"
        ]

        for pat in weather_patterns:
            m = re.match(pat, lowercased)
            if m:
                is_hindi = any(w in lowercased for w in ["mausam", "kaisa", "kitna", "kya", "aaj", "chal raha", "haal"])
                return IntentResult(
                    intent=IntentType.GET_WEATHER,
                    parameters={"is_hindi": is_hindi, "query": lowercased},
                    confidence=0.95,
                    raw_text=text
                )

        # Keyword checks
        if "weather" in lowercased or "mausam" in lowercased or ("temperature" in lowercased and any(k in lowercased for k in ["what", "tell", "kya", "kitna", "current", "show"])):
            is_hindi = any(w in lowercased for w in ["mausam", "kaisa", "kitna", "kya", "aaj", "chal raha", "haal"])
            return IntentResult(
                intent=IntentType.GET_WEATHER,
                parameters={"is_hindi": is_hindi, "query": lowercased},
                confidence=0.90,
                raw_text=text
            )

        return None
