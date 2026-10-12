"""
Morphix Desktop Companion Robot - Weather Intent Plugin & Service
==================================================================
File: src/intent/weather_plugin.py
Description:
    Auto-detects user location via IP-API and fetches live weather via Open-Meteo API.
    Supports geocoding multi-city voice queries, non-blocking execution, offline-friendly caching,
    fallback locations, and bilingual (English/Hindi) spoken responses for Edge TTS.
"""

import json
import logging
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple
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


def extract_city(text: str) -> Optional[str]:
    """
    Parses target city name from English and Hinglish voice queries.
    Strips noise words ("today", "now", "kaisa", "hai", "batao", etc.).
    Returns clean city name in Title Case (e.g. "Delhi", "Nagpur", "Mumbai") or None.
    """
    if not text or not isinstance(text, str):
        return None

    cleaned = text.lower().strip()

    # Noise words to strip from candidate city string
    noise_words = {
        "today", "now", "right now", "kaisa", "hai", "batao", "bataiye", "batao na",
        "ka", "ki", "ke", "ko", "par", "mein", "me", "report", "like", "current",
        "temperature", "weather", "mausam", "temp", "please", "tell", "show",
        "what", "is", "the", "how", "for", "in", "of", "aaj", "chal", "raha", "haal",
        "right", "outside", "details"
    }

    patterns = [
        # "weather in Delhi", "temperature in Delhi", "weather for Delhi", "weather of Mumbai"
        r"(?:weather|temperature|temp|mausam)\s+(?:in|for|of|at)\s+([a-zA-Z\s]+)",
        # "Delhi weather", "Nagpur temperature", "Mumbai mausam"
        r"([a-zA-Z\s]+)\s+(?:weather|temperature|temp|mausam)",
        # "Delhi ka weather", "Nagpur ki temperature", "Mumbai me mausam"
        r"([a-zA-Z\s]+)\s+(?:ka|ki|ke|me|mein)\s+(?:weather|temperature|temp|mausam)",
        # "in Delhi", "for Bangalore", "of Mumbai"
        r"(?:in|for|of|at)\s+([a-zA-Z\s]+)$",
    ]

    for pat in patterns:
        m = re.search(pat, cleaned)
        if m:
            candidate = m.group(1).strip()
            tokens = [w for w in candidate.split() if w not in noise_words]
            if tokens:
                city = " ".join(tokens).strip()
                if len(city) >= 2 and re.match(r"^[a-zA-Z\s]+$", city):
                    return city.title()

    return None


class WeatherService:
    """
    Fetches real-time weather using Open-Meteo (no API key required) and IP-API for auto-location.
    Includes geocoding for target cities and caching for low-latency execution.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._cached_location: Optional[Dict[str, Any]] = None
        self._loc_cache_time: float = 0.0
        self._cached_weather: Optional[Dict[str, Any]] = None
        self._weather_cache_time: float = 0.0
        self._fallback_lat = 28.6139
        self._fallback_lon = 77.2090

    @classmethod
    def get_instance(cls) -> "WeatherService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = WeatherService()
        return cls._instance

    def _get_fallback_city(self) -> str:
        """Dynamically retrieves default city from env or profile config."""
        city_env = os.getenv("DEFAULT_CITY") or os.getenv("WEATHER_DEFAULT_CITY")
        if city_env and city_env.strip():
            return city_env.strip()
        try:
            from src.config.profile_manager import load_user_profile
            profile = load_user_profile()
            cfg_city = profile.get("default_city") or profile.get("city")
            if cfg_city and isinstance(cfg_city, str) and cfg_city.strip():
                return cfg_city.strip()
        except Exception:
            pass
        return "New Delhi"

    def get_location(self) -> Dict[str, Any]:
        """Auto-detects city and coordinates via IP lookup with 1-hour caching."""
        now = time.time()
        if self._cached_location and (now - self._loc_cache_time < 3600.0):
            return self._cached_location

        fallback_city = self._get_fallback_city()
        try:
            req = urllib.request.Request(
                "http://ip-api.com/json/?fields=status,country,city,lat,lon",
                headers={"User-Agent": "Morphix/1.0"}
            )
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("status") == "success":
                    city = data.get("city", fallback_city)
                    lat = float(data.get("lat", self._fallback_lat))
                    lon = float(data.get("lon", self._fallback_lon))
                    self._cached_location = {"city": city, "lat": lat, "lon": lon}
                    self._loc_cache_time = now
                    logger.info("Auto-detected location: %s (Lat: %.4f, Lon: %.4f)", city, lat, lon)
                    return self._cached_location
        except Exception as ex:
            logger.warning("Auto-location lookup failed: %s. Using default %s.", ex, fallback_city)

        self._cached_location = {
            "city": fallback_city,
            "lat": self._fallback_lat,
            "lon": self._fallback_lon
        }
        self._loc_cache_time = now
        return self._cached_location

    def get_weather(self, target_city: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetches current weather from Open-Meteo for auto-detected or target location.
        Caches default location weather for 10 minutes.
        """
        now = time.time()
        clean_target = target_city.strip() if (target_city and isinstance(target_city, str) and target_city.strip()) else None

        if not clean_target and self._cached_weather and (now - self._weather_cache_time < 600.0):
            return self._cached_weather

        city_name = None
        lat = None
        lon = None

        if clean_target:
            try:
                encoded_city = urllib.parse.quote(clean_target)
                geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_city}&count=1&language=en&format=json"
                req = urllib.request.Request(geo_url, headers={"User-Agent": "Morphix/1.0"})
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    geo_data = json.loads(resp.read().decode("utf-8"))
                    results = geo_data.get("results", [])
                    if results and len(results) > 0:
                        top_res = results[0]
                        city_name = top_res.get("name", clean_target).title()
                        lat = float(top_res.get("latitude"))
                        lon = float(top_res.get("longitude"))
                        logger.info("Geocoded city '%s' -> %s (Lat: %.4f, Lon: %.4f)", clean_target, city_name, lat, lon)
            except Exception as ex:
                logger.warning("Geocoding failed for city '%s': %s", clean_target, ex)

        if not city_name or lat is None or lon is None:
            loc = self.get_location()
            city_name = loc["city"]
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
                    "city": city_name,
                    "temperature": round(temp),
                    "temp_exact": temp,
                    "condition": condition,
                    "description": desc,
                    "wind_speed": round(wind, 1),
                    "humidity": humidity
                }
                if not clean_target:
                    self._cached_weather = res
                    self._weather_cache_time = now
                logger.info("Fetched weather for %s: %d°C, %s", city_name, res["temperature"], desc)
                return res
        except Exception as ex:
            logger.error("Failed to fetch weather from Open-Meteo: %s", ex)
            return {
                "city": city_name or "New Delhi",
                "temperature": 25,
                "temp_exact": 25.0,
                "condition": "sunny",
                "description": "pleasant",
                "wind_speed": 10.0,
                "humidity": 50
            }

    def _map_wmo_code(self, code: int) -> Tuple[str, str]:
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
        city = extract_city(text)

        is_weather_query = (
            "weather" in lowercased
            or "mausam" in lowercased
            or ("temperature" in lowercased and any(k in lowercased for k in ["what", "tell", "kya", "kitna", "current", "show", "in", "for", "of", "how"]))
            or "temp" in lowercased
        )

        if is_weather_query:
            is_hindi = any(w in lowercased for w in ["mausam", "kaisa", "kitna", "kya", "aaj", "chal raha", "haal", "batao"])
            return IntentResult(
                intent=IntentType.GET_WEATHER,
                parameters={"is_hindi": is_hindi, "city": city, "query": lowercased},
                confidence=0.95 if city else 0.90,
                raw_text=text
            )

        return None
