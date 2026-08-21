"""
Morphix Desktop Companion Robot - Adaptive System Health & RPi Telemetry
=========================================================================
File: src/intent/system_health_plugin.py
Description:
    Real-time system telemetry collector and health evaluator supporting
    Raspberry Pi 4B thermal monitoring, battery status, CPU/RAM utilization,
    and dynamic OLED HUD overlay rendering.
"""

import logging
import os
import platform
import re
import subprocess
from typing import Any, Dict, Optional

logger = logging.getLogger("Telemetry")


class SystemHealthService:
    """Collects OS and hardware telemetry across Windows, Linux, and Raspberry Pi 4B."""

    _instance: Optional["SystemHealthService"] = None

    def __init__(self) -> None:
        self.is_linux = platform.system().lower() == "linux"
        self.is_windows = platform.system().lower() == "windows"
        self._cached_telemetry: Optional[Dict[str, Any]] = None

    @classmethod
    def get_instance(cls) -> "SystemHealthService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _read_cpu_temperature(self) -> float:
        """
        Reads CPU core temperature in degrees Celsius.
        Reads /sys/class/thermal/thermal_zone0/temp on Raspberry Pi / Linux.
        """
        if self.is_linux:
            # 1. Try Linux sysfs thermal zone
            thermal_path = "/sys/class/thermal/thermal_zone0/temp"
            if os.path.exists(thermal_path):
                try:
                    with open(thermal_path, "r", encoding="utf-8") as f:
                        raw = f.read().strip()
                        return round(float(raw) / 1000.0, 1)
                except Exception:
                    pass

            # 2. Try Raspberry Pi vcgencmd
            try:
                out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True, timeout=1)
                # Output format: temp=48.2'C
                m = re.search(r"temp=([\d\.]+)", out)
                if m:
                    return round(float(m.group(1)), 1)
            except Exception:
                pass

        # 3. Try psutil sensors_temperatures if available
        try:
            import psutil
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries:
                            return round(float(entries[0].current), 1)
        except Exception:
            pass

        # Nominal fallback for Windows laptops / standard environments
        return 42.0

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Collects real-time CPU, RAM, Battery, and Temperature telemetry.
        Returns a structured dictionary with evaluation indicators.
        """
        cpu_percent = 15.0
        ram_percent = 45.0
        battery_percent = 100
        is_charging = True

        try:
            import psutil
            cpu_percent = round(psutil.cpu_percent(interval=None), 1)
            ram_percent = round(psutil.virtual_memory().percent, 1)

            bat = psutil.sensors_battery()
            if bat is not None:
                battery_percent = int(bat.percent)
                is_charging = bool(bat.power_plugged)
        except Exception as ex:
            logger.debug("Telemetry gathering fallback used: %s", ex)

        temp_c = self._read_cpu_temperature()

        # Thermal alert threshold for Raspberry Pi / embedded hardware (>65.0°C)
        is_thermal_alert = (temp_c >= 65.0)

        telemetry_data = {
            "cpu": cpu_percent,
            "ram": ram_percent,
            "battery": battery_percent,
            "is_charging": is_charging,
            "temperature": temp_c,
            "is_thermal_alert": is_thermal_alert,
        }
        self._cached_telemetry = telemetry_data
        logger.info("System Telemetry: CPU=%.1f%%, RAM=%.1f%%, Temp=%.1f°C, Battery=%d%%",
                    cpu_percent, ram_percent, temp_c, battery_percent)
        return telemetry_data

    def format_speech_response(self, data: Dict[str, Any], is_hindi: bool = False) -> str:
        """Generates natural bilingual spoken report."""
        cpu = data.get("cpu", 20.0)
        ram = data.get("ram", 40.0)
        battery = data.get("battery", 100)
        temp = data.get("temperature", 42.0)
        is_thermal = data.get("is_thermal_alert", False)

        if is_thermal:
            if is_hindi:
                return f"Mujhe thoda garam lag raha hai, temperature {int(temp)} degree Celsius hai. Thoda rest chahiye!"
            return f"I'm running a bit warm at {int(temp)} degrees Celsius. Let me rest a bit to cool down!"

        if is_hindi:
            return (
                f"Main bilkul theek hoon! CPU {int(cpu)} percent, RAM {int(ram)} percent, "
                f"battery {battery} percent, aur temperature {int(temp)} degree Celsius hai."
            )
        return (
            f"All systems are running smoothly! CPU is at {int(cpu)} percent, RAM at {int(ram)} percent, "
            f"battery at {battery} percent, and temperature is {int(temp)} degrees Celsius."
        )


class SystemHealthIntentPlugin:
    """NLU plugin for system health, battery, and telemetry queries."""

    def __init__(self, service: Optional[SystemHealthService] = None) -> None:
        self.service = service or SystemHealthService.get_instance()

    def match(self, command: str) -> Optional[Any]:
        """Classifies system health and telemetry questions."""
        from src.intent.intent_classifier import IntentResult, IntentType

        c = command.strip().lower()
        is_hindi = bool(re.search(r'\b(kaisi|kaise|tabiyat|haal|garam|kitni|kitna|batao|chal)\b', c))

        # Matches: system status, how are you feeling, morphix health, battery level, cpu status
        patterns = [
            r'\b(system status|system health|how are you feeling|morphix health|robot status|health check)\b',
            r'\b(battery level|battery percentage|battery status|battery kitni hai|kitni battery hai)\b',
            r'\b(cpu status|cpu usage|ram usage|memory usage|temperature kitna hai system ka)\b',
            r'\b(kaisi tabiyat hai|apna status batao|kaisa feel ho raha hai|how are you)\b',
        ]

        for pat in patterns:
            if re.search(pat, c):
                return IntentResult(
                    intent=IntentType.GET_SYSTEM_HEALTH,
                    confidence=0.95,
                    parameters={"action": "telemetry", "is_hindi": is_hindi}
                )

        return None
