"""
Morphix Desktop Companion Robot - Indian Festival & Dynamic Theming Plugin
==========================================================================
File: src/intent/festival_plugin.py
Description:
    Automatic Indian & Maharashtra festival detection using python-holidays
    and dynamic regional calendar rules, with procedural OLED theme mapping
    and bilingual vocal greetings on the first wake of the day.
"""

from dataclasses import dataclass
from datetime import date, datetime
import enum
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Festival")


class FestivalTheme(str, enum.Enum):
    """Dynamic procedural festival overlay themes for Morphix OLED display."""
    NONE = "NONE"
    MODAK = "MODAK"                   # Ganesh Chaturthi: Modak dumpling + Tilak
    DIWALI_DIYA = "DIWALI_DIYA"       # Diwali & Dussehra: Twin flickering clay diyas + sparklers
    HOLI_SPLASH = "HOLI_SPLASH"       # Holi: Colorful gulal powder burst dots
    SANTA_HAT = "SANTA_HAT"           # Christmas (25 Dec): Santa cap + snowflake particles
    TRICOLOR = "TRICOLOR"             # Republic Day (26 Jan) & Independence Day (15 Aug): Tiranga ribbon badge
    KITE = "KITE"                     # Makar Sankranti (14 Jan): Flying diamond kite
    NEW_YEAR = "NEW_YEAR"             # New Year (1 Jan): Party cone cap + confetti
    BIRTHDAY = "BIRTHDAY"             # Birthday: Lit cupcake + party sparkles
    GUDI_PADWA = "GUDI_PADWA"         # Gudi Padwa: Gudi flag with copper kalash


@dataclass
class FestivalInfo:
    name: str
    theme: FestivalTheme
    greeting_en: str
    greeting_hi: str
    badge_text: str
    is_today: bool = True


class FestivalService:
    """Detects Indian & Maharashtra festivals and coordinates seasonal themes."""

    _instance: Optional["FestivalService"] = None

    def __init__(self, greeting_cache_file: str = "temp/festival_greeting_state.json") -> None:
        self.greeting_cache_file = greeting_cache_file
        self._user_birthday = os.getenv("USER_BIRTHDAY", "06-20")  # MM-DD default (June 20)

    @classmethod
    def get_instance(cls) -> "FestivalService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_todays_festival(self, target_date: Optional[date] = None) -> Optional[FestivalInfo]:
        """
        Determines if the target date (default today) corresponds to any Indian festival,
        national holiday, or user birthday.
        """
        d = target_date or date.today()
        month_day = f"{d.month:02d}-{d.day:02d}"

        # 1. Check User Birthday
        if self._user_birthday.endswith(month_day):
            return FestivalInfo(
                name="Birthday",
                theme=FestivalTheme.BIRTHDAY,
                greeting_en="Happy Birthday! Wishing you a wonderful year ahead!",
                greeting_hi="Janamdin Mubarak ho! Aaj party ka din hai!",
                badge_text="Happy Birthday! 🎂"
            )

        # 2. Fixed Annual Dates
        if month_day == "01-01":
            return FestivalInfo(
                name="New Year",
                theme=FestivalTheme.NEW_YEAR,
                greeting_en="Happy New Year! Welcome to a fresh new year!",
                greeting_hi="Naya Saal Mubarak! Yeh saal aapke liye bohot khushiyan laye!",
                badge_text="Happy New Year! 🎉"
            )
        elif month_day == "01-14":
            return FestivalInfo(
                name="Makar Sankranti",
                theme=FestivalTheme.KITE,
                greeting_en="Happy Makar Sankranti! May your life soar high like kites!",
                greeting_hi="Makar Sankranti ki shubhkamnayein! Til-gul ghya, god-god bola!",
                badge_text="Happy Sankranti! 🪁"
            )
        elif month_day == "01-26":
            return FestivalInfo(
                name="Republic Day",
                theme=FestivalTheme.TRICOLOR,
                greeting_en="Happy Republic Day! Proud to celebrate our nation!",
                greeting_hi="Gantantra Diwas ki hardik shubhkamnayein! Vande Mataram!",
                badge_text="Happy Republic Day! 🇮🇳"
            )
        elif month_day == "05-01":
            return FestivalInfo(
                name="Maharashtra Day",
                theme=FestivalTheme.TRICOLOR,
                greeting_en="Happy Maharashtra Day!",
                greeting_hi="Maharashtra Dinachya hardik shubheccha! Jai Maharashtra!",
                badge_text="Jai Maharashtra! 🇮🇳"
            )
        elif month_day == "08-15":
            return FestivalInfo(
                name="Independence Day",
                theme=FestivalTheme.TRICOLOR,
                greeting_en="Happy Independence Day! Jai Hind!",
                greeting_hi="Swatantrata Diwas ki shubhkamnayein! Jai Hind!",
                badge_text="Happy Independence Day! 🇮🇳"
            )
        elif month_day == "12-25":
            return FestivalInfo(
                name="Christmas",
                theme=FestivalTheme.SANTA_HAT,
                greeting_en="Merry Christmas! Wishing you joy and cheer!",
                greeting_hi="Merry Christmas! Aaj khushiyan baantne ka din hai!",
                badge_text="Merry Christmas! 🎄"
            )

        # 3. Dynamic Lunar Holidays using python-holidays library
        try:
            import holidays
            mh_holidays = holidays.India(subdiv="MH", years=d.year)
            h_name = mh_holidays.get(d)
            if h_name:
                h_lower = h_name.lower()
                if "ganesh" in h_lower:
                    return FestivalInfo(
                        name="Ganesh Chaturthi",
                        theme=FestivalTheme.MODAK,
                        greeting_en="Happy Ganesh Chaturthi! May Lord Ganesha bless you with wisdom and joy!",
                        greeting_hi="Ganpati Bappa Morya! Aaj modak khane ka din hai!",
                        badge_text="Ganpati Bappa Morya! 🥟"
                    )
                elif "diwali" in h_lower or "deepavali" in h_lower or "lakshmi" in h_lower:
                    return FestivalInfo(
                        name="Diwali",
                        theme=FestivalTheme.DIWALI_DIYA,
                        greeting_en="Happy Diwali! May your home be filled with light and prosperity!",
                        greeting_hi="Shubh Deepavali! Aapka din roshni aur mithai se bhara rahe!",
                        badge_text="Happy Diwali! 🪔"
                    )
                elif "dussehra" in h_lower or "vijayadashami" in h_lower:
                    return FestivalInfo(
                        name="Dussehra",
                        theme=FestivalTheme.DIWALI_DIYA,
                        greeting_en="Happy Dussehra! May good always triumph over evil!",
                        greeting_hi="Vijayadashami ki hardik shubhkamnayein!",
                        badge_text="Happy Dussehra! 🏹"
                    )
                elif "holi" in h_lower:
                    return FestivalInfo(
                        name="Holi",
                        theme=FestivalTheme.HOLI_SPLASH,
                        greeting_en="Happy Holi! May your day be filled with vibrant colors of happiness!",
                        greeting_hi="Holi ki dher saari shubhkamnayein! Bura na mano Holi hai!",
                        badge_text="Happy Holi! 🎨"
                    )
                elif "gudi padwa" in h_lower:
                    return FestivalInfo(
                        name="Gudi Padwa",
                        theme=FestivalTheme.GUDI_PADWA,
                        greeting_en="Happy Gudi Padwa! Wishing you prosperity in this new year!",
                        greeting_hi="Gudi Padwyachya hardik shubheccha!",
                        badge_text="Gudi Padwa! 🚩"
                    )
                elif "janmashtami" in h_lower:
                    return FestivalInfo(
                        name="Janmashtami",
                        theme=FestivalTheme.MODAK,
                        greeting_en="Happy Janmashtami! May Lord Krishna bring joy to your life!",
                        greeting_hi="Krishna Janmashtami ki hardik shubhkamnayein! Radhe Radhe!",
                        badge_text="Happy Janmashtami! 🦚"
                    )
                else:
                    return FestivalInfo(
                        name=h_name,
                        theme=FestivalTheme.DIWALI_DIYA,
                        greeting_en=f"Happy {h_name}!",
                        greeting_hi=f"{h_name} ki hardik shubhkamnayein!",
                        badge_text=f"Happy {h_name}!"
                    )
        except Exception as ex:
            logger.debug("Holiday lookup exception: %s", ex)

        return None

    def should_play_first_wake_greeting(self, today_fest: FestivalInfo) -> bool:
        """
        Checks if the festive greeting has already been delivered today.
        Returns True if this is the first wake of the day on a festival.
        """
        today_str = date.today().isoformat()
        try:
            if os.path.exists(self.greeting_cache_file):
                with open(self.greeting_cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("last_greeting_date") == today_str and data.get("festival_name") == today_fest.name:
                        return False
        except Exception:
            pass

        # Mark greeting as delivered today
        try:
            os.makedirs(os.path.dirname(self.greeting_cache_file), exist_ok=True)
            with open(self.greeting_cache_file, "w", encoding="utf-8") as f:
                json.dump({
                    "last_greeting_date": today_str,
                    "festival_name": today_fest.name,
                    "timestamp": time.time()
                }, f)
        except Exception as ex:
            logger.debug("Could not write greeting cache: %s", ex)

        return True


class FestivalIntentPlugin:
    """NLU plugin for detecting festival queries and wishes."""

    def __init__(self, service: Optional[FestivalService] = None) -> None:
        self.service = service or FestivalService.get_instance()

    def match(self, command: str) -> Optional[Any]:
        """
        Classifies whether user is inquiring about festivals/holidays or offering a festival wish.
        """
        from src.intent.intent_classifier import IntentResult, IntentType

        c = command.strip().lower()
        is_hindi = bool(re.search(r'\b(aaj|kaun|sa|hai|tyohar|kya|chhutti|shubh|mubarak)\b', c))

        festival_query_patterns = [
            r'\b(what festival is today|which festival is today|is today a festival|is today a holiday|today festival|any holiday today)\b',
            r'\b(aaj kaun sa festival hai|aaj kya tyohar hai|aaj chhutti hai kya|aaj koi festival hai|festival batao)\b',
            r'\b(happy diwali|happy holi|ganpati bappa morya|merry christmas|happy new year|happy independence day|happy republic day|happy ganesh chaturthi)\b'
        ]

        for p in festival_query_patterns:
            if re.search(p, c):
                # Extract any specific wish if present
                fest_today = self.service.get_todays_festival()
                return IntentResult(
                    intent=IntentType.GET_FESTIVAL,
                    confidence=0.95,
                    parameters={
                        "is_hindi": is_hindi,
                        "today_festival": fest_today.name if fest_today else None
                    }
                )

        return None
