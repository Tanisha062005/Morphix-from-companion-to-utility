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
from typing import Any, Dict, List, Optional, Tuple

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

    @classmethod
    def get_instance(cls) -> "FestivalService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _parse_birthday_month_day(bday_str: str) -> Optional[Tuple[int, int]]:
        """
        Parses a birthday string into (month, day) tuple across multiple date formats:
        - MM-DD or MM/DD (e.g. "06-20")
        - DD-MM or DD/MM (e.g. "20-06")
        - YYYY-MM-DD or YYYY/MM/DD (e.g. "2005-06-20")
        - DD-MM-YYYY or MM-DD-YYYY (e.g. "20-06-2005")
        """
        if not bday_str or not isinstance(bday_str, str):
            return None
        s = bday_str.strip()

        # ISO YYYY-MM-DD or YYYY/MM/DD
        m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", s)
        if m:
            try:
                month, day = int(m.group(2)), int(m.group(3))
                if 1 <= month <= 12 and 1 <= day <= 31:
                    return month, day
            except ValueError:
                pass

        parts = re.split(r"[-/]", s)
        if len(parts) == 2:
            try:
                p1, p2 = int(parts[0]), int(parts[1])
                # If p1 > 12, it must be DD-MM
                if p1 > 12 and 1 <= p2 <= 12:
                    return p2, p1
                # Standard MM-DD
                if 1 <= p1 <= 12 and 1 <= p2 <= 31:
                    return p1, p2
            except ValueError:
                pass
        elif len(parts) == 3:
            try:
                p1, p2, p3 = int(parts[0]), int(parts[1]), int(parts[2])
                if p3 > 31:  # Year is last element (DD-MM-YYYY or MM-DD-YYYY)
                    if p1 > 12 and 1 <= p2 <= 12:  # DD-MM-YYYY
                        return p2, p1
                    elif 1 <= p1 <= 12 and 1 <= p2 <= 31:  # MM-DD-YYYY
                        return p1, p2
            except ValueError:
                pass

        return None

    def _get_user_birthday(self) -> Optional[Tuple[int, int]]:
        """Retrieves user birthday from profile config or env var and parses to (month, day)."""
        bday_raw = os.getenv("USER_BIRTHDAY")
        if not bday_raw:
            try:
                from src.config.profile_manager import load_user_profile
                profile = load_user_profile()
                bday_raw = profile.get("birthday")
            except Exception:
                pass
        if not bday_raw:
            bday_raw = "06-20"  # Default June 20

        return self._parse_birthday_month_day(str(bday_raw))

    def get_todays_festival(self, target_date: Optional[date] = None) -> Optional[FestivalInfo]:
        """
        Determines if the target date (default today) corresponds to any Indian festival,
        national holiday, or user birthday.
        """
        d = target_date or date.today()
        month_day = f"{d.month:02d}-{d.day:02d}"

        # 1. Check User Birthday (robust format parsing)
        user_bday = self._get_user_birthday()
        if user_bday and user_bday == (d.month, d.day):
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

    def handle(self, intent_result: Any, display: Optional[Any] = None) -> str:
        """
        Executes the matching festival response and activates designated OLED overlays.
        """
        active_disp = display
        if active_disp is None:
            try:
                from src.display.eye_animator import get_active_display
                active_disp = get_active_display()
            except Exception:
                active_disp = None

        params = getattr(intent_result, "parameters", {}) if hasattr(intent_result, "parameters") else {}
        is_hindi = params.get("is_hindi", False)
        overlay_tag = params.get("overlay_tag")
        theme_val = params.get("theme")
        greeting_hi = params.get("greeting_hi")
        greeting_en = params.get("greeting_en")

        # 1. If specific festival was matched from utterance
        if overlay_tag and (greeting_hi or greeting_en):
            if active_disp and hasattr(active_disp, "is_alive") and active_disp.is_alive():
                try:
                    if hasattr(active_disp, "set_overlay"):
                        active_disp.set_overlay(overlay_tag, timeout=10.0)
                    elif hasattr(active_disp, "set_festival_theme"):
                        active_disp.set_festival_theme(theme_val or overlay_tag)
                except Exception as ex:
                    logger.warning("Failed setting festive overlay: %s", ex)
            return greeting_hi if is_hindi else greeting_en

        # 2. General date query: look up today's holiday / festival
        fest_today = self.service.get_todays_festival()
        if fest_today:
            if active_disp and hasattr(active_disp, "is_alive") and active_disp.is_alive():
                try:
                    if hasattr(active_disp, "set_festival_theme"):
                        active_disp.set_festival_theme(fest_today.theme, fest_today.badge_text)
                except Exception as ex:
                    logger.warning("Failed setting today's festival theme: %s", ex)
            return fest_today.greeting_hi if is_hindi else fest_today.greeting_en

        return (
            "Aaj koi vishesh tyohar nahi hai, lekin har din ek naya utsav hai!"
            if is_hindi
            else "There are no major festivals today, but every day is a celebration!"
        )

    def match(self, command: str) -> Optional[Any]:
        """
        Classifies whether user is inquiring about festivals/holidays or offering a festival wish.
        """
        from src.intent.intent_classifier import IntentResult, IntentType

        if not command or not isinstance(command, str):
            return None

        c = command.strip().lower()
        is_hindi = bool(re.search(r'\b(aaj|kaun|sa|hai|tyohar|kya|chhutti|shubh|shubhkamnayein|mubarak|bappa|morya|til-gul)\b', c))

        # 1. Check for specific festival keywords & wishes
        # A. Ganesh Chaturthi
        if re.search(r'\b(ganesh|ganpati|bappa|morya|chaturthi|modak)\b', c):
            return IntentResult(
                intent=IntentType.GET_FESTIVAL,
                confidence=0.98,
                parameters={
                    "is_hindi": is_hindi,
                    "specific_festival": "Ganesh Chaturthi",
                    "overlay_tag": "FESTIVAL_GANESH",
                    "theme": FestivalTheme.MODAK.value,
                    "greeting_en": "Ganpati Bappa Morya! Happy Ganesh Chaturthi! May Lord Ganesha bless you with wisdom, joy, and delicious modaks!",
                    "greeting_hi": "Ganpati Bappa Morya! Ganesh Chaturthi ki hardik shubhkamnayein! Aaj modak khane aur khushiyan manane ka din hai!"
                },
                raw_text=command
            )

        # B. Holi
        if re.search(r'\b(holi|gulal|rang)\b', c):
            return IntentResult(
                intent=IntentType.GET_FESTIVAL,
                confidence=0.98,
                parameters={
                    "is_hindi": is_hindi,
                    "specific_festival": "Holi",
                    "overlay_tag": "FESTIVAL_HOLI",
                    "theme": FestivalTheme.HOLI_SPLASH.value,
                    "greeting_en": "Happy Holi! May your life be filled with vibrant colors of happiness and joy!",
                    "greeting_hi": "Holi ki dher saari shubhkamnayein! Bura na mano Holi hai!"
                },
                raw_text=command
            )

        # C. Independence Day & Republic Day (Tricolor)
        if re.search(r'\b(independence\s+day|swatantrata|15th\s+august|15\s+august|jai\s+hind|tiranga)\b', c):
            return IntentResult(
                intent=IntentType.GET_FESTIVAL,
                confidence=0.98,
                parameters={
                    "is_hindi": is_hindi,
                    "specific_festival": "Independence Day",
                    "overlay_tag": "FESTIVAL_INDEPENDENCE",
                    "theme": FestivalTheme.TRICOLOR.value,
                    "greeting_en": "Happy Independence Day! Jai Hind! Proud to celebrate our nation!",
                    "greeting_hi": "Swatantrata Diwas ki hardik shubhkamnayein! Jai Hind!"
                },
                raw_text=command
            )

        if re.search(r'\b(republic\s+day|gantantra|26th\s+january|26\s+january|vande\s+mataram)\b', c):
            return IntentResult(
                intent=IntentType.GET_FESTIVAL,
                confidence=0.98,
                parameters={
                    "is_hindi": is_hindi,
                    "specific_festival": "Republic Day",
                    "overlay_tag": "FESTIVAL_INDEPENDENCE",
                    "theme": FestivalTheme.TRICOLOR.value,
                    "greeting_en": "Happy Republic Day! Proud of our Constitution and nation!",
                    "greeting_hi": "Gantantra Diwas ki hardik shubhkamnayein! Vande Mataram!"
                },
                raw_text=command
            )

        # D. Diwali / Dussehra
        if re.search(r'\b(diwali|deepavali|deepawali|dussehra|vijayadashami|diya)\b', c):
            return IntentResult(
                intent=IntentType.GET_FESTIVAL,
                confidence=0.98,
                parameters={
                    "is_hindi": is_hindi,
                    "specific_festival": "Diwali",
                    "overlay_tag": "FESTIVAL_DIWALI",
                    "theme": FestivalTheme.DIWALI_DIYA.value,
                    "greeting_en": "Happy Diwali! May the festival of lights bring prosperity, peace, and brightness to your home!",
                    "greeting_hi": "Shubh Deepavali! Aapka din roshni, khushiyon aur mithaiyon se bhara rahe!"
                },
                raw_text=command
            )

        # E. Christmas
        if re.search(r'\b(christmas|xmas|santa|merry\s+christmas)\b', c):
            return IntentResult(
                intent=IntentType.GET_FESTIVAL,
                confidence=0.98,
                parameters={
                    "is_hindi": is_hindi,
                    "specific_festival": "Christmas",
                    "overlay_tag": "FESTIVAL_SANTA",
                    "theme": FestivalTheme.SANTA_HAT.value,
                    "greeting_en": "Merry Christmas! Wishing you peace, joy, and wonderful festivities!",
                    "greeting_hi": "Merry Christmas! Aaj khushiyan baantne aur parivaar ke saath manane ka din hai!"
                },
                raw_text=command
            )

        # F. New Year & Makar Sankranti
        if re.search(r'\b(new\s+year|naya\s+saal|nav\s+varsh)\b', c):
            return IntentResult(
                intent=IntentType.GET_FESTIVAL,
                confidence=0.98,
                parameters={
                    "is_hindi": is_hindi,
                    "specific_festival": "New Year",
                    "overlay_tag": "FESTIVAL_NEW_YEAR",
                    "theme": FestivalTheme.NEW_YEAR.value,
                    "greeting_en": "Happy New Year! Welcome to a fresh new journey of adventures!",
                    "greeting_hi": "Naya Saal Mubarak! Yeh saal aapke liye bohot saari khushiyan laye!"
                },
                raw_text=command
            )

        if re.search(r'\b(sankranti|makar\s+sankranti|kite)\b', c):
            return IntentResult(
                intent=IntentType.GET_FESTIVAL,
                confidence=0.98,
                parameters={
                    "is_hindi": is_hindi,
                    "specific_festival": "Makar Sankranti",
                    "overlay_tag": "FESTIVAL_KITE",
                    "theme": FestivalTheme.KITE.value,
                    "greeting_en": "Happy Makar Sankranti! May your life soar high like colorful kites!",
                    "greeting_hi": "Makar Sankranti ki shubhkamnayein! Til-gul ghya, god-god bola!"
                },
                raw_text=command
            )

        # 2. General Festival Query Patterns
        festival_query_patterns = [
            r'\b(what\s+festival\s+is\s+today|which\s+festival\s+is\s+today|is\s+today\s+a\s+festival|is\s+today\s+a\s+holiday|today\s+festival|any\s+holiday\s+today)\b',
            r'\b(aaj\s+kaun\s+sa\s+festival\s+hai|aaj\s+kya\s+tyohar\s+hai|aaj\s+chhutti\s+hai\s+kya|aaj\s+koi\s+festival\s+hai|festival\s+batao|tyohar\s+batao)\b',
            r'\b(festival|tyohar|holidays)\b'
        ]

        for p in festival_query_patterns:
            if re.search(p, c):
                fest_today = self.service.get_todays_festival()
                return IntentResult(
                    intent=IntentType.GET_FESTIVAL,
                    confidence=0.95,
                    parameters={
                        "is_hindi": is_hindi,
                        "today_festival": fest_today.name if fest_today else None
                    },
                    raw_text=command
                )

        return None
