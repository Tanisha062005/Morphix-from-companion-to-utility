"""
Morphix Desktop Companion Robot - Festival Plugin Unit Tests
=============================================================
File: tests/test_festival_plugin.py
"""

from datetime import date
import unittest
from src.intent.festival_plugin import FestivalInfo, FestivalService, FestivalTheme, FestivalIntentPlugin
from src.intent.intent_classifier import IntentType


class TestFestivalPlugin(unittest.TestCase):
    def setUp(self):
        self.service = FestivalService.get_instance()
        self.plugin = FestivalIntentPlugin(self.service)

    def test_fixed_national_festivals(self):
        # 26 Jan: Republic Day
        rep_day = self.service.get_todays_festival(date(2026, 1, 26))
        self.assertIsNotNone(rep_day)
        self.assertEqual(rep_day.theme, FestivalTheme.TRICOLOR)
        self.assertIn("Republic Day", rep_day.name)

        # 15 Aug: Independence Day
        ind_day = self.service.get_todays_festival(date(2026, 8, 15))
        self.assertIsNotNone(ind_day)
        self.assertEqual(ind_day.theme, FestivalTheme.TRICOLOR)

        # 25 Dec: Christmas
        xmas = self.service.get_todays_festival(date(2026, 12, 25))
        self.assertIsNotNone(xmas)
        self.assertEqual(xmas.theme, FestivalTheme.SANTA_HAT)

        # 14 Jan: Makar Sankranti
        sankranti = self.service.get_todays_festival(date(2026, 1, 14))
        self.assertIsNotNone(sankranti)
        self.assertEqual(sankranti.theme, FestivalTheme.KITE)

    def test_dynamic_maharashtra_festivals(self):
        # 14 Sept 2026: Ganesh Chaturthi
        ganesh = self.service.get_todays_festival(date(2026, 9, 14))
        self.assertIsNotNone(ganesh)
        self.assertEqual(ganesh.theme, FestivalTheme.MODAK)
        self.assertIn("Ganpati Bappa Morya", ganesh.greeting_hi)

        # 8 Nov 2026: Diwali
        diwali = self.service.get_todays_festival(date(2026, 11, 8))
        self.assertIsNotNone(diwali)
        self.assertEqual(diwali.theme, FestivalTheme.DIWALI_DIYA)

        # 3 March 2026: Holi
        holi = self.service.get_todays_festival(date(2026, 3, 3))
        self.assertIsNotNone(holi)
        self.assertEqual(holi.theme, FestivalTheme.HOLI_SPLASH)

    def test_intent_matching(self):
        res1 = self.plugin.match("aaj kaun sa festival hai")
        self.assertIsNotNone(res1)
        self.assertEqual(res1.intent, IntentType.GET_FESTIVAL)

        res2 = self.plugin.match("is today a holiday")
        self.assertIsNotNone(res2)
        self.assertEqual(res2.intent, IntentType.GET_FESTIVAL)

        res3 = self.plugin.match("happy diwali")
        self.assertIsNotNone(res3)
        self.assertEqual(res3.intent, IntentType.GET_FESTIVAL)

        res4 = self.plugin.match("ganpati bappa morya")
        self.assertIsNotNone(res4)
        self.assertEqual(res4.intent, IntentType.GET_FESTIVAL)


if __name__ == "__main__":
    unittest.main()
