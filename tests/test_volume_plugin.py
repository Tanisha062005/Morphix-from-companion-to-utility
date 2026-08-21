"""
Morphix Desktop Companion Robot - Volume Controller Test Suite
===============================================================
File: tests/test_volume_plugin.py
Description:
    Unit tests for VolumeService and VolumeIntentPlugin matching.
"""

import unittest
from src.intent.intent_classifier import IntentClassifier, IntentType
from src.intent.volume_plugin import VolumeIntentPlugin, VolumeService


class TestVolumePlugin(unittest.TestCase):
    """Verifies volume intent parsing, scaling, and speech formatting."""

    def setUp(self) -> None:
        self.service = VolumeService()
        self.plugin = VolumeIntentPlugin(service=self.service)

    def test_volume_clamping(self) -> None:
        self.assertEqual(self.service.set_volume(50), 50)
        self.assertEqual(self.service.set_volume(150), 100)
        self.assertEqual(self.service.set_volume(-20), 0)

    def test_increase_decrease(self) -> None:
        self.service.set_volume(50)
        new_vol = self.service.increase_volume(20)
        self.assertEqual(new_vol, 70)

        new_vol = self.service.decrease_volume(30)
        self.assertEqual(new_vol, 40)

    def test_mute_unmute(self) -> None:
        self.service.set_volume(80)
        self.service.mute()
        self.assertEqual(self.service.get_volume(), 0)

        self.service.unmute()
        self.assertEqual(self.service.get_volume(), 80)

    def test_intent_matching_level(self) -> None:
        res = self.plugin.match("set volume to 80")
        self.assertIsNotNone(res)
        self.assertEqual(res.intent, IntentType.SET_VOLUME)
        self.assertEqual(res.parameters.get("level"), 80)

        res2 = self.plugin.match("volume 50 percent")
        self.assertIsNotNone(res2)
        self.assertEqual(res2.intent, IntentType.SET_VOLUME)
        self.assertEqual(res2.parameters.get("level"), 50)

    def test_intent_matching_adjustments(self) -> None:
        # Increase
        res = self.plugin.match("volume badhao")
        self.assertIsNotNone(res)
        self.assertEqual(res.intent, IntentType.INCREASE_VOLUME)

        res_en = self.plugin.match("turn it up")
        self.assertIsNotNone(res_en)
        self.assertEqual(res_en.intent, IntentType.INCREASE_VOLUME)

        # Decrease
        res_dec = self.plugin.match("volume kam karo")
        self.assertIsNotNone(res_dec)
        self.assertEqual(res_dec.intent, IntentType.DECREASE_VOLUME)

        res_quieter = self.plugin.match("turn it down")
        self.assertIsNotNone(res_quieter)
        self.assertEqual(res_quieter.intent, IntentType.DECREASE_VOLUME)

        # Mute
        res_mute = self.plugin.match("mute")
        self.assertIsNotNone(res_mute)
        self.assertEqual(res_mute.intent, IntentType.MUTE_VOLUME)

    def test_classifier_integration(self) -> None:
        classifier = IntentClassifier()
        res = classifier.classify("volume 90%")
        self.assertEqual(res.intent, IntentType.SET_VOLUME)


if __name__ == "__main__":
    unittest.main()
