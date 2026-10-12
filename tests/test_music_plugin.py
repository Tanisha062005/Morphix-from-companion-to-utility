"""
Morphix Desktop Companion Robot - Music Plugin Test Suite
==========================================================
File: tests/test_music_plugin.py
Description:
    Unit tests for MusicService and MusicIntentPlugin matching.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.intent.intent_classifier import IntentClassifier, IntentType
from src.intent.music_plugin import MusicIntentPlugin, MusicService, clean_leading_fillers


class TestMusicPlugin(unittest.TestCase):
    """Verifies music intent classification, filler stripping, query extraction, and controls."""

    def setUp(self) -> None:
        self.service = MusicService()
        self.plugin = MusicIntentPlugin(service=self.service)

    def test_query_extraction_and_filler_stripping(self) -> None:
        cases = [
            ("ok play some calm music", "calm music"),
            ("bro play from movie Tu Hai Meri Hindi song", "from movie Tu Hai Meri Hindi"),
            ("play Believer", "Believer"),
            ("can you play arijit singh songs", "arijit singh songs"),
            ("song play karo kesariya", "kesariya"),
            ("morphix please play shape of you", "shape of you"),
            ("bajao believer", "believer"),
            ("chalao koi accha gaana", "koi accha"),
            ("play music", "top hits trending"),
            ("hey bro please play perfect by ed sheeran for me", "perfect by ed sheeran"),
            ("can you please play shape of you for me", "shape of you"),
            ("kesariya play karo", "kesariya"),
        ]
        for user_cmd, expected_query in cases:
            res = self.plugin.match(user_cmd)
            self.assertIsNotNone(res, f"Failed to match command: {user_cmd}")
            self.assertEqual(res.intent, IntentType.PLAY_MUSIC)
            self.assertEqual(res.parameters.get("query"), expected_query, f"Failed query extract for: {user_cmd}")

    def test_intent_matching_controls(self) -> None:
        # Pause
        for p in ["pause music", "bro pause the music", "pause", "gaana roko", "pause karo"]:
            res = self.plugin.match(p)
            self.assertIsNotNone(res, f"Failed to match pause: {p}")
            self.assertEqual(res.intent, IntentType.PAUSE_MUSIC)

        # Resume
        for r in ["resume music", "ok resume the song", "resume", "chalu karo", "continue song"]:
            res = self.plugin.match(r)
            self.assertIsNotNone(res, f"Failed to match resume: {r}")
            self.assertEqual(res.intent, IntentType.RESUME_MUSIC)

        # Stop
        for s in ["stop music", "hey morphix stop the song", "stop playing", "band karo", "gaana band karo", "stop"]:
            res = self.plugin.match(s)
            self.assertIsNotNone(res, f"Failed to match stop: {s}")
            self.assertEqual(res.intent, IntentType.STOP_MUSIC)

        # Next
        for n in ["next song", "bro play next", "next track", "agla gaana", "next"]:
            res = self.plugin.match(n)
            self.assertIsNotNone(res, f"Failed to match next: {n}")
            self.assertEqual(res.intent, IntentType.NEXT_MUSIC)

    def test_speech_formatting(self) -> None:
        self.assertIn("Believer", self.service.format_speech_response("play", title="Believer", is_hindi=False))
        self.assertIn("Believer", self.service.format_speech_response("play", title="Believer", is_hindi=True))
        self.assertIn("paused", self.service.format_speech_response("pause", is_hindi=False))
        self.assertIn("pause", self.service.format_speech_response("pause", is_hindi=True))

    def test_classifier_integration(self) -> None:
        classifier = IntentClassifier()
        res = classifier.classify("ok play some calm music")
        self.assertEqual(res.intent, IntentType.PLAY_MUSIC)
        self.assertEqual(res.parameters.get("query"), "calm music")


    def test_audio_ducking(self) -> None:
        self.service._volume = 0.8
        self.service._pre_duck_volume = 0.8

        # Mock is_playing to True for ducking test
        with patch.object(self.service, 'is_playing', return_value=True):
            self.service.duck_for_tts()
            self.assertEqual(self.service._volume, 0.18)
            self.assertTrue(self.service._is_ducked)

            self.service.restore_after_tts()
            self.assertEqual(self.service._volume, 0.8)
            self.assertFalse(self.service._is_ducked)

    def test_next_track_queue(self) -> None:
        from src.intent.music_plugin import TrackInfo
        # Test fallback when no history
        q_fallback = self.service._build_next_query()
        self.assertEqual(q_fallback, "top trending songs today")

        # Test history push and smart next query
        self.service._push_history("play believer", "Believer - Imagine Dragons")
        self.service.current_track = TrackInfo(title="Believer - Imagine Dragons", url="http://test")
        q_next = self.service._build_next_query()
        self.assertIn("Imagine Dragons", q_next)


if __name__ == "__main__":
    unittest.main()

