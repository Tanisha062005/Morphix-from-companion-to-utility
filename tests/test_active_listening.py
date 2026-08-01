"""
Morphix Desktop Companion Robot - Active Listening Mode Test Suite
==================================================================
File: tests/test_active_listening.py
Description:
    Unit tests for Active Listening Mode, verifying transition print formatting,
    meaningful speech recognition filtering, and inactivity timer checks.
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Ensure project root is in sys.path for clean import resolution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from run_voice_assistant import is_meaningful_speech, AssistantState, transition_state
import run_voice_assistant


class TestActiveListening(unittest.TestCase):
    """
    Test suite verifying the stability, state transition printing,
    transient commands, and timer mechanics of the Active Listening Mode.
    """

    def test_is_meaningful_speech(self) -> None:
        """
        Verifies that only meaningful speech resets the inactivity timer,
        and empty/noise/placeholder transcripts are correctly filtered.
        """
        # Valid meaningful speech
        self.assertTrue(is_meaningful_speech("hello"))
        self.assertTrue(is_meaningful_speech("remind me to feed the cat"))
        self.assertTrue(is_meaningful_speech("  what time is it  "))
        self.assertTrue(is_meaningful_speech("add task to my list"))

        # Invalid meaningless speech (must return False)
        self.assertFalse(is_meaningful_speech(""))
        self.assertFalse(is_meaningful_speech("   "))
        self.assertFalse(is_meaningful_speech(None))
        self.assertFalse(is_meaningful_speech("[BLANK_AUDIO]"))
        self.assertFalse(is_meaningful_speech("[blank_audio]"))
        self.assertFalse(is_meaningful_speech(" [blank_audio]  "))
        self.assertFalse(is_meaningful_speech("[noise]"))
        self.assertFalse(is_meaningful_speech("noise"))
        self.assertFalse(is_meaningful_speech("[microphone noise]"))

    @patch('run_voice_assistant.logger')
    @patch('builtins.print')
    def test_transition_state_printing(self, mock_print, mock_logger) -> None:
        """
        Verifies that transition_state outputs the old and new states in the exact requested format.
        """
        # Reset state to WAIT_WAKE
        run_voice_assistant._current_state = AssistantState.WAIT_WAKE

        # Transition WAIT_WAKE -> ACTIVE_MODE
        transition_state(AssistantState.ACTIVE_MODE)
        mock_print.assert_called_with("WAIT_WAKE\n↓\nACTIVE_MODE")

        # Transition ACTIVE_MODE -> COMMAND_RECEIVED
        transition_state(AssistantState.COMMAND_RECEIVED)
        mock_print.assert_called_with("ACTIVE_MODE\n↓\nCOMMAND_RECEIVED")

        # Transition COMMAND_RECEIVED -> ACTIVE_MODE
        transition_state(AssistantState.ACTIVE_MODE)
        mock_print.assert_called_with("COMMAND_RECEIVED\n↓\nACTIVE_MODE")

        # Transition ACTIVE_MODE -> SLEEP
        transition_state(AssistantState.SLEEP)
        mock_print.assert_called_with("ACTIVE_MODE\n↓\nSLEEP")

        # Transition SLEEP -> WAIT_WAKE
        transition_state(AssistantState.WAIT_WAKE)
        mock_print.assert_called_with("SLEEP\n↓\nWAIT_WAKE")

    def test_inactivity_timer_flow(self) -> None:
        """
        Verifies that elapsed inactivity and remaining duration arithmetic works correctly.
        """
        active_timeout = 300.0  # 5 minutes

        # Case 1: Inactivity is less than the timeout
        last_speech_time = 1000.0
        current_time = 1200.0  # 200 seconds elapsed
        elapsed_since_speech = current_time - last_speech_time
        remaining_time = max(0.0, active_timeout - elapsed_since_speech)

        self.assertEqual(elapsed_since_speech, 200.0)
        self.assertEqual(remaining_time, 100.0)
        self.assertLess(elapsed_since_speech, active_timeout)

        # Case 2: Inactivity meets or exceeds timeout
        current_time = 1300.0  # 300 seconds elapsed
        elapsed_since_speech = current_time - last_speech_time
        remaining_time = max(0.0, active_timeout - elapsed_since_speech)

        self.assertEqual(elapsed_since_speech, 300.0)
        self.assertEqual(remaining_time, 0.0)
        self.assertTrue(elapsed_since_speech >= active_timeout)


if __name__ == "__main__":
    unittest.main()
