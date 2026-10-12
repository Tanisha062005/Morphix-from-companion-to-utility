"""
Morphix Desktop Companion Robot - Eye Expressions Unit Tests
============================================================
File: tests/test_expressions.py
Description:
    Unit tests for EyeAnimator states, categories, and ExpressionManager integration.
"""

import unittest
from src.emotions.expression_manager import ExpressionManager


class TestExpressions(unittest.TestCase):
    """Unit tests for ExpressionManager and EyeState bridge."""

    def setUp(self) -> None:
        self.mgr = ExpressionManager()

    def test_expression_defaults_and_switching(self) -> None:
        self.assertEqual(self.mgr.get_current_expression(), "Neutral")
        self.assertEqual(self.mgr.get_emoji(), "😐")

        self.mgr.set_expression("Happy")
        self.assertEqual(self.mgr.get_current_expression(), "Happy")
        self.assertEqual(self.mgr.get_emoji(), "😊")

        # Invalid fallback test
        self.mgr.set_expression("InvalidExpressionName")
        self.assertEqual(self.mgr.get_current_expression(), "Neutral")

    def test_to_eye_state_bridge(self) -> None:
        from src.display.eye_animator import EyeState

        self.mgr.set_expression("Happy")
        self.assertEqual(self.mgr.to_eye_state(), EyeState.HAPPY)

        self.mgr.set_expression("Sleeping")
        self.assertEqual(self.mgr.to_eye_state(), EyeState.DEEP_SLEEP)

        self.mgr.set_expression("Thinking")
        self.assertEqual(self.mgr.to_eye_state(), EyeState.THINKING)

    def test_available_expressions(self) -> None:
        exprs = self.mgr.get_available_expressions()
        self.assertIn("Neutral", exprs)
        self.assertIn("Happy", exprs)
        self.assertIn("Sad", exprs)
        self.assertIn("Protective", exprs)


if __name__ == "__main__":
    unittest.main()
