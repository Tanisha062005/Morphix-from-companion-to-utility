"""
Morphix Desktop Companion Robot - Personality Test Suite
=========================================================
File: tests/test_personality.py
Description:
    Demonstrates and tests all public methods of the Personality class
    from personality.py, verifying identity outputs, dialogue messages,
    and valid/invalid mood state transitions.
"""

from pathlib import Path
import sys

# Ensure project root directory is in sys.path for clean script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from personality import Personality


def run_tests() -> None:
    """
    Executes tests for all public methods in the Personality class.
    """
    # Instantiate the Personality object
    bot = Personality()

    # 1. Test introduce()
    print("==============================")
    print("Testing Introduction")
    print("==============================")
    print(bot.introduce())
    print()

    # 2. Test greet()
    print("==============================")
    print("Testing Greeting")
    print("==============================")
    print(bot.greet())
    print()

    # 3. Test goodbye()
    print("==============================")
    print("Testing Goodbye")
    print("==============================")
    print(bot.goodbye())
    print()

    # 4. Test idle_message()
    print("==============================")
    print("Testing Idle Message")
    print("==============================")
    print(bot.idle_message())
    print()

    # 5. Test error_message()
    print("==============================")
    print("Testing Error Message")
    print("==============================")
    print(bot.error_message())
    print()

    # 6. Test small_talk()
    print("==============================")
    print("Testing Small Talk")
    print("==============================")
    print(bot.small_talk())
    print()

    # 7. Test get_current_mood()
    print("==============================")
    print("Testing Get Current Mood")
    print("==============================")
    print(f"Initial Mood: {bot.get_current_mood()}")
    print()

    # 8. Test set_mood() with a valid mood
    print("==============================")
    print("Testing Valid Mood Change")
    print("==============================")
    new_mood = "Excited"
    success = bot.set_mood(new_mood)
    print(f"Attempted to set mood to '{new_mood}' -> Success: {success}")
    print(f"Updated Current Mood: {bot.get_current_mood()}")
    print()

    # 9. Test set_mood() with an invalid mood
    print("==============================")
    print("Testing Invalid Mood Change")
    print("==============================")
    invalid_mood = "SuperAngry"  # Mood not in available_moods list
    success = bot.set_mood(invalid_mood)
    print(f"Attempted to set mood to '{invalid_mood}' -> Success: {success}")
    print(f"Current Mood (should remain '{bot.get_current_mood()}'): {bot.get_current_mood()}")
    print()

    print("==============================")
    print("All Personality tests completed!")
    print("==============================")


if __name__ == "__main__":
    run_tests()
