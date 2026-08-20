"""
Morphix Desktop Companion Robot - Offline To-Do System Test Suite
==================================================================
File: tests/test_todo_system.py
Description:
    Comprehensive verification tests for the updated To-Do System:
    Add, Delete, Edit, Complete, Show, Empty List, Duplicate Detection,
    Restart Persistence, JSON Integrity, updated_at metadata, Exact matching,
    No substring false matches, and Arbitrary-length tasks.
"""

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

# Ensure project root is in sys.path for clean import resolution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.todo.todo_manager import TodoManager
from src.intent.todo_intent import TodoIntentParser as TodoParser


class TestTodoSystem(unittest.TestCase):
    """
    To-Do Management System verification suite.
    """

    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.test_json = Path(self.temp_dir.name) / "todos.json"
        self.mgr = TodoManager(file_path=str(self.test_json))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_add_todo(self) -> None:
        response = self.mgr.add_todo("submit internship application")
        self.assertIn("submit internship application", response)
        pending = self.mgr.get_pending_todos()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0], "submit internship application")

    def test_delete_todo(self) -> None:
        self.mgr.add_todo("finish robotics report")
        deleted = self.mgr.delete_todo("finish robotics report")
        self.assertTrue(deleted)
        self.assertEqual(len(self.mgr.get_pending_todos()), 0)

    def test_edit_todo_and_updated_at(self) -> None:
        self.mgr.add_todo("finish OpenCV project")
        todo = self.mgr.tasks[0]
        self.assertIsNone(todo.updated_at)

        edited = self.mgr.edit_todo("finish OpenCV project", "prepare IEEE paper")
        self.assertIsNotNone(edited)
        self.assertEqual(edited["task"], "prepare IEEE paper")
        self.assertIsNotNone(edited["updated_at"])

        # Verify persistence after edit
        all_todos = self.mgr.get_all_tasks()
        self.assertEqual(all_todos[0]["task"], "prepare IEEE paper")
        self.assertEqual(all_todos[0]["updated_at"], edited["updated_at"])

    def test_mark_completed(self) -> None:
        self.mgr.add_todo("book train tickets")
        completed = self.mgr.complete_todo("book train tickets")
        self.assertIsNotNone(completed)
        self.assertTrue(completed["completed"])
        self.assertIsNotNone(completed["completed_at"])

    def test_show_todo(self) -> None:
        self.mgr.add_todo("renew passport")
        self.mgr.add_todo("buy Raspberry Pi camera")
        pending = self.mgr.get_pending_todos()
        self.assertEqual(len(pending), 2)
        self.assertIn("renew passport", pending)
        self.assertIn("buy Raspberry Pi camera", pending)

    def test_empty_todo_list(self) -> None:
        self.assertEqual(len(self.mgr.get_pending_todos()), 0)

    def test_duplicate_todo(self) -> None:
        self.mgr.add_todo("call Tanmay")
        with self.assertRaises(ValueError):
            self.mgr.add_todo("call Tanmay")

    def test_json_save_and_load(self) -> None:
        self.mgr.add_todo("birthday planning")
        # Read raw json to verify integrity
        with open(self.test_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["task"], "birthday planning")

    def test_restart_persistence(self) -> None:
        self.mgr.add_todo("water the plants")
        # Instantiate new manager pointing to same file
        new_mgr = TodoManager(file_path=str(self.test_json))
        pending = new_mgr.get_pending_todos()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0], "water the plants")

    def test_arbitrary_length_tasks(self) -> None:
        long_task = "Finish the Morphix reminder documentation before tomorrow evening and upload the latest version to GitHub."
        self.mgr.add_todo(long_task)

        # Reload and check
        new_mgr = TodoManager(file_path=str(self.test_json))
        pending = new_mgr.get_pending_todos()
        self.assertEqual(pending[0], long_task)

    def test_strict_normalized_matching(self) -> None:
        # Match case-insensitive
        self.mgr.add_todo("Pay Electricity Bill")
        completed = self.mgr.complete_todo("pay electricity bill")
        self.assertIsNotNone(completed)
        self.assertTrue(completed["completed"])

        # Match whitespace-normalized
        self.mgr.add_todo("GitHub   upload")
        deleted = self.mgr.delete_todo("github upload")
        self.assertTrue(deleted)

    def test_no_substring_false_matches(self) -> None:
        # "milk" should NOT match "milkshake"
        self.mgr.add_todo("milkshake")
        completed = self.mgr.complete_todo("milk")
        self.assertIsNone(completed)

        # "room" should NOT match "bedroom"
        self.mgr.add_todo("bedroom")
        deleted = self.mgr.delete_todo("room")
        self.assertFalse(deleted)

    def test_parser_natural_add_variations(self) -> None:
        parsed_adds = [
            ("add finish robotics report to my todo", "finish robotics report"),
            ("put submit internship application in my todo list", "submit internship application"),
            ("add task buy Raspberry Pi camera module to my todo", "buy Raspberry Pi camera module"),
            ("add remember to call Tanmay after class to my todo", "remember to call Tanmay after class"),
            ("put water the plants in my todo list", "water the plants"),
            ("add buy notebook in my todo", "buy notebook"),
            ("todo mein recharge karna add kar do", "recharge karna")
        ]

        for text, expected_item in parsed_adds:
            with self.subTest(text=text):
                parsed = TodoParser.parse(text)
                self.assertEqual(parsed["intent"], "ADD_TODO")
                self.assertEqual(parsed["item"], expected_item)

    def test_parser_natural_show_variations(self) -> None:
        show_texts = [
            "show my todo",
            "show my todos",
            "show me my todo",
            "show me my todos",
            "list my todos",
            "list my todo list",
            "what are my todos",
            "read my todo list",
            "tell me my todos",
            "mera todo dikhao",
            "mere todo batao",
            "todo kya kya hai"
        ]
        for text in show_texts:
            with self.subTest(text=text):
                parsed = TodoParser.parse(text)
                self.assertEqual(parsed["intent"], "SHOW_TODO")

    def test_parser_natural_delete_variations(self) -> None:
        delete_texts = [
            ("delete passport renewal from my todo", "passport renewal"),
            ("remove passport renewal from my todo list", "passport renewal"),
            ("complete passport renewal from my todo", "passport renewal"),
            ("finish passport renewal in my todo list", "passport renewal"),
            ("delete buy notebook from my todo", "buy notebook"),
            ("buy milk todo se delete kar do", "buy milk")
        ]
        for text, expected in delete_texts:
            with self.subTest(text=text):
                parsed = TodoParser.parse(text)
                self.assertEqual(parsed["intent"], "DELETE_TODO")
                self.assertEqual(parsed["item"], expected)

    def test_parser_natural_clear_all_variations(self) -> None:
        clear_texts = [
            "delete all my todos",
            "clear all my todos",
            "remove all my todos",
            "delete all todos",
            "clear all todos list",
            "remove all my todo list",
            "clear todo list",
            "saare todo delete kar do",
            "saare todo clear karo"
        ]
        for text in clear_texts:
            with self.subTest(text=text):
                parsed = TodoParser.parse(text)
                self.assertEqual(parsed["intent"], "CLEAR_TODO")

    def test_corrupted_json_recovery(self) -> None:
        # Intentionally write corrupted JSON to file
        with open(self.test_json, "w", encoding="utf-8") as f:
            f.write("{invalid_json_corrupted: true}")

        # Re-initialize manager, which should recover cleanly by starting empty
        recovered_mgr = TodoManager(file_path=str(self.test_json))
        self.assertEqual(len(recovered_mgr.get_all_tasks()), 0)

        # Verify adding task after recovery works successfully
        recovered_mgr.add_todo("finish OpenCV project")
        self.assertEqual(len(recovered_mgr.get_all_tasks()), 1)

    def test_non_existent_task_handling(self) -> None:
        # Verify non-existent query results in None safely
        self.assertIsNone(self.mgr.complete_todo("non-existent task"))
        self.assertFalse(self.mgr.delete_todo("non-existent task"))
        self.assertIsNone(self.mgr.edit_todo("non-existent task", "new task text"))


if __name__ == "__main__":
    unittest.main()
