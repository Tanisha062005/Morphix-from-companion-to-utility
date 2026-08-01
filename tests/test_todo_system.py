"""
Morphix Desktop Companion Robot - Offline To-Do System Test Suite
==================================================================
File: tests/test_todo_system.py
Description:
    Comprehensive verification tests for the new To-Do System:
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
from src.todo.todo_parser import TodoParser


class TestTodoSystem(unittest.TestCase):
    """
    To-Do Management System verification suite.
    """

    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.test_json = Path(self.temp_dir.name) / "todo.json"
        self.mgr = TodoManager(file_path=str(self.test_json))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_add_todo(self) -> None:
        todo = self.mgr.add_todo("submit internship application")
        self.assertIsNotNone(todo["id"])
        self.assertEqual(todo["task"], "submit internship application")
        self.assertFalse(todo["completed"])
        self.assertIsNone(todo["completed_at"])
        self.assertIsNone(todo["updated_at"])

    def test_delete_todo(self) -> None:
        todo = self.mgr.add_todo("finish robotics report")
        deleted = self.mgr.delete_todo("finish robotics report")
        self.assertIsNotNone(deleted)
        self.assertEqual(deleted["id"], todo["id"])
        self.assertEqual(len(self.mgr.get_pending_tasks()), 0)

    def test_edit_todo_and_updated_at(self) -> None:
        todo = self.mgr.add_todo("finish OpenCV project")
        self.assertIsNone(todo["updated_at"])

        edited = self.mgr.edit_todo("finish OpenCV project", "prepare IEEE paper")
        self.assertIsNotNone(edited)
        self.assertEqual(edited["task"], "prepare IEEE paper")
        self.assertIsNotNone(edited["updated_at"])

        # Verify persistence after edit
        all_todos = self.mgr.get_all_tasks()
        self.assertEqual(all_todos[0]["task"], "prepare IEEE paper")
        self.assertEqual(all_todos[0]["updated_at"], edited["updated_at"])

    def test_mark_completed(self) -> None:
        todo = self.mgr.add_todo("book train tickets")
        completed = self.mgr.complete_todo("book train tickets")
        self.assertIsNotNone(completed)
        self.assertTrue(completed["completed"])
        self.assertIsNotNone(completed["completed_at"])

    def test_show_todo(self) -> None:
        self.mgr.add_todo("renew passport")
        self.mgr.add_todo("buy Raspberry Pi camera")
        pending = self.mgr.get_pending_tasks()
        self.assertEqual(len(pending), 2)

    def test_empty_todo_list(self) -> None:
        self.assertEqual(len(self.mgr.get_pending_tasks()), 0)

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
        pending = new_mgr.get_pending_tasks()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["task"], "water the plants")

    def test_arbitrary_length_tasks(self) -> None:
        long_task = "Finish the Morphix reminder documentation before tomorrow evening and upload the latest version to GitHub."
        todo = self.mgr.add_todo(long_task)
        self.assertEqual(todo["task"], long_task)

        # Reload and check
        new_mgr = TodoManager(file_path=str(self.test_json))
        all_todos = new_mgr.get_all_tasks()
        self.assertEqual(all_todos[0]["task"], long_task)

    def test_strict_normalized_matching(self) -> None:
        # Match case-insensitive
        self.mgr.add_todo("Pay Electricity Bill")
        completed = self.mgr.complete_todo("pay electricity bill")
        self.assertIsNotNone(completed)
        self.assertTrue(completed["completed"])

        # Match whitespace-normalized
        self.mgr.add_todo("GitHub   upload")
        deleted = self.mgr.delete_todo("github upload")
        self.assertIsNotNone(deleted)

    def test_no_substring_false_matches(self) -> None:
        # "milk" should NOT match "milkshake"
        self.mgr.add_todo("milkshake")
        completed = self.mgr.complete_todo("milk")
        self.assertIsNone(completed)

        # "room" should NOT match "bedroom"
        self.mgr.add_todo("bedroom")
        deleted = self.mgr.delete_todo("room")
        self.assertIsNone(deleted)

    def test_parser_natural_add_variations(self) -> None:
        parsed_adds = [
            ("add finish robotics report before Friday", "finish robotics report before Friday"),
            ("please add submit internship application tonight", "submit internship application tonight"),
            ("can you add buy Raspberry Pi camera module", "buy Raspberry Pi camera module"),
            ("could you add remember to call Tanmay after class", "remember to call Tanmay after class"),
            ("remember water the plants", "water the plants"),
            ("remember to pay electricity bill", "pay electricity bill"),
            ("put birthday planning in my todo", "birthday planning"),
            ("create a task Morphix documentation", "Morphix documentation")
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
            "show me my tasks",
            "list my tasks",
            "list my todos",
            "what are my todos"
        ]
        for text in show_texts:
            with self.subTest(text=text):
                parsed = TodoParser.parse(text)
                self.assertEqual(parsed["intent"], "SHOW_TODO")

    def test_parser_natural_delete_variations(self) -> None:
        delete_texts = [
            ("delete passport renewal", "passport renewal"),
            ("remove passport renewal", "passport renewal"),
            ("remove my passport renewal", "passport renewal"),
            ("delete my passport renewal", "passport renewal")
        ]
        for text, expected in delete_texts:
            with self.subTest(text=text):
                parsed = TodoParser.parse(text)
                self.assertEqual(parsed["intent"], "DELETE_TODO")
                self.assertEqual(parsed["item"], expected)

    def test_parser_natural_complete_variations(self) -> None:
        complete_texts = [
            ("complete book train tickets", "book train tickets"),
            ("mark book train tickets completed", "book train tickets"),
            ("mark book train tickets done", "book train tickets"),
            ("tick book train tickets", "book train tickets"),
            ("done book train tickets", "book train tickets"),
            ("done with book train tickets", "book train tickets"),
            ("mark book train tickets as completed", "book train tickets"),
            ("mark book train tickets as done", "book train tickets")
        ]
        for text, expected in complete_texts:
            with self.subTest(text=text):
                parsed = TodoParser.parse(text)
                self.assertEqual(parsed["intent"], "COMPLETE_TODO")
                self.assertEqual(parsed["item"], expected)

    def test_parser_natural_edit_variations(self) -> None:
        edit_texts = [
            ("edit robotics report to AI report", "robotics report", "AI report"),
            ("rename project presentation to final presentation", "project presentation", "final presentation"),
            ("change buy servo to buy MG996R servo", "buy servo", "buy MG996R servo"),
            ("modify buy servo to buy MG996R servo", "buy servo", "buy MG996R servo")
        ]
        for text, expected_old, expected_new in edit_texts:
            with self.subTest(text=text):
                parsed = TodoParser.parse(text)
                self.assertEqual(parsed["intent"], "EDIT_TODO")
                self.assertEqual(parsed["old_item"], expected_old)
                self.assertEqual(parsed["new_item"], expected_new)

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
        self.assertIsNone(self.mgr.delete_todo("non-existent task"))
        self.assertIsNone(self.mgr.edit_todo("non-existent task", "new task text"))


if __name__ == "__main__":
    unittest.main()
