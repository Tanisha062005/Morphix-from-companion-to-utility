"""
Morphix Desktop Companion Robot - To-Do Manager Unit Tests
===========================================================
File: tests/test_todo.py
"""

from pathlib import Path
import tempfile
import unittest
from src.todo.todo_manager import TodoManager


class TestTodoManager(unittest.TestCase):
    """
    Tests TodoManager CRUD operations and JSON persistence.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_json = Path(self.temp_dir.name) / "test_todo.json"
        self.manager = TodoManager(file_path=str(self.test_json))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_add_task(self) -> None:
        task = self.manager.add_task("Buy milk")
        self.assertIsNotNone(task.get("id"))
        self.assertEqual(task.get("task"), "Buy milk")
        self.assertFalse(task.get("completed"))
        self.assertIn("T", task.get("created_at", ""))

        all_tasks = self.manager.get_all_tasks()
        self.assertEqual(len(all_tasks), 1)

    def test_pending_tasks(self) -> None:
        t1 = self.manager.add_task("Buy milk")
        t2 = self.manager.add_task("Finish robotics project")

        pending = self.manager.get_pending_tasks()
        self.assertEqual(len(pending), 2)

        self.manager.complete_task("Buy milk")
        pending_after = self.manager.get_pending_tasks()
        self.assertEqual(len(pending_after), 1)
        self.assertEqual(pending_after[0]["task"], "Finish robotics project")

    def test_complete_task(self) -> None:
        t1 = self.manager.add_task("Call mom")
        res = self.manager.complete_task("Call mom")
        self.assertIsNotNone(res)
        self.assertTrue(res.get("completed"))

    def test_delete_task(self) -> None:
        self.manager.add_task("Buy milk")
        self.manager.add_task("Call mom")

        deleted = self.manager.delete_task("Buy milk")
        self.assertIsNotNone(deleted)
        self.assertEqual(deleted.get("task"), "Buy milk")

        remaining = self.manager.get_all_tasks()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["task"], "Call mom")

    def test_clear_all_tasks(self) -> None:
        self.manager.add_task("Task 1")
        self.manager.add_task("Task 2")
        self.manager.add_task("Task 3")

        cleared_count = self.manager.clear_all_tasks()
        self.assertEqual(cleared_count, 3)
        self.assertEqual(len(self.manager.get_all_tasks()), 0)

    def test_auto_recovery_on_corrupted_json(self) -> None:
        # Intentionally write invalid JSON
        with open(self.test_json, "w", encoding="utf-8") as f:
            f.write("{invalid_json_corrupted: true}")

        new_manager = TodoManager(file_path=str(self.test_json))
        tasks = new_manager.get_all_tasks()
        self.assertEqual(len(tasks), 0)

        # Ensure adding task works after recovery
        added = new_manager.add_task("Recovered task")
        self.assertEqual(added.get("task"), "Recovered task")


if __name__ == "__main__":
    unittest.main()
