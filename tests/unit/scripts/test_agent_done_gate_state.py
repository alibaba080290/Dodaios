"""Read gate state only from temporary files, without running the gate."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.agent_done_gate import Ctx, STATE_DIR, load_state


class LoadStateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gate-state-test-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.ctx = Ctx(cwd=root, common_dir=root, run=self.reject_command)
        self.path = root / STATE_DIR / "sessions" / "session.json"
        self.path.parent.mkdir(parents=True)

    def reject_command(self, *args, **kwargs):
        self.fail("load_state must not execute external commands")

    def read_json(self, value):
        self.path.write_text(json.dumps(value), encoding="utf-8")
        return load_state(self.ctx, "session")

    def test_objects_preserve_their_contents(self):
        for value in ({}, {"branch": "example", "nested": {"items": [1, None]}}):
            with self.subTest(value=value):
                self.assertEqual(self.read_json(value), value)

    def test_lists_return_empty_state(self):
        for value in ([], [{"branch": "example"}]):
            with self.subTest(value=value):
                self.assertEqual(self.read_json(value), {})

    def test_null_returns_empty_state(self):
        self.assertEqual(self.read_json(None), {})

    def test_strings_return_empty_state(self):
        for value in ("", "state"):
            with self.subTest(value=value):
                self.assertEqual(self.read_json(value), {})

    def test_numbers_return_empty_state(self):
        for value in (0, 42, 1.5):
            with self.subTest(value=value):
                self.assertEqual(self.read_json(value), {})

    def test_booleans_return_empty_state(self):
        for value in (False, True):
            with self.subTest(value=value):
                self.assertEqual(self.read_json(value), {})

    def test_missing_file_returns_empty_state(self):
        self.assertEqual(load_state(self.ctx, "session"), {})

    def test_invalid_json_returns_empty_state(self):
        for text in ("", "{"):
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                self.assertEqual(load_state(self.ctx, "session"), {})
