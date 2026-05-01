"""Tests for Gemma-4 channel markers in CSV assembly (openai_server_inference)."""

import unittest
from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from scripts.openai_server_inference import (  # noqa: E402
    _GEMMA4_CHANNEL_END,
    _GEMMA4_CHANNEL_START,
    _join_gemma4_reasoning_for_csv,
    _join_thinking_for_csv,
    _is_gemma4_model,
)


class TestIsGemma4Model(unittest.TestCase):
    def test_positive(self) -> None:
        self.assertTrue(_is_gemma4_model("google/gemma-4-26B-it"))
        self.assertTrue(_is_gemma4_model("Gemma4-9B"))
        self.assertTrue(_is_gemma4_model("foo_gemma_4_bar"))

    def test_negative(self) -> None:
        self.assertFalse(_is_gemma4_model("google/gemma-2-9b-it"))
        self.assertFalse(_is_gemma4_model(""))


class TestJoinGemma4ReasoningForCsv(unittest.TestCase):
    def test_empty_reasoning_falls_back_to_redacted(self) -> None:
        c = '{"answer": "x"}'
        out = _join_gemma4_reasoning_for_csv("", c)
        self.assertEqual(out, _join_thinking_for_csv("", c))

    def test_wraps_reasoning_and_content(self) -> None:
        out = _join_gemma4_reasoning_for_csv("step1 step2", '{"answer": 1}')
        self.assertTrue(out.startswith(_GEMMA4_CHANNEL_START))
        self.assertIn("thought\nstep1 step2", out)
        self.assertIn(_GEMMA4_CHANNEL_END, out)
        self.assertTrue(out.endswith('{"answer": 1}'))

    def test_strips_thought_label_before_wrap(self) -> None:
        out = _join_gemma4_reasoning_for_csv("thought\nonly body", "tail")
        self.assertIn("only body", out)
        self.assertTrue(out.startswith(_GEMMA4_CHANNEL_START + "thought\nonly body"))
        self.assertTrue(out.endswith(_GEMMA4_CHANNEL_END + "tail"))

    def test_passthrough_when_channel_end_present(self) -> None:
        r = ""
        c = (
            "prefix"
            + _GEMMA4_CHANNEL_START
            + "thought\nx"
            + _GEMMA4_CHANNEL_END
            + '{"answer": 2}'
        )
        out = _join_gemma4_reasoning_for_csv(r, c)
        self.assertEqual(out, c)

    def test_passthrough_reasoning_plus_content_with_close(self) -> None:
        r = "a"
        c = "b" + _GEMMA4_CHANNEL_END + "c"
        self.assertEqual(_join_gemma4_reasoning_for_csv(r, c), "a" + "b" + _GEMMA4_CHANNEL_END + "c")


if __name__ == "__main__":
    unittest.main()
