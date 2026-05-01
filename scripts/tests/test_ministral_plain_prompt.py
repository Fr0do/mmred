"""Tests for Ministral-style /v1/completions prompt building (chat_template.jinja parity)."""

import os
import unittest
from pathlib import Path

import sys

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from scripts.openai_server_inference import (  # noqa: E402
    MINISTRAL_DEFAULT_SYSTEM_MESSAGE,
    SYSTEM_PROMPT,
    _messages_to_plain_prompt,
    _mistral_completion_to_answer,
)


class TestMinistralPlainPrompt(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("MISTRAL_PLAIN_BOS_TOKEN", None)
        os.environ.pop("MISTRAL_PLAIN_EOS_TOKEN", None)

    def test_plain_prompt_system_user_matches_structure(self) -> None:
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ]
        p = _messages_to_plain_prompt(messages)
        self.assertEqual(
            p,
            "[SYSTEM_PROMPT]Be brief.[/SYSTEM_PROMPT][INST]Hi[/INST]",
        )

    def test_empty_leading_system_dropped_injects_default(self) -> None:
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "x"},
        ]
        p = _messages_to_plain_prompt(messages)
        self.assertTrue(
            p.startswith(
                "[SYSTEM_PROMPT]" + MINISTRAL_DEFAULT_SYSTEM_MESSAGE + "[/SYSTEM_PROMPT]"
            )
        )
        self.assertTrue(p.endswith("[INST]x[/INST]"))

    def test_user_multimodal_two_blocks_sorted_by_type(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Q"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            },
        ]
        p = _messages_to_plain_prompt(messages)
        self.assertIn(MINISTRAL_DEFAULT_SYSTEM_MESSAGE, p)
        self.assertIn("[INST][IMG]Q[/INST]", p)

    def test_assistant_appends_eos_and_tool_calls(self) -> None:
        os.environ["MISTRAL_PLAIN_EOS_TOKEN"] = "</s>"
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "fn", "arguments": '{"a": 1}'}}
                ],
            },
        ]
        p = _messages_to_plain_prompt(messages)
        self.assertIn('[TOOL_CALLS]fn[ARGS]{"a": 1}</s>', p)

    def test_offline_row_shape_thinking_empty_system(self) -> None:
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
        ]
        p = _messages_to_plain_prompt(messages)
        self.assertIn("[SYSTEM_PROMPT]" + MINISTRAL_DEFAULT_SYSTEM_MESSAGE, p)
        self.assertIn("[INST]task[/INST]", p)

    def test_non_thinking_uses_system_prompt(self) -> None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
        ]
        p = _messages_to_plain_prompt(messages)
        self.assertIn("[SYSTEM_PROMPT]" + SYSTEM_PROMPT + "[/SYSTEM_PROMPT]", p)
        self.assertIn("[INST]q[/INST]", p)

    def test_alternation_violation(self) -> None:
        messages = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        with self.assertRaises(ValueError) as ctx:
            _messages_to_plain_prompt(messages)
        self.assertIn("alternate", str(ctx.exception).lower())


class TestMistralCompletionToAnswer(unittest.TestCase):
    def test_thinking_false_passthrough(self) -> None:
        raw = "  plain text  "
        self.assertEqual(_mistral_completion_to_answer(raw, False), "plain text")

    def test_thinking_true_no_think_tag_unchanged(self) -> None:
        raw = "Long CoT\n**Final Answer:**\n{\"answer\": \"John\"}"
        self.assertEqual(_mistral_completion_to_answer(raw, True), raw)

    def test_thinking_true_splits_on_last_think_close(self) -> None:
        raw = "[THINK]first[/THINK]noise[THINK]second[/THINK]\n{\"answer\": \"x\"}"
        out = _mistral_completion_to_answer(raw, True)
        # Content after last [/THINK] is .lstrip()'d (newline removed), matching chat assembly.
        expected = "[THINK]first[/THINK]noise[THINK]second[/THINK]{\"answer\": \"x\"}"
        self.assertEqual(out, expected)
        tail = out.rsplit("[/THINK]", 1)[-1]
        self.assertEqual(tail, '{"answer": "x"}')

    def test_thinking_true_single_think_block(self) -> None:
        raw = "[THINK]draft[/THINK]{\"answer\": 3}"
        out = _mistral_completion_to_answer(raw, True)
        self.assertEqual(out, raw)


if __name__ == "__main__":
    unittest.main()
