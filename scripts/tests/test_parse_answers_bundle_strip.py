#!/usr/bin/env python3
"""
Regression tests for strip_until_first_brace + optional smoke stats on a bundle answers CSV.

Run from repo root:
  python scripts/tests/test_parse_answers_bundle_strip.py
  python scripts/tests/test_parse_answers_bundle_strip.py /workspace-SR004.nfs2/acherepanov/mmred_project/mmred/data_cache/Qwen/Qwen3-4B/bun_seq16_k16_nq100_seed12345_t1_p1_spend_alone_at_step_crowded_room/qa_pairs_answers_bundle.csv

After smoke metrics, prints a 4-column summary (legacy, current, and Final_Answer from
legacy parse — with sample rows), then a sample table with Final vs Fin_leg per row.
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

import pandas as pd
from json_repair import repair_json

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "scripts" / "utils"))

from mmred.const import NOBODY  # noqa: E402

from parse_answers import (  # noqa: E402
    parse_predicted_answer,
    strip_until_first_brace,
    _prepare_hits_dataframe,
    validate_and_compare,
    numeric_pattern,
    answer_pattern,
)


def _strip_until_first_brace_old(string: str) -> str:
    """Previous implementation (bug: used pre-slice index in find after slice)."""
    think_end = string.find("</think>")
    if think_end == -1:
        return string
    string = string[think_end:]
    string = string.replace("</answer>", "").replace("<answer>", "").strip()
    brace_start = string.find("{", think_end)
    if brace_start == -1:
        return string
    return string[brace_start:]


def parse_predicted_answer_old(predicted_answer):
    """
    Previous parse_predicted_answer: substring 'no' -> Nobody set, list rules,
    same regex fallbacks as production.
    """
    try:
        parsed_answer = json.loads(repair_json(predicted_answer))
        assert isinstance(parsed_answer, dict)
        parsed_answer = parsed_answer.get("answer", "None")
        if "no" in str(parsed_answer).lower():
            parsed_answer = {"Nobody"}
        if isinstance(parsed_answer, list):
            if len(parsed_answer) == 1:
                parsed_answer = {parsed_answer[0]}
            else:
                parsed_answer = {"Nobody"}
        return parsed_answer
    except Exception:
        if isinstance(predicted_answer, str):
            numeric_match = re.search(numeric_pattern, predicted_answer)
            if numeric_match:
                return int(numeric_match.group(1))
        match = re.search(answer_pattern, predicted_answer, flags=re.IGNORECASE)
        return match[0] if match else "None"


def _prepare_hits_dataframe_legacy(df_answers: pd.DataFrame) -> pd.DataFrame:
    """Same as parse_answers._prepare_hits_dataframe but old strip + old parse_predicted_answer."""
    if "Answer" in df_answers.columns:
        df_answers = df_answers.rename(
            columns={
                "Answer": "answer",
                "N_steps": "seq_len",
                "Type": "qtype",
            }
        )
    df_answers = df_answers[
        ~df_answers["Predicted_Answer"].str.lower().str.contains("error", na=True)
    ]
    df_answers = df_answers.copy()
    df_answers["Predicted_Answer"] = df_answers["Predicted_Answer"].apply(
        _strip_until_first_brace_old
    )
    df_answers["Final_Answer"] = [
        parse_predicted_answer_old(x) for x in df_answers["Predicted_Answer"]
    ]
    df_answers["hit"] = df_answers.apply(validate_and_compare, axis=1).astype(int)
    return df_answers


def _json_tail_ok(s: str) -> bool:
    try:
        obj = json.loads(repair_json(s))
        return isinstance(obj, dict) and "answer" in obj
    except Exception:
        return False


class TestStripUntilBrace(unittest.TestCase):
    def test_post_slice_index_bug_regression(self):
        """Long prefix + thinking close: old code used wrong start index for find('{')."""
        prefix = "x" * 150
        raw = prefix + '</think>\n{"answer": "John"}'
        old = _strip_until_first_brace_old(raw)
        new = strip_until_first_brace(raw)
        self.assertNotEqual(old.strip(), '{"answer": "John"}')
        self.assertEqual(new.strip(), '{"answer": "John"}')
        self.assertTrue(_json_tail_ok(new))

    def test_double_thinking_takes_last_close(self):
        raw = (
            "<think>draft</think>"
            "noise"
            "<think>final</think>"
            '\n{"answer": "Mary"}'
        )
        out = strip_until_first_brace(raw)
        self.assertTrue(out.strip().startswith("{"))
        self.assertTrue(_json_tail_ok(out))

    def test_no_thinking_passthrough(self):
        raw = '{"answer": "John"}'
        self.assertEqual(strip_until_first_brace(raw), raw)

    def test_untagged_long_cot_json_tail(self):
        """No [/THINK] or redacted close: take last {\"answer\" ... tail (Ministral-style)."""
        raw = (
            "Long markdown CoT.\n**Final Answer:**\n"
            '{"answer": "Garden"}'
        )
        out = strip_until_first_brace(raw)
        self.assertEqual(out.strip(), '{"answer": "Garden"}')
        self.assertTrue(_json_tail_ok(out))

    def test_gemma4_channel_close_strips_to_json(self):
        """Gemma-4 native reasoning ends with <channel|> before final JSON."""
        prefix = "preamble " * 20
        raw = (
            prefix
            + "<|channel>thought\nscratch reasoning here<channel|>"
            '\n{"answer": "Kitchen"}'
        )
        out = strip_until_first_brace(raw)
        self.assertEqual(out.strip(), '{"answer": "Kitchen"}')
        self.assertTrue(_json_tail_ok(out))

    def test_parse_predicted_answer_after_strip(self):
        raw = (
            "<think>time 2 step 4</think>"
            '{"answer": "Daniel"}'
        )
        stripped = strip_until_first_brace(raw)
        ans = parse_predicted_answer(stripped)
        self.assertEqual(ans, "Daniel")

    def test_parse_nobody_explicit_string(self):
        self.assertEqual(parse_predicted_answer('{"answer": "Nobody"}'), NOBODY)

    def test_parse_nobody_null(self):
        self.assertEqual(parse_predicted_answer('{"answer": null}'), NOBODY)

    def test_parse_noah_not_nobody(self):
        """Substring 'no' must not map names like Noah to Nobody."""
        self.assertEqual(parse_predicted_answer('{"answer": "Noah"}'), "Noah")

    def test_parse_nobody_empty_list(self):
        self.assertEqual(parse_predicted_answer('{"answer": []}'), NOBODY)


DEFAULT_CSV = (
    _project_root
    / "data_cache/Qwen/Qwen3-14B/bun_seq16_k16_nq100_seed12345_t1_p1_spend_alone_at_step_crowded_room"
    / "qa_pairs_answers_bundle.csv"
)


def _fmt_cell(x: object, width: int) -> str:
    s = repr(x) if isinstance(x, (set, frozenset)) else str(x)
    s = s.replace("\n", " ").replace("\r", "")
    if len(s) > width:
        s = s[: max(0, width - 3)] + "..."
    return s.ljust(width)


def print_inspection_table(
    prep: pd.DataFrame,
    *,
    prep_legacy: pd.DataFrame | None = None,
    max_rows: int = 15,
    question_width: int = 56,
) -> None:
    """Pretty-print a mix of miss/hit rows (same style as manual notebook checks)."""
    miss = prep[prep["hit"] == 0]
    hit = prep[prep["hit"] == 1]
    want_miss = (max_rows + 1) // 2
    n_miss = min(want_miss, len(miss))
    n_hit = min(max_rows - n_miss, len(hit))
    chunks: list[pd.DataFrame] = []
    if n_miss:
        chunks.append(
            miss.sample(n_miss, random_state=0) if len(miss) > n_miss else miss.head(n_miss)
        )
    if n_hit:
        chunks.append(
            hit.sample(n_hit, random_state=1) if len(hit) > n_hit else hit.head(n_hit)
        )
    if not chunks:
        tab = prep.head(max_rows)
    else:
        tab = pd.concat(chunks, axis=0)

    cols_show = ["hit", "qid", "answer", "Final_Answer", "question"]
    for c in cols_show:
        if c not in tab.columns:
            raise KeyError(f"expected column {c!r} in prepared frame, got {list(tab.columns)}")

    w_hit, w_qid, w_ans, w_pred, w_leg = 4, 8, 12, 12, 12
    has_legacy = prep_legacy is not None
    if has_legacy:
        if not prep.index.equals(prep_legacy.index):
            prep_legacy = prep_legacy.reindex(prep.index)

    title = "Sample rows (hit, qid, answer, Final_Answer"
    if has_legacy:
        title += ", Final legacy"
    title += f", question) — {len(tab)} lines:"
    print(f"\n{title}")
    if has_legacy:
        header = (
            f"{'#':>4}  {'hit':>4}  {'qid':<{w_qid}}  "
            f"{'answer':<{w_ans}}  {'Final':<{w_pred}}  {'Fin_leg':<{w_leg}}  question"
        )
    else:
        header = (
            f"{'#':>4}  {'hit':>4}  {'qid':<{w_qid}}  "
            f"{'answer':<{w_ans}}  {'Final_Answer':<{w_pred}}  question"
        )
    print(header)
    sep_w = 4 + 4 + w_qid + w_ans + w_pred + question_width + 24 + (w_leg if has_legacy else 0)
    print("-" * min(160, sep_w))
    for i, (idx, row) in enumerate(tab.iterrows()):
        q = _fmt_cell(row["question"], question_width)
        fin_leg = ""
        if has_legacy:
            fin_leg = _fmt_cell(prep_legacy.loc[idx, "Final_Answer"], w_leg)
        if has_legacy:
            line = (
                f"{i:4d}  {int(row['hit']):{w_hit}d}  "
                f"{_fmt_cell(row['qid'], w_qid)}  {_fmt_cell(row['answer'], w_ans)}  "
                f"{_fmt_cell(row['Final_Answer'], w_pred)}  {fin_leg}  {q}"
            )
        else:
            line = (
                f"{i:4d}  {int(row['hit']):{w_hit}d}  "
                f"{_fmt_cell(row['qid'], w_qid)}  {_fmt_cell(row['answer'], w_ans)}  "
                f"{_fmt_cell(row['Final_Answer'], w_pred)}  {q}"
            )
        print(line)


def print_comparison_summary(
    *,
    prep_n: int,
    json_sample_n: int,
    json_old_ok: int,
    json_new_ok: int,
    prep_legacy: pd.DataFrame,
    prep_current: pd.DataFrame,
    sample_parsed_rows: int = 3,
) -> None:
    """Side-by-side metrics: legacy strip+parse vs current parse_answers + legacy parsed column."""
    m_leg = float(prep_legacy["hit"].mean())
    m_cur = float(prep_current["hit"].mean())
    int_leg = int(prep_legacy["Final_Answer"].map(lambda x: isinstance(x, int)).sum())
    int_cur = int(prep_current["Final_Answer"].map(lambda x: isinstance(x, int)).sum())
    set_leg = int(prep_legacy["Final_Answer"].map(lambda x: isinstance(x, set)).sum())
    set_cur = int(prep_current["Final_Answer"].map(lambda x: isinstance(x, set)).sum())
    rows_leg = len(prep_legacy)
    rows_cur = len(prep_current)

    label_l = "legacy (old strip + old parse)"
    label_c = "current (parse_answers.py)"
    label_p = "Final_Answer (legacy parse)"
    w_l = max(len(label_l), 22)
    w_c = max(len(label_c), 22)
    w_p = max(len(label_p), 22)

    dash = "—"

    def row4(metric: str, a: str, b: str, p: str) -> str:
        return f"{metric:<38}  {a:>{w_l}}  {b:>{w_c}}  {p:>{w_p}}"

    total_w = 38 + 2 + w_l + 2 + w_c + 2 + w_p
    bar = "=" * total_w
    print(bar)
    print("Summary: legacy vs current (same CSV rows)")
    print(bar)
    print(row4("", label_l, label_c, label_p))
    print("-" * total_w)
    print(
        row4(
            f"JSON dict OK after strip ({json_sample_n})",
            f"{json_old_ok}/{json_sample_n}",
            f"{json_new_ok}/{json_sample_n}",
            dash,
        )
    )
    print(row4(f"Rows scored (first {prep_n})", str(rows_leg), str(rows_cur), dash))
    print(row4("Mean row hit", f"{m_leg:.4f}", f"{m_cur:.4f}", dash))
    print(row4("Final_Answer int count", str(int_leg), str(int_cur), dash))
    print(row4("Final_Answer set count", str(set_leg), str(set_cur), dash))
    print("-" * total_w)
    n_show = min(sample_parsed_rows, len(prep_legacy))
    for i in range(n_show):
        qid = prep_legacy["qid"].iloc[i] if "qid" in prep_legacy.columns else i
        fl = _fmt_cell(prep_legacy["Final_Answer"].iloc[i], w_p).rstrip()
        print(
            row4(
                f"Sample row {i} (qid={qid})",
                f"hit={int(prep_legacy['hit'].iloc[i])}",
                f"hit={int(prep_current['hit'].iloc[i])}",
                fl,
            )
        )
    print(f"{bar}\n")


def run_csv_smoke(csv_path: Path, sample_size: int = 250, seed: int = 0) -> None:
    if not csv_path.is_file():
        print(f"Skip dataset smoke: missing file {csv_path}")
        return
    df = pd.read_csv(csv_path)
    df = df[~df["Predicted_Answer"].str.lower().str.contains("error", na=True)]
    n = min(sample_size, len(df))
    sub = df.sample(n=n, random_state=seed) if len(df) > n else df

    def count_json_ok(strip_fn):
        ok = 0
        for s in sub["Predicted_Answer"].astype(str):
            t = strip_fn(s)
            if _json_tail_ok(t):
                ok += 1
        return ok

    old_ok = count_json_ok(_strip_until_first_brace_old)
    new_ok = count_json_ok(strip_until_first_brace)
    print(f"\nDataset smoke ({csv_path.name}): sampled {n} rows (no error rows)")

    prep_n = min(500, len(df))
    chunk = df.head(prep_n).copy()
    prep_legacy = _prepare_hits_dataframe_legacy(chunk.copy())
    prep_current = _prepare_hits_dataframe(chunk.copy())
    print_comparison_summary(
        prep_n=prep_n,
        json_sample_n=n,
        json_old_ok=old_ok,
        json_new_ok=new_ok,
        prep_legacy=prep_legacy,
        prep_current=prep_current,
    )
    print("Sample rows: Final + Fin_leg (legacy strip+parse).")
    print_inspection_table(prep_current, prep_legacy=prep_legacy, max_rows=15)


if __name__ == "__main__":
    csv_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestStripUntilBrace)
    r = unittest.TextTestRunner(verbosity=2).run(suite)
    run_csv_smoke(csv_arg)
    sys.exit(0 if r.wasSuccessful() else 1)
