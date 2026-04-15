#!/usr/bin/env python3
"""Analyze MMReD benchmark results with reasoning token metrics.

Usage:
    python scripts/analyze_results.py [--results_dir data/mera_results]
"""

import json
import glob
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mera_integration', 'benchmark_tasks', 'mmred'))
from utils import extract_answer, normalize_answer


TASK_MAP = {
    'sa_c': 'DC-SA-C',
    'sr_i': 'DC-SR-I',
    'cc_i': 'DC-CC-I',
    'ws_r': 'DC-WS-R',
    'whs_c': 'DC-WHS-C',
}
TASK_ORDER = ['sa_c', 'sr_i', 'cc_i', 'ws_r', 'whs_c']
SEQ_LENS = [32, 64, 128]


def count_reasoning_tokens(text: str) -> int:
    """Estimate reasoning tokens in model output."""
    reasoning_text = ""

    # Gemma-4 format: <|channel>thought...reasoning...<channel|>answer
    gemma_match = re.search(r"<\|channel>thought(.*?)<channel\|>", text, flags=re.DOTALL)
    if gemma_match:
        reasoning_text = gemma_match.group(1)

    # Qwen/generic format: <think>...</think>
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if think_match:
        reasoning_text = think_match.group(1)

    # If no explicit markers but long output, count everything before last line as reasoning
    if not reasoning_text and len(text) > 200:
        lines = text.strip().split('\n')
        if len(lines) > 3:
            reasoning_text = '\n'.join(lines[:-1])

    # Estimate tokens: ~4 chars per token for mixed Russian/English
    return len(reasoning_text) // 4


def analyze_model(result_dir: str) -> dict:
    """Analyze all results for a model."""
    sample_files = glob.glob(os.path.join(result_dir, '*/samples_*.jsonl'))
    if not sample_files:
        sample_files = glob.glob(os.path.join(result_dir, '*/*/samples_*.jsonl'))
    if not sample_files:
        return {}

    metrics = {}
    for f in sorted(sample_files):
        task = os.path.basename(f).split('_2026')[0].replace('samples_mmred_dc_', '')
        parts = task.rsplit('_', 1)
        if len(parts) != 2:
            continue
        ttype, slen = parts[0], int(parts[1])

        correct = total = 0
        total_reason_tokens = 0
        total_output_chars = 0
        has_reasoning = 0

        for line in open(f):
            d = json.loads(line)
            resp = d.get('resps', [[]])[0][0] if d.get('resps') else ''
            target = str(d.get('target', ''))
            meta = d.get('doc', {}).get('meta', {})
            atype = meta.get('categories', {}).get('atype', meta.get('atype', 'person'))

            pred = extract_answer(resp, atype)
            gold = normalize_answer(target, atype)
            if pred.lower() == gold.lower():
                correct += 1
            total += 1

            reason_tok = count_reasoning_tokens(resp)
            total_reason_tokens += reason_tok
            total_output_chars += len(resp)
            if reason_tok > 10:
                has_reasoning += 1

        em = correct / total if total else 0
        avg_reason = total_reason_tokens / total if total else 0
        avg_output = total_output_chars / (4 * total) if total else 0  # est tokens

        metrics[(ttype, slen)] = {
            'em': em,
            'n': total,
            'avg_reasoning_tokens': avg_reason,
            'avg_output_tokens': avg_output,
            'pct_reasoning': has_reasoning / total if total else 0,
        }

    return metrics


def print_table(all_results: dict):
    """Print comparison table."""
    # Header
    print(f"\n{'Model':<35}", end='')
    for t in TASK_ORDER:
        for sl in SEQ_LENS:
            print(f" {TASK_MAP[t][:2]}{sl:>3}", end='')
    print(f" {'AVG':>5} {'H-AGG':>5} {'R.Tok':>7}")
    print('-' * 120)

    for name, metrics in sorted(all_results.items(), key=lambda x: -sum(m['em'] for m in x[1].values()) / max(len(x[1]), 1)):
        print(f"{name:<35}", end='')
        all_em = []
        all_reason = []
        for t in TASK_ORDER:
            for sl in SEQ_LENS:
                m = metrics.get((t, sl))
                if m:
                    print(f" {m['em']:>5.0%}", end='')
                    all_em.append(m['em'])
                    all_reason.append(m['avg_reasoning_tokens'])
                else:
                    print(f" {'--':>5}", end='')
        avg_em = sum(all_em) / len(all_em) if all_em else 0
        avg_reason = sum(all_reason) / len(all_reason) if all_reason else 0

        # Harmonic mean of per-task length-weighted scores
        eps = 1e-6
        per_task_weighted = {}
        for t in TASK_ORDER:
            w_sum = w_total = 0
            for sl in SEQ_LENS:
                m = metrics.get((t, sl))
                if m:
                    w = 2 ** (sl / 32)
                    w_sum += m['em'] * w
                    w_total += w
            if w_total > 0:
                per_task_weighted[t] = w_sum / w_total
        if per_task_weighted:
            n = len(per_task_weighted)
            h_mean = n / sum(1.0 / (v + eps) for v in per_task_weighted.values())
        else:
            h_mean = 0.0

        print(f" {avg_em:>5.1%} {h_mean:>5.1%} {avg_reason:>7.0f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='data/mera_results')
    args = parser.parse_args()

    all_results = {}
    for d in sorted(os.listdir(args.results_dir)):
        full_path = os.path.join(args.results_dir, d)
        if not os.path.isdir(full_path):
            continue
        metrics = analyze_model(full_path)
        if metrics:
            all_results[d] = metrics

    print_table(all_results)

    # Summary
    print(f"\n{'Model':<35} {'EM':>6} {'Reasoning%':>10} {'AvgReasonTok':>13} {'AvgOutTok':>10}")
    print('-' * 80)
    for name, metrics in sorted(all_results.items(), key=lambda x: -sum(m['em'] for m in x[1].values()) / max(len(x[1]), 1)):
        all_em = [m['em'] for m in metrics.values()]
        all_reason = [m['avg_reasoning_tokens'] for m in metrics.values()]
        all_pct = [m['pct_reasoning'] for m in metrics.values()]
        all_out = [m['avg_output_tokens'] for m in metrics.values()]
        print(f"{name:<35} {sum(all_em)/len(all_em):>5.1%} {sum(all_pct)/len(all_pct):>9.0%} {sum(all_reason)/len(all_reason):>13.0f} {sum(all_out)/len(all_out):>10.0f}")


if __name__ == '__main__':
    main()
