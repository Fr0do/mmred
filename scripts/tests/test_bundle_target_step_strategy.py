#!/usr/bin/env python3
"""Tests for sequence-bundle target time-step placement strategies."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from mmred.qgen.bundles import generate_bundle_dataset, select_target_frames  # noqa: E402


class TestTargetStepStrategy(unittest.TestCase):
    def test_prefix_frames(self) -> None:
        rng = random.Random(0)
        self.assertEqual(select_target_frames(8, 4, "prefix", rng), [0, 1, 2, 3])

    def test_random_frames_are_distinct_and_sized(self) -> None:
        rng = random.Random(123)
        frames = select_target_frames(16, 5, "random", rng)
        self.assertEqual(len(frames), 5)
        self.assertEqual(len(set(frames)), 5)
        self.assertEqual(frames, sorted(frames))
        self.assertTrue(all(0 <= f < 16 for f in frames))

    def test_random_full_k_covers_all_steps(self) -> None:
        rng = random.Random(123)
        self.assertEqual(select_target_frames(6, 6, "random", rng), list(range(6)))

    def test_unknown_strategy_raises(self) -> None:
        with self.assertRaises(ValueError):
            select_target_frames(6, 2, "suffix", random.Random(0))

    def test_generated_bundle_has_k_targets_per_episode(self) -> None:
        rows = generate_bundle_dataset(
            n_episodes=2,
            seq_len=6,
            k_target=3,
            bundle_size=6,
            target_question_type="spend_alone_at_step",
            question_types=["spend_alone_at_step", "crowded_room"],
            seed=12345,
            target_step_strategy="random",
        )
        by_episode: dict[str, int] = {}
        for row in rows:
            if row["is_target"]:
                by_episode[row["episode_id"]] = by_episode.get(row["episode_id"], 0) + 1
        self.assertEqual(by_episode, {"00000": 3, "00001": 3})


if __name__ == "__main__":
    unittest.main()
