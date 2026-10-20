"""Scoring rules shared by the miner and the validator."""

import numpy as np
import pytest

from neurons.incentive import expected_dummy_output
from neurons.validator.reward import apply_moving_average, get_rewards, reward


def test_expected_output_is_double_the_query():
    assert expected_dummy_output(0) == 0
    assert expected_dummy_output(4) == 8
    assert expected_dummy_output(-3) == -6


def test_reward_is_one_only_for_an_exact_double():
    assert reward(4, 8) == 1.0
    assert reward(4, 7) == 0.0
    assert reward(4, 9) == 0.0
    assert reward(0, 0) == 1.0


def test_reward_treats_missing_and_non_numeric_responses_as_zero():
    assert reward(4, None) == 0.0
    assert reward(None, None) == 0.0
    assert reward(4, "8") == 0.0


def test_get_rewards_scores_each_response_independently():
    scores = get_rewards(None, query=3, responses=[6, 1, None, 0])
    assert scores.dtype == np.float32
    assert scores.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_get_rewards_accepts_no_responses():
    assert get_rewards(None, query=1, responses=None).tolist() == []
    assert get_rewards(None, query=1, responses=[]).tolist() == []


def test_moving_average_mixes_new_rewards_into_scores():
    scores = np.zeros(3, dtype=np.float32)
    updated, status = apply_moving_average(scores, [1.0, 0.0], [1, 2], 0.1)
    assert status == "ok"
    assert updated.tolist() == pytest.approx([0.0, 0.1, 0.0])
    # The previous score vector is left untouched.
    assert scores.tolist() == [0.0, 0.0, 0.0]


def test_moving_average_keeps_history_for_unscored_uids():
    scores = np.ones(3, dtype=np.float32)
    updated, status = apply_moving_average(scores, [1.0], [0], 0.25)
    assert status == "ok"
    assert updated[0] == pytest.approx(1.0)
    assert updated[1] == pytest.approx(0.75)
    assert updated[2] == pytest.approx(0.75)


def test_moving_average_replaces_nan_rewards_with_zero():
    scores = np.ones(2, dtype=np.float32)
    updated, status = apply_moving_average(
        scores, [float("nan")], [1], alpha=0.5
    )
    assert status == "nan"
    assert np.isfinite(updated).all()
    # Queried and unqueried UIDs both decay when the new reward is NaN,
    # because a NaN is treated as 0 rather than left to poison the score.
    assert updated.tolist() == pytest.approx([0.5, 0.5])


def test_moving_average_rejects_mismatched_or_out_of_range_uids():
    scores = np.zeros(2, dtype=np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        apply_moving_average(scores, [1.0, 1.0], [0], 0.1)
    with pytest.raises(ValueError, match="out of range"):
        apply_moving_average(scores, [1.0], [-1], 0.1)
    with pytest.raises(ValueError, match="out of range"):
        apply_moving_average(scores, [1.0], [2], 0.1)


def test_moving_average_leaves_scores_unchanged_when_nothing_arrived():
    scores = np.array([0.4, 0.6], dtype=np.float32)
    updated, status = apply_moving_average(scores, [], [], 0.2)
    assert status == "empty"
    assert updated.tolist() == pytest.approx([0.4, 0.6])
