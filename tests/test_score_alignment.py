"""Score alignment when the metagraph hotkey table changes size."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCORES_PATH = (
    Path(__file__).resolve().parents[1]
    / "neurons"
    / "base"
    / "utils"
    / "scores.py"
)


def _load_scores():
    # Import the module by path. ``import neurons`` pulls in bittensor.
    spec = importlib.util.spec_from_file_location(
        "muxnet_score_alignment", _SCORES_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


align_scores_to_hotkeys = _load_scores().align_scores_to_hotkeys


def test_unchanged_hotkeys_keep_scores_and_do_not_mutate_the_input():
    scores = np.array([0.5, 0.25, 0.125], dtype=np.float32)
    original = scores.copy()
    hotkeys = ["hk-a", "hk-b", "hk-c"]

    aligned = align_scores_to_hotkeys(scores, hotkeys, list(hotkeys))

    assert aligned.dtype == np.float32
    assert aligned.tolist() == pytest.approx([0.5, 0.25, 0.125])
    assert np.array_equal(scores, original)


def test_replaced_uid_is_reset_and_the_others_are_kept():
    aligned = align_scores_to_hotkeys(
        np.array([0.5, 0.25, 0.125], dtype=np.float32),
        ["hk-a", "hk-b", "hk-c"],
        ["hk-a", "hk-new", "hk-c"],
    )

    assert aligned.tolist() == pytest.approx([0.5, 0.0, 0.125])


def test_growing_network_appends_zeros_for_new_uids():
    aligned = align_scores_to_hotkeys(
        [1.0, 0.5],
        ["hk-a", "hk-b"],
        ["hk-a", "hk-b", "hk-c"],
    )

    assert aligned.shape == (3,)
    assert aligned.tolist() == pytest.approx([1.0, 0.5, 0.0])


def test_shrinking_network_drops_removed_uids_and_keeps_survivors():
    aligned = align_scores_to_hotkeys(
        np.array([0.5, 0.25, 0.125, 0.75], dtype=np.float32),
        ["hk-a", "hk-b", "hk-c", "hk-d"],
        ["hk-a", "hk-b"],
    )

    assert aligned.shape == (2,)
    assert aligned.tolist() == pytest.approx([0.5, 0.25])
    # The shortened vector can be indexed by the new UID list.
    uids = np.arange(aligned.shape[0])
    nonzero = np.flatnonzero(aligned > 0)
    assert uids[nonzero].tolist() == [0, 1]


def test_shrink_also_resets_a_replaced_survivor():
    aligned = align_scores_to_hotkeys(
        np.array([0.5, 0.25, 0.125, 0.75], dtype=np.float32),
        ["hk-a", "hk-b", "hk-c", "hk-d"],
        ["hk-a", "hk-new"],
    )

    assert aligned.tolist() == pytest.approx([0.5, 0.0])


def test_short_score_vector_does_not_raise_when_hotkeys_are_longer():
    aligned = align_scores_to_hotkeys(
        np.array([0.5], dtype=np.float32),
        ["hk-a", "hk-b", "hk-c"],
        ["hk-a", "hk-other", "hk-c", "hk-d"],
    )

    assert aligned.tolist() == pytest.approx([0.5, 0.0, 0.0, 0.0])


def test_empty_hotkey_table_returns_an_empty_vector():
    aligned = align_scores_to_hotkeys(
        np.array([0.5, 0.25], dtype=np.float32),
        ["hk-a", "hk-b"],
        [],
    )

    assert aligned.shape == (0,)
    assert aligned.dtype == np.float32


def test_rejects_a_matrix_and_a_bare_hotkey_string():
    with pytest.raises(ValueError, match="1-D"):
        align_scores_to_hotkeys(
            np.ones((2, 2), dtype=np.float32),
            ["hk-a", "hk-b"],
            ["hk-a", "hk-b"],
        )
    with pytest.raises(TypeError, match="not a string"):
        align_scores_to_hotkeys([0.5], "hk-a", ["hk-a"])
