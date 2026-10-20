"""Validator checkpoint files."""

import os

import numpy as np
import pytest

from neurons.base.state import (
    fit_scores_to_network,
    load_validator_state,
    save_validator_state,
)


def test_state_round_trip_and_missing_file(tmp_path):
    scores = np.array([0.25, 0.75], dtype=np.float32)
    hotkeys = ["5short", "5GabcDEFGHJKLMNopqrstuvwxyz"]
    path = save_validator_state(str(tmp_path), step=3, scores=scores, hotkeys=hotkeys)
    assert os.path.isfile(path)

    loaded = load_validator_state(str(tmp_path))
    assert loaded["step"] == 3
    assert loaded["scores"].tolist() == pytest.approx(scores.tolist())
    assert loaded["hotkeys"] == hotkeys
    assert load_validator_state(str(tmp_path / "absent")) is None


def test_fit_scores_grows_and_shrinks_with_the_metagraph():
    grown = fit_scores_to_network(np.array([0.2, 0.4], dtype=np.float32), 4)
    assert grown.tolist() == pytest.approx([0.2, 0.4, 0.0, 0.0])
    shrunk = fit_scores_to_network(np.array([0.2, 0.4, 0.6], dtype=np.float32), 2)
    assert shrunk.tolist() == pytest.approx([0.2, 0.4])
    same = fit_scores_to_network(np.array([1.0], dtype=np.float32), 1)
    assert same.tolist() == pytest.approx([1.0])
    with pytest.raises(ValueError):
        fit_scores_to_network(np.array([1.0]), -1)
