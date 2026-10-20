"""Weight normalization and the integer format emitted on chain."""

import numpy as np
import pytest

from neurons.base.utils.weight_utils import (
    U16_MAX,
    convert_weights_and_uids_for_emit,
    normalize_max_weight,
    process_weights_for_netuid,
)


def test_normalize_evenly_when_there_is_no_signal():
    out = normalize_max_weight(np.zeros(4), limit=0.1)
    assert out.tolist() == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_normalize_evenly_when_the_limit_cannot_express_a_peak():
    # Four neurons and a 0.2 cap cannot sum to 1 unless every weight is 0.25.
    out = normalize_max_weight(np.array([5.0, 1.0, 0.0, 0.0]), limit=0.2)
    assert out.tolist() == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_normalize_keeps_proportions_inside_the_limit():
    out = normalize_max_weight(np.array([1.0, 3.0]), limit=0.8)
    assert out.tolist() == pytest.approx([0.25, 0.75])


def test_normalize_caps_a_dominant_weight():
    out = normalize_max_weight(
        np.array([100.0, 1.0, 1.0, 1.0, 1.0]), limit=0.3
    )
    assert out.sum() == pytest.approx(1.0)
    assert out.max() <= 0.3 + 1e-6


def test_convert_scales_the_max_weight_to_uint16_and_drops_zeros():
    uids, vals = convert_weights_and_uids_for_emit(
        np.array([1, 2, 3]), np.array([0.0, 0.5, 1.0])
    )
    assert uids == [2, 3]
    assert vals == [round(0.5 * U16_MAX), U16_MAX]
    assert all(isinstance(uid, int) for uid in uids)
    assert all(isinstance(val, int) for val in vals)


def test_convert_rejects_bad_inputs_and_emits_nothing_for_zeros():
    with pytest.raises(ValueError, match="negative"):
        convert_weights_and_uids_for_emit(np.array([0]), np.array([-0.1]))
    with pytest.raises(ValueError, match="uid is negative"):
        convert_weights_and_uids_for_emit(np.array([-1]), np.array([1.0]))
    with pytest.raises(ValueError, match="same length"):
        convert_weights_and_uids_for_emit(np.array([0, 1]), np.array([1.0]))
    with pytest.raises(ValueError, match="NaN"):
        convert_weights_and_uids_for_emit(
            np.array([0]), np.array([float("nan")])
        )
    assert convert_weights_and_uids_for_emit(
        np.array([0, 1]), np.array([0.0, 0.0])
    ) == ([], [])
    assert convert_weights_and_uids_for_emit(np.array([]), np.array([])) == (
        [],
        [],
    )


class _Chain:
    def __init__(self, min_allowed, max_limit):
        self._min_allowed = min_allowed
        self._max_limit = max_limit

    def min_allowed_weights(self, netuid):
        return self._min_allowed

    def max_weight_limit(self, netuid):
        return self._max_limit


class _Meta:
    def __init__(self, n):
        self.n = n


def test_process_weights_returns_uniform_when_nothing_was_scored():
    uids, weights = process_weights_for_netuid(
        uids=np.arange(4),
        weights=np.zeros(4, dtype=np.float32),
        netuid=1,
        subtensor=_Chain(min_allowed=1, max_limit=1.0),
        metagraph=_Meta(4),
    )
    assert uids.tolist() == [0, 1, 2, 3]
    assert weights.tolist() == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_process_weights_keeps_the_scored_uid_largest_when_filling_minimums():
    uids, weights = process_weights_for_netuid(
        uids=np.arange(4),
        weights=np.array([0, 0, 1, 0], dtype=np.float32),
        netuid=1,
        subtensor=_Chain(min_allowed=4, max_limit=1.0),
        metagraph=_Meta(n=4),
    )
    assert weights.sum() == pytest.approx(1.0)
    assert int(np.argmax(weights)) == 2


def test_process_weights_accepts_a_plain_int_network_size():
    uids, weights = process_weights_for_netuid(
        uids=np.array([0, 1]),
        weights=np.array([1.0, 1.0], dtype=np.float32),
        netuid=1,
        subtensor=_Chain(min_allowed=1, max_limit=1.0),
        metagraph=_Meta(n=2),
    )
    assert uids.tolist() == [0, 1]
    assert weights.sum() == pytest.approx(1.0)
    assert weights.max() <= 1.0 + 1e-6
