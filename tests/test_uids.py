"""Miner availability and random UID sampling."""

import numpy as np
import pytest

from neurons.utils.misc import as_int
from neurons.utils.uids import check_uid_availability, get_random_uids


class _Axon:
    def __init__(self, serving):
        self.is_serving = serving


class _Metagraph:
    def __init__(self, n, serving, permit, stake):
        self.n = n
        self.axons = [_Axon(flag) for flag in serving]
        self.validator_permit = permit
        self.S = stake


class _Neuron:
    def __init__(self, metagraph, limit=4096):
        self.metagraph = metagraph
        self.config = type("Cfg", (), {})()
        self.config.neuron = type("NeuronCfg", (), {})()
        self.config.neuron.vpermit_tao_limit = limit


def _network():
    # uid 1 is offline. uid 3 is a validator over the stake limit.
    return _Metagraph(
        n=5,
        serving=[True, False, True, True, True],
        permit=[False, False, True, True, False],
        stake=[0, 0, 100, 5000, 1],
    )


def test_as_int_accepts_python_and_tensor_scalars():
    class _Scalar:
        def item(self):
            return 7

    assert as_int(3) == 3
    assert as_int(np.int64(4)) == 4
    assert as_int(_Scalar()) == 7


def test_uid_is_available_only_when_serving_and_under_the_stake_limit():
    meta = _network()
    assert check_uid_availability(meta, 0, 4096) is True
    assert check_uid_availability(meta, 1, 4096) is False
    assert check_uid_availability(meta, 2, 4096) is True
    assert check_uid_availability(meta, 3, 4096) is False
    # Stake equal to the limit is still allowed; only a greater stake is not.
    assert check_uid_availability(meta, 3, 5000) is True
    # A high stake does not matter when the uid has no validator permit.
    assert check_uid_availability(meta, 4, 0) is True


def test_get_random_uids_caps_at_the_available_set_and_accepts_plain_int_n():
    neuron = _Neuron(_network())
    uids = get_random_uids(neuron, k=10)
    assert uids.dtype == np.int64
    assert sorted(uids.tolist()) == [0, 2, 4]


def test_get_random_uids_reads_tensor_style_network_size():
    meta = _network()

    class _Count:
        def item(self):
            return 5

    meta.n = _Count()
    uids = get_random_uids(_Neuron(meta), k=1)
    assert uids.tolist()[0] in {0, 2, 4}


def test_get_random_uids_skips_excluded_until_it_must_fill():
    neuron = _Neuron(_network())
    preferred = get_random_uids(neuron, k=2, exclude=[0])
    assert sorted(preferred.tolist()) == [2, 4]

    # Asking for every available uid forces the excluded ones back in.
    filled = get_random_uids(neuron, k=3, exclude=[0, 2, 4])
    assert sorted(filled.tolist()) == [0, 2, 4]


def test_get_random_uids_rejects_bad_k_and_returns_empty_for_zero():
    neuron = _Neuron(_network())
    empty = get_random_uids(neuron, k=0)
    assert empty.tolist() == []
    assert empty.dtype == np.int64
    with pytest.raises(ValueError):
        get_random_uids(neuron, k=-1)
    with pytest.raises(TypeError):
        get_random_uids(neuron, k=1.5)


def test_get_random_uids_returns_empty_when_nobody_is_serving():
    meta = _Metagraph(
        n=2,
        serving=[False, False],
        permit=[False, False],
        stake=[0, 0],
    )
    uids = get_random_uids(_Neuron(meta), k=4)
    assert uids.tolist() == []
