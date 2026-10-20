"""Miner request blacklist and priority decisions."""

import asyncio
from types import SimpleNamespace

from neurons.miner import Miner
from neurons.utils.access import decide_blacklist, request_priority

HOTKEYS = ["alice", "bob", "carol"]
PERMIT = [True, False, True]
STAKE = [10.0, 2.5, 0.0]


def test_blacklist_rejects_anonymous_and_unknown_callers():
    assert decide_blacklist(
        None,
        HOTKEYS,
        PERMIT,
        allow_non_registered=False,
        force_validator_permit=False,
    ) == (True, "Missing dendrite or hotkey")
    assert decide_blacklist(
        "dave",
        HOTKEYS,
        PERMIT,
        allow_non_registered=False,
        force_validator_permit=False,
    ) == (True, "Unrecognized hotkey")


def test_blacklist_can_allow_unregistered_callers_unless_permit_is_required():
    assert decide_blacklist(
        "dave",
        HOTKEYS,
        PERMIT,
        allow_non_registered=True,
        force_validator_permit=False,
    ) == (False, "Hotkey recognized!")
    assert decide_blacklist(
        "dave",
        HOTKEYS,
        PERMIT,
        allow_non_registered=True,
        force_validator_permit=True,
    ) == (True, "Non-validator hotkey")


def test_blacklist_enforces_validator_permit_for_registered_hotkeys():
    assert decide_blacklist(
        "bob",
        HOTKEYS,
        PERMIT,
        allow_non_registered=False,
        force_validator_permit=True,
    ) == (True, "Non-validator hotkey")
    assert decide_blacklist(
        "alice",
        HOTKEYS,
        PERMIT,
        allow_non_registered=False,
        force_validator_permit=True,
    ) == (False, "Hotkey recognized!")


def test_priority_is_stake_and_zero_for_unknown_callers():
    assert request_priority("bob", HOTKEYS, STAKE) == 2.5
    assert request_priority("dave", HOTKEYS, STAKE) == 0.0
    assert request_priority(None, HOTKEYS, STAKE) == 0.0


def _miner():
    miner = Miner.__new__(Miner)
    miner.metagraph = SimpleNamespace(
        hotkeys=HOTKEYS, validator_permit=PERMIT, S=STAKE
    )
    miner.config = SimpleNamespace(
        blacklist=SimpleNamespace(
            allow_non_registered=False,
            force_validator_permit=False,
        )
    )
    return miner


def test_miner_forward_returns_double_the_input():
    miner = Miner.__new__(Miner)
    synapse = SimpleNamespace(dummy_input=5, dummy_output=None)
    result = asyncio.run(miner.forward(synapse))
    assert result.dummy_output == 10


def test_miner_blacklist_does_not_crash_on_an_unknown_hotkey():
    miner = _miner()
    synapse = SimpleNamespace(dendrite=SimpleNamespace(hotkey="dave"))
    reject, reason = asyncio.run(miner.blacklist(synapse))
    assert reject is True
    assert reason == "Unrecognized hotkey"


def test_miner_priority_does_not_crash_on_an_unknown_hotkey():
    miner = _miner()
    synapse = SimpleNamespace(dendrite=SimpleNamespace(hotkey="dave"))
    assert asyncio.run(miner.priority(synapse)) == 0.0
    missing = SimpleNamespace(dendrite=None)
    assert asyncio.run(miner.priority(missing)) == 0.0
