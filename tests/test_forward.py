"""Validator forward pass against stand-in axons."""

import asyncio

import numpy as np
import pytest

from neurons.base.validator import BaseValidatorNeuron
from neurons.validator.forward import forward


class _Axon:
    def __init__(self, uid, serving=True):
        self.uid = uid
        self.is_serving = serving


class _Metagraph:
    def __init__(self):
        self.n = 4
        self.axons = [_Axon(uid) for uid in range(4)]
        self.validator_permit = [False, False, False, False]
        self.S = [0, 0, 0, 0]
        self.uids = np.arange(4)


class _Dendrite:
    def __init__(self, good_uids):
        self.good_uids = set(good_uids)
        self.calls = []

    async def __call__(self, axons, synapse, deserialize):
        self.calls.append((list(axons), synapse.dummy_input, deserialize))
        outputs = []
        for axon in axons:
            if axon.uid in self.good_uids:
                outputs.append(synapse.dummy_input * 2)
            else:
                outputs.append(0)
        return outputs


def _validator(good_uids):
    neuron = type("Validator", (), {})()
    neuron.step = 4
    neuron.scores = np.zeros(4, dtype=np.float32)
    neuron.metagraph = _Metagraph()
    neuron.dendrite = _Dendrite(good_uids)
    neuron.config = type("Cfg", (), {})()
    neuron.config.neuron = type("NeuronCfg", (), {})()
    neuron.config.neuron.sample_size = 4
    neuron.config.neuron.vpermit_tao_limit = 4096
    neuron.config.neuron.moving_average_alpha = 0.5
    neuron.update_scores = lambda rewards, uids: BaseValidatorNeuron.update_scores(
        neuron, rewards, uids
    )
    return neuron


def test_forward_rewards_exact_doubles_and_does_not_block(monkeypatch):
    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(
        "neurons.validator.forward.asyncio.sleep", _no_sleep
    )
    neuron = _validator({0, 2})
    asyncio.run(forward(neuron))

    assert neuron.dendrite.calls[0][1] == 4
    assert neuron.dendrite.calls[0][2] is True
    # alpha 0.5, correct miners at the sampled uids get 0.5, others stay 0.
    rewarded = {
        int(uid): float(neuron.scores[uid])
        for uid in range(4)
        if neuron.scores[uid] > 0
    }
    assert set(rewarded) == {0, 2}
    assert rewarded[0] == pytest.approx(0.5)
    assert rewarded[2] == pytest.approx(0.5)
