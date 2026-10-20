"""Client-side synapse preparation and API node selection."""

from types import SimpleNamespace

from neurons.api.dummy import DummyAPI
from neurons.api.selection import candidate_api_uids
from neurons.protocol import Dummy


def test_dummy_synapse_round_trips_its_output():
    synapse = Dummy(dummy_input=4)
    assert synapse.deserialize() is None
    synapse.dummy_output = 8
    assert synapse.deserialize() == 8


def test_dummy_api_prepares_a_synapse_and_keeps_every_success():
    api = DummyAPI(wallet=object())
    synapse = api.prepare_synapse(3)
    assert isinstance(synapse, Dummy)
    assert synapse.dummy_input == 3

    def response(status, output, with_dendrite=True):
        dendrite = SimpleNamespace(status_code=status) if with_dendrite else None
        return SimpleNamespace(dendrite=dendrite, dummy_output=output)

    outputs = api.process_responses(
        [
            response(200, 4),
            response(500, 9),
            response(200, 6),
            response(200, None),
            response(200, 0, with_dendrite=False),
        ]
    )
    assert outputs == [4, 6]


def test_candidate_api_uids_include_ties_at_the_stake_cutoff():
    uids = list(range(5))
    trust = [0, 1, 1, 0, 1]
    # Every stake is equal, so a strict ">" cutoff used to return nobody.
    selected = candidate_api_uids(uids, trust, [10, 10, 10, 10, 10], fraction=0.1)
    assert selected == [1, 2, 4]


def test_candidate_api_uids_prefer_the_highest_stake_among_trusted_nodes():
    uids = list(range(5))
    trust = [1, 1, 1, 1, 1]
    stake = [1, 1, 1, 1, 100]
    selected = candidate_api_uids(uids, trust, stake, fraction=0.1)
    assert selected == [4]


def test_candidate_api_uids_ignore_untrusted_whales_and_empty_networks():
    selected = candidate_api_uids(
        list(range(3)),
        validator_trust=[0, 0, 0],
        stake=[1, 50, 3],
        fraction=1,
    )
    assert selected == []
    assert candidate_api_uids([], [], [], fraction=0.1) == []
    assert candidate_api_uids([0], [1], [5], fraction=0) == []
