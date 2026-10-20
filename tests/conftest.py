"""Install a stand-in bittensor SDK when the real one is not installed.

Unit tests cover incentive, sampling, and weight logic without a chain node.
Modules that only need ``bt.Synapse`` or ``bt.logging`` can still be imported.
"""

import sys
import types


def _install_bittensor_stub() -> None:
    try:
        import bittensor  # noqa: F401
    except ImportError:
        pass
    else:
        return

    from pydantic import BaseModel, ConfigDict

    bt = types.ModuleType("bittensor")

    class Synapse(BaseModel):
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

        def deserialize(self):
            return None

    class _Logging:
        def __getattr__(self, _name):
            def _log(*_args, **_kwargs):
                return None

            return _log

    class MockSubtensor:
        def __init__(self, *args, **kwargs):
            pass

    class _Metagraph:
        def __init__(self, *args, **kwargs):
            pass

    class _Dendrite:
        def __init__(self, *args, **kwargs):
            pass

    class _Axon:
        def __init__(self, *args, **kwargs):
            pass

    bt.Synapse = Synapse
    bt.logging = _Logging()
    bt.MockSubtensor = MockSubtensor
    bt.MockWallet = object
    bt.metagraph = _Metagraph
    bt.dendrite = _Dendrite
    bt.axon = _Axon
    bt.wallet = types.SimpleNamespace(add_args=lambda parser: None)
    bt.subtensor = types.SimpleNamespace(add_args=lambda parser: None)
    bt.config = lambda parser: parser.parse_args([])

    subnets = types.ModuleType("bittensor.subnets")

    class SubnetsAPI:
        def __init__(self, wallet=None):
            self.wallet = wallet

    subnets.SubnetsAPI = SubnetsAPI
    bt.subnets = subnets
    sys.modules["bittensor"] = bt
    sys.modules["bittensor.subnets"] = subnets


_install_bittensor_stub()
