# muxnet

Muxnet is a Bittensor subnet. Validators ask miners for a number, and miners
are paid when they return exactly twice that number. Scores turn into weights
on the subnet.

The Python package lives in `neurons` and installs as `muxnet`.

## How a round works

1. A validator samples miners that are serving and under the validator-permit
   stake limit (`neurons/utils/uids.py`).
2. It sends a `Dummy` synapse whose `dummy_input` is the validator's current
   step (`neurons/protocol.py`).
3. The miner sets `dummy_output` to `2 * dummy_input` (`neurons/miner.py`,
   `neurons/incentive.py`).
4. The response scores `1.0` on an exact match and `0.0` otherwise, including
   timeouts (`neurons/validator/reward.py`).
5. Scores are an exponential moving average (`--neuron.moving_average_alpha`,
   default `0.1`). Each step also decays miners that were not queried.
6. When the epoch elapses, the validator normalizes those scores and sets
   weights (`neurons/base/utils/weight_utils.py`).

The validator waits 5 seconds at the end of each forward pass. That wait is
asynchronous, so concurrent forwards do not block each other.

Validator state (`step`, `scores`, `hotkeys`) is written to `state.npz` under
the neuron log directory. A missing file starts the validator from zero.

## Requirements

- Python 3.10, 3.11, or 3.12
- A Bittensor wallet and a subtensor endpoint to run a miner or validator
- The [bittensor](https://pypi.org/project/bittensor/) SDK

Unit tests do not install `bittensor` or talk to a chain. `bittensor` is not
pinned in `requirements.txt`: install the SDK release that matches the network
you join. `pip install -e .` still installs the libraries listed in
`requirements.txt`, including PyTorch.

Hardware for this reference incentive is modest. See
[`min_compute.yml`](min_compute.yml). A GPU is not required to double an
integer.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install bittensor
```

## Run a miner or validator

From an environment with `bittensor` installed:

```bash
python -m neurons.miner --help
python -m neurons.validator --help
```

Example, after the wallet is registered on the subnet:

```bash
python -m neurons.miner \
  --netuid 1 \
  --wallet.name miner \
  --wallet.hotkey default \
  --subtensor.network test \
  --blacklist.force_validator_permit

python -m neurons.validator \
  --netuid 1 \
  --wallet.name validator \
  --wallet.hotkey default \
  --subtensor.network test \
  --neuron.sample_size 16 \
  --neuron.moving_average_alpha 0.1
```

`--wallet.*` and `--subtensor.*` come from the Bittensor SDK. `--netuid` and
the `--neuron.*` flags are defined in `neurons/utils/config.py`. Pass `--mock`
to construct the in-repo mock subtensor, metagraph, and dendrite instead of
dialing a public endpoint.

Miners reject requests that arrive without a hotkey. Unregistered hotkeys are
rejected unless `--blacklist.allow_non_registered` is set. With
`--blacklist.force_validator_permit`, only hotkeys that have a validator
permit are served. Priority follows the caller's stake.

Longer notes on local chains, testnet, and mainnet were inherited from the
upstream subnet template:

- [Local / staging chain](docs/running_on_staging.md)
- [Testnet](docs/running_on_testnet.md)
- [Mainnet](docs/running_on_mainnet.md)

Those guides still mention cloning `bittensor-subnet-template` and importing
`template`. In this repository the package is `neurons`, the project name is
`muxnet`, and the commands above are the ones to use.

## Development and tests

Runtime dependencies and test dependencies are separate. The test extra does
not need PyTorch or `bittensor`.

```bash
python -m pip install -r requirements-dev.txt
ruff check .
pytest
```

`requirements-dev.txt` installs NumPy, Pydantic, pytest, and ruff. To install
the package together with those tools (this also installs `requirements.txt`,
including PyTorch):

```bash
python -m pip install -e ".[dev]"
```

GitHub Actions (`.github/workflows/ci.yml`) runs `ruff check` and `pytest` on
Python 3.10, 3.11, and 3.12. Ruff is configured in `pyproject.toml`. The
`docs/` tutorials are excluded from lint; they are standalone examples.

The tests cover the incentive rule, miner blacklist and priority, UID
sampling, weight normalization and chain integer encoding, the dummy subnet
API, validator checkpoints, the TTL cache, and one validator forward pass
against stand-in axons.

## Layout

| Path | Role |
| --- | --- |
| `neurons/miner.py` | Miner entry point |
| `neurons/validator.py` | Validator entry point |
| `neurons/protocol.py` | `Dummy` synapse |
| `neurons/incentive.py` | `dummy_output = 2 * dummy_input` |
| `neurons/validator/forward.py` | Query, score, update |
| `neurons/validator/reward.py` | Reward and moving average |
| `neurons/api/dummy.py` | Client helper for the same synapse |
| `neurons/base/` | Wallet, axon, metagraph, and weight setting |
| `tests/` | Pytest suite |

`from neurons import SUBNET_LINKS` returns the bundled list of example
subnets. Importing `neurons` does not import `bittensor`; the miner,
validator, and protocol modules load it when they are used.

## License

MIT. See [LICENSE](LICENSE).
