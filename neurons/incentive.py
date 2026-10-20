"""Shared incentive rule for the dummy miner and validator.

Validators query miners with an integer. A miner is rewarded only when it
returns exactly twice that integer.
"""


def expected_dummy_output(dummy_input: int) -> int:
    """Return the miner output that earns a full reward for ``dummy_input``."""
    return dummy_input * 2
