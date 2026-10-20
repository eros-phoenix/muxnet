# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import numpy as np
from typing import List

from neurons.incentive import expected_dummy_output
from neurons.utils.btlog import info


def reward(query: int, response: int) -> float:
    """
    Reward the miner response to the dummy request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    # A missing or non-matching response scores 0. Timeouts often show up as None.
    try:
        value = (
            1.0 if response == expected_dummy_output(query) else 0.0
        )
    except TypeError:
        value = 0.0
    info(
        "In rewards, query val: "
        f"{query}, response val: {response}, rewards val: {value}"
    )
    return value


def get_rewards(
    self,
    query: int,
    responses: List[float],
) -> np.ndarray:
    """
    Returns an array of rewards for the given query and responses.

    Args:
    - query (int): The query sent to the miner.
    - responses (List[float]): A list of responses from the miner.

    Returns:
    - np.ndarray: An array of rewards for the given query and responses.
    """
    # Get all the reward results by iteratively calling your reward() function.
    if responses is None:
        responses = []
    return np.asarray(
        [reward(query, response) for response in responses], dtype=np.float32
    )


def apply_moving_average(scores, rewards, uids, alpha: float):
    """Blend new rewards into ``scores`` with an exponential moving average.

    Returns ``(updated_scores, status)`` where status is ``"ok"``, ``"nan"``,
    or ``"empty"``. Empty inputs leave ``scores`` unchanged. NaN rewards are
    replaced with 0 before the update. UIDs outside the score vector raise
    ``ValueError`` instead of wrapping around via negative indexes.
    """
    scores = np.asarray(scores, dtype=np.float32)
    rewards = np.asarray([] if rewards is None else rewards, dtype=np.float64)
    uids_array = np.asarray([] if uids is None else uids)

    nan_found = bool(rewards.size) and bool(np.isnan(rewards).any())
    if nan_found:
        rewards = np.nan_to_num(rewards, nan=0.0)

    if rewards.size == 0 or uids_array.size == 0:
        return scores, "empty"

    if rewards.size != uids_array.size:
        raise ValueError(
            f"Shape mismatch: rewards array of shape {rewards.shape} "
            f"cannot be broadcast to uids array of shape {uids_array.shape}"
        )

    uids_array = uids_array.astype(np.int64, copy=False)
    if np.any(uids_array < 0) or np.any(uids_array >= scores.shape[0]):
        raise ValueError(
            "UID out of range for scores of length "
            f"{scores.shape[0]}: {uids_array.tolist()}"
        )

    scattered = np.zeros(scores.shape[0], dtype=np.float64)
    scattered[uids_array] = rewards
    updated = (alpha * scattered) + ((1.0 - alpha) * scores)
    status = "nan" if nan_found else "ok"
    return updated.astype(np.float32, copy=False), status
