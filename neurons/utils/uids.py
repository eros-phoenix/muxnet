import random
from typing import List

import numpy as np

from neurons.utils.misc import as_int


def check_uid_availability(metagraph, uid: int, vpermit_tao_limit: int) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True


def get_random_uids(self, k: int, exclude: List[int] = None) -> np.ndarray:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (np.ndarray): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
        Excluded UIDs are skipped until there are not enough remaining candidates.
        In that case the sample is filled from the other available UIDs so the
        validator still queries `k` miners.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if k < 0:
        raise ValueError("k must be non-negative")

    excluded = set() if exclude is None else {as_int(uid) for uid in exclude}
    candidate_uids = []
    avail_uids = []

    for uid in range(as_int(self.metagraph.n)):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        if not uid_is_available:
            continue
        avail_uids.append(uid)
        if uid not in excluded:
            candidate_uids.append(uid)

    # If k is larger than the number of available uids, set k to the number of available uids.
    k = min(k, len(avail_uids))
    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = list(candidate_uids)
    if len(candidate_uids) < k:
        filler = [uid for uid in avail_uids if uid not in candidate_uids]
        available_uids += random.sample(filler, k - len(candidate_uids))
    k = min(k, len(available_uids))
    uids = np.array(random.sample(available_uids, k), dtype=np.int64)
    return uids
