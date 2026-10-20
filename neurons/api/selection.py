"""Choose candidate API nodes from stake and validator trust.

Top-node selection is index based: metagraph UIDs are expected to be
``0 .. n-1``, matching ``np.where`` on the stake vector.
"""

import numpy as np

from neurons.utils.misc import as_int


def candidate_api_uids(uids, validator_trust, stake, fraction=0.1):
    """Return sorted UIDs that are both trusted and in the top stake fraction.

    Nodes tied at the quantile cutoff are included. A strict ``>`` cutoff
    drops every node when stakes are uniform, which leaves the client with
    nobody to query.
    """
    uid_list = [as_int(uid) for uid in uids]
    trust = list(validator_trust)
    stake_arr = np.asarray(stake, dtype=np.float64)
    if stake_arr.size == 0 or not uid_list:
        return []

    fraction = float(fraction)
    if fraction <= 0:
        return []
    if fraction > 1:
        fraction = 1.0

    trusted = {
        uid
        for uid in uid_list
        if 0 <= uid < len(trust) and trust[uid] > 0
    }
    cutoff = float(np.quantile(stake_arr, 1 - fraction))
    top = set(np.where(stake_arr >= cutoff)[0].tolist())
    return sorted(trusted & top)
