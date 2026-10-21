"""Align validator scores with the metagraph hotkey table.

Scores are kept per UID slot. A slot whose hotkey changed is reset to 0.
Slots that disappear when the subnet shrinks are dropped, and slots that
appear when it grows start at 0.
"""

from typing import Sequence

import numpy as np


def align_scores_to_hotkeys(
    scores, previous_hotkeys: Sequence, new_hotkeys: Sequence
) -> np.ndarray:
    """Return a float32 score vector positioned on ``new_hotkeys``.

    Values stay with the UID index, not with the hotkey identity. A UID
    whose hotkey differs from ``previous_hotkeys`` is set to 0. UIDs that
    exist only in ``new_hotkeys`` start at 0, and UIDs that were removed
    are left out. The input array is not modified.

    The returned vector has one entry per new hotkey, so it can be indexed
    by the current metagraph UID list. Indexing the previous hotkey list
    with every old UID raises ``IndexError`` once the subnet shrinks, and
    a score vector longer than the new UID list breaks weight emission.
    """
    if isinstance(previous_hotkeys, (str, bytes)) or isinstance(
        new_hotkeys, (str, bytes)
    ):
        raise TypeError(
            "hotkeys must be a sequence of hotkey ids, not a string"
        )

    vector = np.asarray(scores, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError(
            f"scores must be a 1-D vector, got shape {vector.shape}"
        )

    previous = list(previous_hotkeys)
    updated = list(new_hotkeys)
    aligned = np.zeros(len(updated), dtype=np.float32)
    overlap = min(len(previous), len(updated), int(vector.shape[0]))
    if overlap:
        aligned[:overlap] = vector[:overlap]
    for uid in range(min(len(previous), len(updated))):
        if previous[uid] != updated[uid]:
            aligned[uid] = np.float32(0.0)
    return aligned
