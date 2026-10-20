"""Decide whether a miner should serve a request, and in what order.

These checks used to live inline on ``Miner`` and called ``list.index``
before testing membership, so an unknown hotkey raised ``ValueError``
instead of being blacklisted.
"""


def _uid_for_hotkey(hotkey, hotkeys):
    try:
        return list(hotkeys).index(hotkey)
    except ValueError:
        return None


def decide_blacklist(
    hotkey,
    hotkeys,
    validator_permit,
    *,
    allow_non_registered,
    force_validator_permit,
):
    """Return ``(reject, reason)`` for one incoming request.

    ``hotkey`` is the caller advertised on the request. ``hotkeys`` is the
    metagraph hotkey list, aligned with ``validator_permit``.
    """
    if hotkey is None:
        return True, "Missing dendrite or hotkey"

    uid = _uid_for_hotkey(hotkey, hotkeys)
    if uid is None and not allow_non_registered:
        return True, "Unrecognized hotkey"

    if force_validator_permit and (
        uid is None or not bool(validator_permit[uid])
    ):
        return True, "Non-validator hotkey"

    return False, "Hotkey recognized!"


def request_priority(hotkey, hotkeys, stake) -> float:
    """Priority is the caller's stake. Unknown or anonymous callers get 0."""
    if hotkey is None:
        return 0.0
    uid = _uid_for_hotkey(hotkey, hotkeys)
    if uid is None:
        return 0.0
    return float(stake[uid])
