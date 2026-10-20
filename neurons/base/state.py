"""Save and load validator score state without requiring a chain connection."""

import os

import numpy as np


def state_path(full_path: str) -> str:
    return os.path.join(full_path, "state.npz")


def save_validator_state(full_path, step, scores, hotkeys) -> str:
    """Persist step, scores, and hotkeys. Returns the file path written."""
    path = state_path(full_path)
    np.savez(
        path,
        step=np.int64(step),
        scores=np.asarray(scores, dtype=np.float32),
        hotkeys=np.asarray(list(hotkeys)),
    )
    return path


def load_validator_state(full_path):
    """Load a saved validator state, or ``None`` when no file exists.

    Object arrays (possible for ragged hotkey lists) need ``allow_pickle``.
    """
    path = state_path(full_path)
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=True) as state:
        hotkeys = state["hotkeys"]
        return {
            "step": int(state["step"]),
            "scores": np.asarray(state["scores"], dtype=np.float32),
            "hotkeys": [str(hotkey) for hotkey in hotkeys.tolist()],
        }


def fit_scores_to_network(scores, n_neurons: int) -> np.ndarray:
    """Resize a saved score vector to the current metagraph size.

    New UIDs start at 0. Scores that no longer have a UID are dropped.
    """
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    n_neurons = int(n_neurons)
    if n_neurons < 0:
        raise ValueError("n_neurons must be non-negative")
    if scores.shape == (n_neurons,):
        return scores
    resized = np.zeros(n_neurons, dtype=np.float32)
    copy_n = min(n_neurons, int(scores.shape[0]))
    if copy_n:
        resized[:copy_n] = scores[:copy_n]
    return resized
