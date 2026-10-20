"""Utility helpers for muxnet neurons.

Submodules are imported lazily. ``config`` pulls in bittensor, while
``misc`` and ``uids`` are usable without it.
"""

_LAZY_MODULES = ("config", "logging", "misc", "uids")


def __getattr__(name):
    if name in _LAZY_MODULES:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
