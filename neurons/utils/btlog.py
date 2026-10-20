"""Log through bittensor when it is installed, otherwise the stdlib logger.

Core scoring and weight code should stay importable in unit tests and in
environments that have not installed the chain SDK yet.
"""

import logging

_logger = logging.getLogger("neurons")


def _fallback(level, args):
    message = " ".join(str(arg) for arg in args)
    getattr(_logger, level)(message)


def _emit(level, args):
    try:
        import bittensor as bt
    except ImportError:
        _fallback(level, args)
        return
    getattr(bt.logging, level)(*args)


def debug(*args):
    _emit("debug", args)


def info(*args):
    _emit("info", args)


def warning(*args):
    _emit("warning", args)


def error(*args):
    _emit("error", args)
