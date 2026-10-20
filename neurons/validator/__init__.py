"""Validator forward pass and reward helpers.

``forward`` imports the chain client, so it is loaded only when asked for.
``reward`` stays importable for local scoring tests.
"""

_LAZY = ("forward", "reward")


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
