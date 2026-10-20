"""TTL cache, event log setup, and the retention-size CLI flag."""

import argparse
import logging

import pytest

from neurons.utils.config import add_args
from neurons.utils.logging import EVENTS_LEVEL_NUM, setup_events_logger
from neurons.utils.misc import ttl_cache
import neurons.utils.misc as misc


def test_ttl_cache_reuses_a_value_until_the_window_expires(monkeypatch):
    clock = {"now": 1_000.0}

    def _now():
        return clock["now"]

    monkeypatch.setattr(misc.time, "time", _now)
    calls = {"n": 0}

    @ttl_cache(maxsize=8, ttl=5)
    def compute(value):
        calls["n"] += 1
        return value + 1

    assert compute(1) == 2
    assert compute(1) == 2
    assert compute(2) == 3
    assert calls["n"] == 2

    clock["now"] += 5
    assert compute(1) == 2
    assert calls["n"] == 3


def test_events_logger_writes_once_and_does_not_stack_handlers(tmp_path):
    logger = logging.getLogger("event")
    logger.handlers.clear()
    try:
        first = setup_events_logger(str(tmp_path), 1024)
        second = setup_events_logger(str(tmp_path), 1024)
        assert first is second
        assert len(logger.handlers) == 1
        logger.event("validator scored a miner")
        for handler in logger.handlers:
            handler.flush()
        contents = (tmp_path / "events.log").read_text()
        assert "validator scored a miner" in contents
        assert "EVENT" in contents
    finally:
        for handler in list(logger.handlers):
            handler.close()
        logger.handlers.clear()


def test_events_logger_rejects_a_bad_retention_size(tmp_path):
    with pytest.raises(ValueError):
        setup_events_logger(str(tmp_path), "lots")
    with pytest.raises(ValueError):
        setup_events_logger(str(tmp_path), 0)
    assert EVENTS_LEVEL_NUM == 38


def test_events_retention_cli_flag_is_an_int():
    parser = argparse.ArgumentParser()
    add_args(object(), parser)
    args = parser.parse_args(["--neuron.events_retention_size", "4096"])
    assert args.__dict__["neuron.events_retention_size"] == 4096
    default = parser.parse_args([])
    assert isinstance(default.__dict__["neuron.events_retention_size"], int)
