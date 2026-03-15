"""Tests for CompositeLogger and BoundComposite (trackers/composite.py)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gigaevo.utils.trackers.composite import BoundComposite, CompositeLogger
from tests.conftest import NullWriter


def _mock_writer():
    m = MagicMock(spec=NullWriter)
    m.bind.return_value = m
    return m


# ---------------------------------------------------------------------------
# CompositeLogger construction
# ---------------------------------------------------------------------------


class TestCompositeLoggerInit:
    def test_requires_at_least_one_logger(self):
        with pytest.raises(ValueError, match="at least one"):
            CompositeLogger([])

    def test_single_logger_allowed(self):
        w = _mock_writer()
        cl = CompositeLogger([w])
        cl.scalar("loss", 1.0)
        w.scalar.assert_called_once_with("loss", 1.0)


# ---------------------------------------------------------------------------
# Fan-out: every write method must broadcast to all loggers
# ---------------------------------------------------------------------------


class TestCompositeLoggerFanout:
    def test_scalar_broadcasts_to_all(self):
        w1, w2 = _mock_writer(), _mock_writer()
        cl = CompositeLogger([w1, w2])
        cl.scalar("loss", 0.5, step=3)
        w1.scalar.assert_called_once_with("loss", 0.5, step=3)
        w2.scalar.assert_called_once_with("loss", 0.5, step=3)

    def test_hist_broadcasts_to_all(self):
        w1, w2 = _mock_writer(), _mock_writer()
        cl = CompositeLogger([w1, w2])
        cl.hist("weights", [1.0, 2.0])
        w1.hist.assert_called_once_with("weights", [1.0, 2.0])
        w2.hist.assert_called_once_with("weights", [1.0, 2.0])

    def test_text_broadcasts_to_all(self):
        w1, w2 = _mock_writer(), _mock_writer()
        cl = CompositeLogger([w1, w2])
        cl.text("log", "hello")
        w1.text.assert_called_once_with("log", "hello")
        w2.text.assert_called_once_with("log", "hello")

    def test_close_broadcasts_to_all(self):
        w1, w2 = _mock_writer(), _mock_writer()
        cl = CompositeLogger([w1, w2])
        cl.close()
        w1.close.assert_called_once()
        w2.close.assert_called_once()

    def test_three_loggers_all_receive(self):
        writers = [_mock_writer() for _ in range(3)]
        cl = CompositeLogger(writers)
        cl.scalar("x", 1.0)
        for w in writers:
            w.scalar.assert_called_once_with("x", 1.0)


# ---------------------------------------------------------------------------
# Isolation: one failing writer must not silence others
# ---------------------------------------------------------------------------


class TestCompositeLoggerIsolation:
    def test_scalar_exception_does_not_propagate(self):
        w1, w2 = _mock_writer(), _mock_writer()
        w1.scalar.side_effect = RuntimeError("backend died")
        cl = CompositeLogger([w1, w2])
        cl.scalar("loss", 1.0)  # must not raise
        w2.scalar.assert_called_once_with("loss", 1.0)

    def test_hist_exception_does_not_propagate(self):
        w1, w2 = _mock_writer(), _mock_writer()
        w1.hist.side_effect = OSError("disk full")
        cl = CompositeLogger([w1, w2])
        cl.hist("w", [1.0])  # must not raise
        w2.hist.assert_called_once_with("w", [1.0])

    def test_text_exception_does_not_propagate(self):
        w1, w2 = _mock_writer(), _mock_writer()
        w1.text.side_effect = ValueError("bad text")
        cl = CompositeLogger([w1, w2])
        cl.text("msg", "hi")  # must not raise
        w2.text.assert_called_once_with("msg", "hi")

    def test_close_exception_does_not_propagate(self):
        w1, w2 = _mock_writer(), _mock_writer()
        w1.close.side_effect = RuntimeError("close failed")
        cl = CompositeLogger([w1, w2])
        cl.close()  # must not raise
        w2.close.assert_called_once()


# ---------------------------------------------------------------------------
# bind → BoundComposite
# ---------------------------------------------------------------------------


class TestCompositeLoggerBind:
    def test_bind_returns_bound_composite(self):
        w = _mock_writer()
        cl = CompositeLogger([w])
        bound = cl.bind(path=["x"])
        assert isinstance(bound, BoundComposite)

    def test_bind_no_path_returns_bound_composite(self):
        w = _mock_writer()
        cl = CompositeLogger([w])
        bound = cl.bind()
        assert isinstance(bound, BoundComposite)


# ---------------------------------------------------------------------------
# BoundComposite
# ---------------------------------------------------------------------------


class TestBoundComposite:
    def test_scalar_prepends_path(self):
        w = _mock_writer()
        cl = CompositeLogger([w])
        bound = cl.bind(path=["train"])
        bound.scalar("loss", 0.5)
        w.scalar.assert_called_once_with("loss", 0.5, path=["train"])

    def test_nested_bind_concatenates_path(self):
        w = _mock_writer()
        cl = CompositeLogger([w])
        bound = cl.bind(path=["a"]).bind(path=["b"])
        bound.scalar("loss", 0.5)
        w.scalar.assert_called_once_with("loss", 0.5, path=["a", "b"])

    def test_hist_prepends_path(self):
        w = _mock_writer()
        cl = CompositeLogger([w])
        bound = cl.bind(path=["eval"])
        bound.hist("weights", [1.0])
        w.hist.assert_called_once_with("weights", [1.0], path=["eval"])

    def test_text_prepends_path(self):
        w = _mock_writer()
        cl = CompositeLogger([w])
        bound = cl.bind(path=["debug"])
        bound.text("msg", "hi")
        w.text.assert_called_once_with("msg", "hi", path=["debug"])

    def test_close_delegates_to_base(self):
        w = _mock_writer()
        cl = CompositeLogger([w])
        bound = cl.bind(path=["x"])
        bound.close()
        w.close.assert_called_once()

    def test_extra_path_in_call_merges_after_bound_path(self):
        """path kwarg in call is appended after the bound path."""
        w = _mock_writer()
        cl = CompositeLogger([w])
        bound = cl.bind(path=["train"])
        bound.scalar("loss", 1.0, path=["epoch"])
        w.scalar.assert_called_once_with("loss", 1.0, path=["train", "epoch"])

    def test_empty_bound_path_passes_kwargs_unchanged(self):
        w = _mock_writer()
        cl = CompositeLogger([w])
        bound = cl.bind(path=[])
        bound.scalar("loss", 1.0, step=5)
        w.scalar.assert_called_once_with("loss", 1.0, path=[], step=5)

    def test_bound_bind_returns_new_bound_composite(self):
        w = _mock_writer()
        cl = CompositeLogger([w])
        b1 = cl.bind(path=["x"])
        b2 = b1.bind(path=["y"])
        assert isinstance(b2, BoundComposite)
        assert b1 is not b2
