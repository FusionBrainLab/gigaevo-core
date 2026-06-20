"""LazyGroup pipe handling.

`gigaevo <cmd> | head` (or any reader that closes early) must terminate
cleanly: no BrokenPipeError traceback, and no buffered output lost to a
second BrokenPipeError during the interpreter's shutdown flush. The handler
lives in ``LazyGroup.invoke`` so it covers every subcommand, including the
nested ``plot``/``manifest``/``flush`` groups.
"""

from __future__ import annotations

import os
import sys
import types

import click
import pytest

from gigaevo.cli import LazyGroup


def _ctx() -> click.Context:
    return click.Context(LazyGroup(name="gigaevo"))


def test_broken_pipe_during_subcommand_is_swallowed(monkeypatch):
    """EPIPE from a subcommand -> clean SystemExit + stdout redirected to devnull."""
    grp = LazyGroup(name="gigaevo")

    def boom(self, ctx):
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(click.Group, "invoke", boom)

    # Don't clobber the test runner's real stdout: fake the fd + record dup2.
    monkeypatch.setattr(
        sys, "stdout", types.SimpleNamespace(fileno=lambda: 7, flush=lambda: None)
    )
    redirected: dict[str, int] = {}
    monkeypatch.setattr(os, "open", lambda path, flags: 1234)
    monkeypatch.setattr(
        os, "dup2", lambda fd, target: redirected.update(fd=fd, target=target)
    )

    with pytest.raises(SystemExit):
        grp.invoke(_ctx())

    assert redirected.get("fd") == 1234, "devnull fd should be duped over stdout"
    assert redirected.get("target") == 7, (
        "stdout's fileno should be the redirect target"
    )


def test_broken_pipe_on_final_flush_is_swallowed(monkeypatch):
    """Small buffered output: EPIPE surfaces on the post-invoke flush, not mid-run."""
    grp = LazyGroup(name="gigaevo")

    monkeypatch.setattr(click.Group, "invoke", lambda self, ctx: None)

    def flush_epipe():
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(
        sys, "stdout", types.SimpleNamespace(fileno=lambda: 7, flush=flush_epipe)
    )
    monkeypatch.setattr(os, "open", lambda path, flags: 1234)
    monkeypatch.setattr(os, "dup2", lambda fd, target: None)

    with pytest.raises(SystemExit):
        grp.invoke(_ctx())


def test_missing_manifest_still_becomes_clickexception(monkeypatch):
    """Regression: the pre-existing experiment.yaml FileNotFoundError path is intact."""
    grp = LazyGroup(name="gigaevo")
    monkeypatch.setattr(
        click.Group,
        "invoke",
        lambda self, ctx: (_ for _ in ()).throw(
            FileNotFoundError("experiment.yaml missing")
        ),
    )
    with pytest.raises(click.ClickException):
        grp.invoke(_ctx())


def test_unrelated_filenotfound_reraises(monkeypatch):
    """A FileNotFoundError unrelated to experiment.yaml must still propagate raw."""
    grp = LazyGroup(name="gigaevo")
    monkeypatch.setattr(
        click.Group,
        "invoke",
        lambda self, ctx: (_ for _ in ()).throw(FileNotFoundError("some other file")),
    )
    with pytest.raises(FileNotFoundError):
        grp.invoke(_ctx())
