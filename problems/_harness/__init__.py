"""Shared grading harness for the ACI, hexagon and spherical-code problem families.

The families under `problems/` supply the task; this package supplies the parts they
must not each own a copy of — the sandbox every candidate runs in, the controller the
improve-arms drive, and the protocol constants that make three arms of one benchmark
comparable. A constant that lived in three `validate.py` files could drift in one of
them, and the three-way comparison would quietly become a comparison of harnesses.
"""

from __future__ import annotations

from pathlib import Path

HARNESS = Path(__file__).resolve().parent
REPO = HARNESS.parents[1]
