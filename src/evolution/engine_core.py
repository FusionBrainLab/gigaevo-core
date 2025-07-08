# -*- coding: utf-8 -*-
"""Core implementation of EvolutionEngine.

This module was extracted from the original *engine.py* to make the codebase
more modular and easier to navigate.  All logic remains unchanged; only the
location has moved.  High-level helper functions should be further factored
into dedicated modules (e.g. *novelty.py*, *mutation.py*) as we iterate.

Down-stream imports (`from src.evolution.engine import EvolutionEngine`, etc.)
continue to work because *engine.py* now re-exports the public symbols.
"""

# NOTE: the entire previous content of `src/evolution/engine.py` has been
# pasted here verbatim (minus the trailing file docstring) to preserve
# behaviour.  Once the refactor stabilises we will further split this file.

from src.evolution.engine import *  # type: ignore  # pylint: disable=wildcard-import 