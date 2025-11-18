"""GigaEvo – evolutionary computation framework."""

__version__ = "1.11.1"

from pydantic import config as _pyd_config

# Enable Pydantic V2 JIT compilation for faster model (de)serialisation.
try:
    _pyd_config.configure(compile="jit")
except Exception:  # pragma: no cover – older pydantic
    pass
