"""GigaEvo – evolutionary computation framework."""

__version__ = "2.3.0"

from pydantic import config as _pyd_config

# Enable Pydantic V2 JIT compilation for faster model (de)serialisation.
try:
    _pyd_config.configure(compile="jit")
except Exception:  # pragma: no cover – older pydantic
    pass

# Force canonical-event registration at package init. Each module's subclass
# definitions auto-register in gigaevo.monitoring.events.CANONICAL_EVENTS via
# BaseEvent.__init_subclass__.
from gigaevo.adversarial import events as _adv_events  # noqa: F401,E402
from gigaevo.monitoring import events as _mon_events  # noqa: F401,E402
