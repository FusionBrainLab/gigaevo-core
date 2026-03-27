#!/usr/bin/env python3
"""Print NO_PROXY value derived from experiments/infrastructure.yaml.

Usage in shell scripts:
    export NO_PROXY=$(python tools/no_proxy.py)
    export no_proxy="$NO_PROXY"

Usage in Python:
    from tools.no_proxy import ensure_no_proxy
    ensure_no_proxy()  # idempotent, safe to call multiple times
"""

import os
from pathlib import Path

import yaml

_INFRA_YAML = Path(__file__).parent.parent / "experiments" / "infrastructure.yaml"


def get_no_proxy_hosts() -> list[str]:
    """Read no_proxy_hosts from infrastructure.yaml."""
    with open(_INFRA_YAML) as f:
        infra = yaml.safe_load(f)
    return infra.get("no_proxy_hosts", [])


def get_no_proxy_string() -> str:
    """Return comma-separated NO_PROXY value."""
    return ",".join(get_no_proxy_hosts())


def ensure_no_proxy() -> None:
    """Set NO_PROXY/no_proxy env vars from infrastructure.yaml (idempotent)."""
    hosts = get_no_proxy_hosts()
    current = os.environ.get("NO_PROXY", "")
    for h in hosts:
        if h not in current:
            current = ",".join(filter(None, [current, h]))
    os.environ["NO_PROXY"] = current
    os.environ["no_proxy"] = current


if __name__ == "__main__":
    print(get_no_proxy_string())
