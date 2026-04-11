"""GigaEvo CLI tools for experiment analysis."""

from __future__ import annotations

import click


@click.group()
def main() -> None:
    """GigaEvo CLI -- experiment analysis tools."""


from gigaevo.cli.analyze import analyze  # noqa: E402
from gigaevo.cli.collect import collect  # noqa: E402
from gigaevo.cli.plot import plot  # noqa: E402
from gigaevo.cli.status import status  # noqa: E402

main.add_command(status)
main.add_command(collect)
main.add_command(plot)
main.add_command(analyze)
