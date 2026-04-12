"""OutputFormatter: structured output in table/json/csv/markdown formats."""

from __future__ import annotations

import csv
from io import StringIO
import json
import sys

import click


class OutputFormatter:
    """Render data rows in multiple output formats.

    Supports table (Rich), json, csv, and markdown. Auto-detects pipe
    (non-tty stdout) and switches to json unless an explicit format is given.
    """

    def __init__(self, format_name: str | None = None, quiet: bool = False):
        self._format_name = format_name
        self._quiet = quiet

    @property
    def effective_format(self) -> str:
        if self._format_name is not None:
            return self._format_name
        if not sys.stdout.isatty():
            return "json"
        return "table"

    def render(
        self,
        rows: list[dict],
        columns: list[str] | None = None,
        title: str | None = None,
    ) -> str:
        fmt = self.effective_format
        if fmt == "json":
            return self._render_json(rows)
        if fmt == "csv":
            return self._render_csv(rows, columns)
        if fmt == "markdown":
            return self._render_markdown(rows, columns)
        return self._render_table(rows, columns, title)

    def echo(
        self,
        rows: list[dict],
        columns: list[str] | None = None,
        title: str | None = None,
    ) -> None:
        if self._quiet:
            return
        output = self.render(rows, columns=columns, title=title)
        click.echo(output)

    def _render_json(self, rows: list[dict]) -> str:
        return json.dumps(rows, indent=2, default=str)

    def _render_csv(self, rows: list[dict], columns: list[str] | None) -> str:
        if not rows:
            return ""
        cols = columns or list(rows[0].keys())
        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return buf.getvalue()

    def _render_markdown(self, rows: list[dict], columns: list[str] | None) -> str:
        if not rows:
            return ""
        cols = columns or list(rows[0].keys())
        lines = []
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for row in rows:
            vals = [str(row.get(c, "")) for c in cols]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    def _render_table(
        self, rows: list[dict], columns: list[str] | None, title: str | None
    ) -> str:
        try:
            from rich.console import Console
            from rich.table import Table

            cols = columns or (list(rows[0].keys()) if rows else [])
            table = Table(title=title)
            for col in cols:
                table.add_column(col)
            for row in rows:
                table.add_row(*[str(row.get(c, "")) for c in cols])
            buf = StringIO()
            console = Console(file=buf, width=120)
            console.print(table)
            return buf.getvalue()
        except ImportError:
            return self._render_markdown(rows, columns)
