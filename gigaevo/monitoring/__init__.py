"""GigaEvo monitoring library -- shared Redis queries, snapshots, and alerts."""

from gigaevo.monitoring.alerts import Alert, AlertDetector, AlertSeverity, AlertType
from gigaevo.monitoring.experiment_monitor import ExperimentMonitor, RunConfig
from gigaevo.monitoring.manifest_schema import ExperimentManifest, export_json_schema
from gigaevo.monitoring.notifications import (
    NotificationChannel,
    PlotAttachment,
    StatusUpdate,
    format_alert_message,
    format_status_table_markdown,
    format_status_table_telegram,
)
from gigaevo.monitoring.redis_queries import collect_snapshot
from gigaevo.monitoring.run_spec import RunSpec
from gigaevo.monitoring.snapshot import RunSnapshot

__all__ = [
    "Alert",
    "AlertDetector",
    "AlertSeverity",
    "AlertType",
    "ExperimentManifest",
    "ExperimentMonitor",
    "NotificationChannel",
    "PlotAttachment",
    "RunConfig",
    "RunSpec",
    "RunSnapshot",
    "StatusUpdate",
    "collect_snapshot",
    "export_json_schema",
    "format_alert_message",
    "format_status_table_markdown",
    "format_status_table_telegram",
]
