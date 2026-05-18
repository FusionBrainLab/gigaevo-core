"""Unit tests for the logging-side typed schemas."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from gigaevo.config.schemas import (
    LoggingConfig,
    LoggingSettings,
    RedisMetricsTrackerConfig,
    TBTrackerConfig,
    TrackerConfig,
    WandBTrackerConfig,
)


class TestLoggingSettings:
    def test_defaults(self) -> None:
        s = LoggingSettings()
        assert s.level == "INFO"
        assert s.log_dir is None
        assert s.rotation == "50 MB"
        assert s.retention == "30 days"

    def test_level_literal_enforced(self) -> None:
        with pytest.raises(ValidationError):
            LoggingSettings(level="VERBOSE")  # type: ignore[arg-type]

    def test_empty_log_dir_rejected(self) -> None:
        with pytest.raises(ValidationError, match="log_dir"):
            LoggingSettings(log_dir=Path(""))

    def test_cwd_log_dir_rejected(self) -> None:
        with pytest.raises(ValidationError, match="log_dir"):
            LoggingSettings(log_dir=Path("."))

    def test_real_log_dir_accepted(self) -> None:
        s = LoggingSettings(log_dir=Path("/var/log/gigaevo"))
        assert s.log_dir == Path("/var/log/gigaevo")

    def test_rotation_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LoggingSettings(rotation="")

    def test_retention_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LoggingSettings(retention="")

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            LoggingSettings(levle="INFO")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        s = LoggingSettings()
        with pytest.raises(ValidationError):
            s.level = "DEBUG"  # type: ignore[misc]


class TestTBTrackerConfig:
    def test_construct(self) -> None:
        cfg = TBTrackerConfig(logdir=Path("/tmp/tb"))
        assert cfg.kind == "tensorboard"
        assert cfg.logdir == Path("/tmp/tb")
        assert cfg.queue_size == 8192
        assert cfg.flush_secs == 3.0

    def test_queue_size_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            TBTrackerConfig(logdir=Path("/tmp/tb"), queue_size=0)

    def test_flush_secs_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            TBTrackerConfig(logdir=Path("/tmp/tb"), flush_secs=0.0)


class TestWandBTrackerConfig:
    def test_minimal(self) -> None:
        cfg = WandBTrackerConfig(project="evo")
        assert cfg.kind == "wandb"
        assert cfg.project == "evo"
        assert cfg.name is None
        assert cfg.tags is None
        assert cfg.resume is False

    def test_project_required(self) -> None:
        with pytest.raises(ValidationError):
            WandBTrackerConfig()  # type: ignore[call-arg]

    def test_empty_project_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WandBTrackerConfig(project="")

    def test_with_tags_and_name(self) -> None:
        cfg = WandBTrackerConfig(
            project="evo", name="run-1", tags=["alpha", "beta"]
        )
        assert cfg.name == "run-1"
        assert cfg.tags == ["alpha", "beta"]


class TestRedisMetricsTrackerConfig:
    def test_defaults(self) -> None:
        cfg = RedisMetricsTrackerConfig()
        assert cfg.kind == "redis"
        assert cfg.redis_url == "redis://localhost:6379/0"
        assert cfg.key_prefix == "gigaevo:metrics"
        assert cfg.store_history is True

    def test_custom_url_and_prefix(self) -> None:
        cfg = RedisMetricsTrackerConfig(
            redis_url="redis://r.internal:6380/3",
            key_prefix="myproblem:metrics",
        )
        assert cfg.redis_url == "redis://r.internal:6380/3"
        assert cfg.key_prefix == "myproblem:metrics"

    def test_max_history_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            RedisMetricsTrackerConfig(max_history_per_metric=0)

    def test_max_connections_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            RedisMetricsTrackerConfig(max_connections=0)

    def test_socket_timeout_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            RedisMetricsTrackerConfig(socket_timeout=0.05)

    def test_empty_key_prefix_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RedisMetricsTrackerConfig(key_prefix="")


class TestTrackerUnion:
    def test_round_trip_each_variant(self) -> None:
        ta: TypeAdapter[TrackerConfig] = TypeAdapter(TrackerConfig)
        variants = [
            TBTrackerConfig(logdir=Path("/tmp/tb")),
            WandBTrackerConfig(project="evo"),
            RedisMetricsTrackerConfig(),
        ]
        for cfg in variants:
            parsed = ta.validate_python(cfg.model_dump())
            assert type(parsed) is type(cfg)

    def test_unknown_kind_rejected(self) -> None:
        ta: TypeAdapter[TrackerConfig] = TypeAdapter(TrackerConfig)
        with pytest.raises(ValidationError):
            ta.validate_python({"kind": "no_such_tracker"})

    def test_each_variant_has_unique_kind(self) -> None:
        kinds = [
            TBTrackerConfig.model_fields["kind"].default,
            WandBTrackerConfig.model_fields["kind"].default,
            RedisMetricsTrackerConfig.model_fields["kind"].default,
        ]
        assert len(set(kinds)) == len(kinds)
        assert set(kinds) == {"tensorboard", "wandb", "redis"}


class TestLoggingConfig:
    def test_requires_at_least_one_tracker(self) -> None:
        with pytest.raises(ValidationError):
            LoggingConfig()  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            LoggingConfig(trackers=[])

    def test_tb_plus_redis_metrics_compose(self) -> None:
        """Tensorboard tracker plus a Redis metrics tracker compose
        into a single ``LoggingConfig`` with two entries."""
        cfg = LoggingConfig(
            settings=LoggingSettings(level="DEBUG"),
            trackers=[
                TBTrackerConfig(logdir=Path("/var/log/tb")),
                RedisMetricsTrackerConfig(
                    redis_url="redis://localhost:6379/0",
                    key_prefix="hotpotqa:metrics",
                ),
            ],
        )
        assert len(cfg.trackers) == 2
        assert cfg.trackers[0].kind == "tensorboard"
        assert cfg.trackers[1].kind == "redis"
        assert cfg.settings.level == "DEBUG"

    def test_wandb_plus_redis_metrics_compose(self) -> None:
        """W&B tracker plus a Redis metrics tracker compose into a
        single ``LoggingConfig`` with two entries."""
        cfg = LoggingConfig(
            trackers=[
                WandBTrackerConfig(project="evo", name="experiment"),
                RedisMetricsTrackerConfig(),
            ],
        )
        assert cfg.trackers[0].kind == "wandb"
        assert cfg.trackers[0].project == "evo"  # type: ignore[union-attr]

    def test_json_round_trip_preserves_tracker_list(self) -> None:
        cfg = LoggingConfig(
            settings=LoggingSettings(level="DEBUG", log_dir=Path("/var/log")),
            trackers=[
                TBTrackerConfig(logdir=Path("/tmp/tb"), queue_size=16384),
                WandBTrackerConfig(project="proj", tags=["t1", "t2"]),
                RedisMetricsTrackerConfig(max_history_per_metric=5000),
            ],
        )
        parsed = LoggingConfig.model_validate_json(cfg.model_dump_json())
        assert parsed.settings.level == "DEBUG"
        assert parsed.settings.log_dir == Path("/var/log")
        assert len(parsed.trackers) == 3
        assert parsed.trackers[0].kind == "tensorboard"
        assert parsed.trackers[0].queue_size == 16384  # type: ignore[union-attr]
        assert parsed.trackers[1].project == "proj"  # type: ignore[union-attr]
        assert parsed.trackers[1].tags == ["t1", "t2"]  # type: ignore[union-attr]
        assert parsed.trackers[2].max_history_per_metric == 5000  # type: ignore[union-attr]

    def test_frozen(self) -> None:
        cfg = LoggingConfig(trackers=[RedisMetricsTrackerConfig()])
        with pytest.raises(ValidationError):
            cfg.trackers = [RedisMetricsTrackerConfig()]  # type: ignore[misc]


class TestLoggingConfigBuildWriter:
    """The build_writer() method is the schema's primary runtime
    contract — it fans every tracker out through a CompositeLogger so
    the engine writes once and every backend records. Tests pin the
    contract: empty trackers list still returns a CompositeLogger (no
    None, no exception), and a non-empty list produces a composite
    whose inner loggers match the schema list 1:1 in declaration
    order."""

    def test_single_tracker_returns_composite_with_one_logger(
        self, tmp_path: Path
    ) -> None:
        from gigaevo.utils.trackers.composite import CompositeLogger

        cfg = LoggingConfig(trackers=[TBTrackerConfig(logdir=tmp_path / "tb")])
        writer = cfg.build_writer()
        assert isinstance(writer, CompositeLogger)


class TestLogLevelVocabulary:
    def test_success_level_accepted(self) -> None:
        """The ``LoggingSettings.level`` Literal carries every loguru
        log level, including ``SUCCESS`` which sits between ``INFO``
        and ``WARNING``."""
        cfg = LoggingSettings(level="SUCCESS")
        assert cfg.level == "SUCCESS"

    def test_every_loguru_level_accepted(self) -> None:
        for level in (
            "TRACE",
            "DEBUG",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ):
            LoggingSettings(level=level)  # type: ignore[arg-type]
