"""Eval-time predictors for program scheduling.

Each predictor estimates how long a program's DAG evaluation will take,
using features available before evaluation starts.  Predictions drive
scheduling decisions (LPT, SJF, etc.).

Predictors learn online from completed evaluations via ``update()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
import math
import statistics
import threading
from typing import TYPE_CHECKING

from loguru import logger

from gigaevo.evolution.scheduling.feature_extractor import (
    CodeFeatureExtractor,
    FeatureExtractor,
)

if TYPE_CHECKING:
    from gigaevo.programs.program import Program


class EvalTimePredictor(ABC):
    """Predict evaluation wall-clock time for a Program.

    Contract:
    - ``predict()`` returns seconds (float).  During cold start, return a
      reasonable default (NOT zero — zero defeats LPT).
    - ``update()`` is called after evaluation completes.
    - Thread/async safety: ``predict()`` and ``update()`` may be called
      concurrently.  Implementations must handle this.
    """

    @abstractmethod
    def predict(self, program: Program) -> float:
        """Predicted eval time in seconds.  Must not block on I/O."""

    @abstractmethod
    def update(self, program: Program, actual_duration: float) -> None:
        """Online learning callback after program evaluation completes."""

    @abstractmethod
    def is_warm(self) -> bool:
        """True if enough data for meaningful predictions."""


class ConstantPredictor(EvalTimePredictor):
    """Return a fixed constant.  All programs equal => FIFO behavior.

    This is the identity element for the scheduling system.
    """

    def __init__(self, constant: float = 1.0) -> None:
        self._constant = constant

    def predict(self, program: Program) -> float:
        return self._constant

    def update(self, program: Program, actual_duration: float) -> None:
        pass

    def is_warm(self) -> bool:
        return True


class SimpleHeuristicPredictor(EvalTimePredictor):
    """Predict eval time as a linear function of code length.

    Uses an online running average of ``actual_time / code_length`` to
    calibrate.  No sklearn dependency.  Suitable for cold-start.

    Rationale: longer code => more tokens for LLM-based stages => longer
    eval.  Crude but robust first-order approximation.
    """

    _MIN_CODE_LENGTH = 100

    def __init__(
        self,
        *,
        default_rate: float = 0.1,
        window_size: int = 50,
    ) -> None:
        self._default_rate = default_rate
        self._window: deque[float] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def predict(self, program: Program) -> float:
        code_len = max(len(program.code), self._MIN_CODE_LENGTH)
        with self._lock:
            if self._window:
                rate = statistics.median(self._window)
            else:
                rate = self._default_rate
        return code_len * rate

    def update(self, program: Program, actual_duration: float) -> None:
        if actual_duration <= 0:
            return
        code_len = max(len(program.code), self._MIN_CODE_LENGTH)
        rate = actual_duration / code_len
        # Clip extreme outliers (>10x median) to prevent poisoning
        with self._lock:
            if self._window:
                med = statistics.median(self._window)
                if med > 0 and rate > 10 * med:
                    rate = 10 * med
            self._window.append(rate)

    def is_warm(self) -> bool:
        with self._lock:
            return len(self._window) >= 5


class RidgePredictor(EvalTimePredictor):
    """Online Ridge regression over user-defined features.

    Retrains on the full replay buffer (bounded) after each update.
    sklearn is a soft dependency — imported lazily.

    For typical workloads (buffer_size=500, 4-10 features), retrain
    cost is <1ms.
    """

    def __init__(
        self,
        *,
        feature_extractor: FeatureExtractor | None = None,
        buffer_size: int = 500,
        min_samples: int = 10,
        default_prediction: float = 300.0,
        alpha: float = 1.0,
    ) -> None:
        self._extractor = feature_extractor or CodeFeatureExtractor()
        self._buffer_size = buffer_size
        self._min_samples = min_samples
        self._default_prediction = default_prediction
        self._alpha = alpha

        self._buffer: deque[tuple[dict[str, float], float]] = deque(maxlen=buffer_size)
        self._model = None
        self._feature_keys: list[str] | None = None
        self._lock = threading.Lock()
        self._sklearn_available: bool | None = None  # None = not checked yet

    def predict(self, program: Program) -> float:
        with self._lock:
            if self._model is None or self._feature_keys is None:
                return max(
                    self._default_prediction,
                    float(len(program.code)) * 0.1,
                )
            features = self._extractor.extract(program)
            x = [features.get(k, 0.0) for k in self._feature_keys]
            pred = float(self._model.predict([x])[0])
            # Guard against NaN/Inf from degenerate training data
            if not math.isfinite(pred) or pred < 1.0:
                return self._default_prediction
            return pred

    def update(self, program: Program, actual_duration: float) -> None:
        if actual_duration <= 0:
            return
        features = self._extractor.extract(program)
        with self._lock:
            self._buffer.append((features, actual_duration))
            if len(self._buffer) >= self._min_samples:
                self._retrain()

    def is_warm(self) -> bool:
        with self._lock:
            return self._model is not None

    def _retrain(self) -> None:
        """Retrain Ridge on full buffer.  Called under lock."""
        if self._sklearn_available is False:
            return
        try:
            from sklearn.linear_model import Ridge
        except ImportError:
            if self._sklearn_available is None:
                logger.warning(
                    "[RidgePredictor] sklearn not available; "
                    "falling back to default predictions"
                )
            self._sklearn_available = False
            return
        self._sklearn_available = True

        if not self._buffer:
            return

        # Recompute feature keys from union of all buffer entries
        all_keys: set[str] = set()
        for feat, _ in self._buffer:
            all_keys.update(feat.keys())
        self._feature_keys = sorted(all_keys)

        X = [[feat.get(k, 0.0) for k in self._feature_keys] for feat, _ in self._buffer]
        y = [t for _, t in self._buffer]

        model = Ridge(alpha=self._alpha)
        model.fit(X, y)
        self._model = model

        # Log retrain event with feature weights
        weights = dict(zip(self._feature_keys, model.coef_))
        top_features = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)
        weight_str = ", ".join(f"{k}={v:+.3f}" for k, v in top_features[:5])
        logger.info(
            "[RidgePredictor] Retrained on {} samples | intercept={:.1f}s | "
            "top weights: {}",
            len(y),
            model.intercept_,
            weight_str,
        )
