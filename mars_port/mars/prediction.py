"""Predictors for the API invoke-interval and execution time.

The original MARS used a LoRA-finetuned OPT-125m classifier to predict the
completion length before an API call. Those adapter weights do not exist in this
environment, so we default to an **oracle** predictor that returns the known
values from the workload (optionally perturbed by Gaussian noise to emulate
predictor error, matching the original benchmark's ``percentage_error_*`` knobs).

The :class:`Predictor` protocol keeps the door open to drop in the real model
later without touching call sites.
"""

from __future__ import annotations

import abc

import numpy as np


class Predictor(abc.ABC):
    """Predicts, at request arrival, the scheduling-relevant API quantities."""

    @abc.abstractmethod
    def predict_invoke_interval(self, *, actual: int | None = None) -> int:
        """Predicted #tokens generated before the next API call."""

    @abc.abstractmethod
    def predict_exec_time(self, *, actual: float | None = None) -> float:
        """Predicted API execution time (seconds)."""


class OraclePredictor(Predictor):
    """Returns the known/actual values (optionally with Gaussian error).

    With both error percentages 0 this is an exact oracle (predicted == actual),
    which is what we use while the real predictor weights are unavailable.
    """

    def __init__(
        self,
        *,
        interval_pct_error: float = 0.0,
        exec_time_pct_error: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.interval_pct_error = interval_pct_error
        self.exec_time_pct_error = exec_time_pct_error
        self._rng = np.random.default_rng(seed)

    def predict_invoke_interval(self, *, actual: int | None = None) -> int:
        if actual is None:
            raise ValueError("OraclePredictor requires the actual invoke interval")
        if self.interval_pct_error == 0.0:
            return int(actual)
        noisy = actual + self._rng.normal(0.0, self.interval_pct_error * actual)
        return max(1, int(noisy))

    def predict_exec_time(self, *, actual: float | None = None) -> float:
        if actual is None:
            raise ValueError("OraclePredictor requires the actual exec time")
        if self.exec_time_pct_error == 0.0:
            return float(actual)
        noisy = actual + self._rng.normal(0.0, self.exec_time_pct_error * actual)
        return max(1e-9, float(noisy))
