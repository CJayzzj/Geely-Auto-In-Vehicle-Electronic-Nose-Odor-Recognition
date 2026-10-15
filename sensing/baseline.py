# -*- coding: utf-8 -*-
"""Adaptive EMA baseline tracking for compensated sensor resistances."""

import math

from config.constants import (
    BASELINE_INITIAL_ALPHA,
    BASELINE_INITIAL_ALPHA_DURATION_S,
    BASELINE_SIGMOID_ALPHA_HI_DEFAULT,
    BASELINE_SIGMOID_ALPHA_LO_DEFAULT,
    BASELINE_SIGMOID_K_DEFAULT,
    BASELINE_SIGMOID_X_C,
    BASELINE_SIGMOID_X_CAP,
    BASELINE_STUCK_RECOVERY_DURATION_S,
)


def compute_baseline_alpha_sigmoid(x, alpha_hi, alpha_lo, k, x_c, x_cap):
    """Map a relative residual to the adaptive EMA alpha."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return float(alpha_lo)
    if not math.isfinite(x) or x < 0.0:
        return float(alpha_lo)
    if x >= float(x_cap):
        return float(alpha_lo)

    z = float(k) * (x - float(x_c))
    if z > 50.0:
        return float(alpha_lo)
    if z < -50.0:
        return float(alpha_hi)
    return float(alpha_lo + (alpha_hi - alpha_lo) / (1.0 + math.exp(z)))


class BaselineTracker:
    """Track an independent compensated-resistance baseline for each channel."""

    def __init__(
        self,
        num_channels=16,
        sigmoid_params=None,
        initial_alpha_duration_s=BASELINE_INITIAL_ALPHA_DURATION_S,
        initial_alpha=BASELINE_INITIAL_ALPHA,
        stuck_recovery_duration_s=BASELINE_STUCK_RECOVERY_DURATION_S,
    ):
        self.num_channels = int(num_channels)
        if self.num_channels <= 0:
            raise ValueError("num_channels must be greater than zero")

        params = {
            "k": float(BASELINE_SIGMOID_K_DEFAULT),
            "alpha_hi": float(BASELINE_SIGMOID_ALPHA_HI_DEFAULT),
            "alpha_lo": float(BASELINE_SIGMOID_ALPHA_LO_DEFAULT),
            "x_c": float(BASELINE_SIGMOID_X_C),
            "x_cap": float(BASELINE_SIGMOID_X_CAP),
        }
        if sigmoid_params:
            params.update(sigmoid_params)
        self.sigmoid_params = self._validated_sigmoid_params(params)

        self.initial_alpha_duration_s = float(initial_alpha_duration_s)
        self.initial_alpha = float(initial_alpha)
        self.stuck_recovery_duration_s = float(stuck_recovery_duration_s)
        if not math.isfinite(self.initial_alpha_duration_s) or self.initial_alpha_duration_s < 0.0:
            raise ValueError("initial_alpha_duration_s must not be negative")
        if not math.isfinite(self.initial_alpha) or not 0.0 <= self.initial_alpha <= 1.0:
            raise ValueError("initial_alpha must be between zero and one")
        if (
            not math.isfinite(self.stuck_recovery_duration_s)
            or self.stuck_recovery_duration_s < 0.0
        ):
            raise ValueError("stuck_recovery_duration_s must not be negative")

        self.baseline_values = [None] * self.num_channels
        self.baseline_start_times = [None] * self.num_channels
        self.stuck_since_ts = [None] * self.num_channels
        self.invalid_baseline_recovery_counts = [0] * self.num_channels
        self.system_start_time = None

    @staticmethod
    def _validated_sigmoid_params(params):
        required = ("k", "alpha_hi", "alpha_lo", "x_c", "x_cap")
        try:
            out = {key: float(params[key]) for key in required}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid sigmoid_params") from exc
        if not all(math.isfinite(value) for value in out.values()):
            raise ValueError("sigmoid_params must contain finite values")
        if out["k"] < 0.0:
            raise ValueError("sigmoid k must not be negative")
        if not 0.0 <= out["alpha_lo"] <= out["alpha_hi"] <= 1.0:
            raise ValueError("sigmoid alphas must satisfy 0 <= alpha_lo <= alpha_hi <= 1")
        if not 0.0 <= out["x_c"] < out["x_cap"]:
            raise ValueError("sigmoid thresholds must satisfy 0 <= x_c < x_cap")
        return out

    def update(self, sensor_idx, value, timestamp):
        """Update one channel and return ``(baseline, alpha_used)``."""
        idx = self._validated_sensor_idx(sensor_idx)
        timestamp = self._finite_timestamp(timestamp)
        if self.system_start_time is None:
            self.system_start_time = timestamp

        try:
            value = float(value)
        except (TypeError, ValueError):
            return self.baseline_values[idx], None
        if not math.isfinite(value):
            return self.baseline_values[idx], None

        baseline = self.baseline_values[idx]
        if baseline is None:
            self.baseline_values[idx] = value
            self.baseline_start_times[idx] = timestamp
            self.stuck_since_ts[idx] = None
            return value, self.initial_alpha

        params = self.sigmoid_params
        if not math.isfinite(baseline):
            self.baseline_values[idx] = value
            self.baseline_start_times[idx] = timestamp
            self.stuck_since_ts[idx] = None
            self.invalid_baseline_recovery_counts[idx] += 1
            return value, self.initial_alpha

        if (timestamp - self.system_start_time) < self.initial_alpha_duration_s:
            self.stuck_since_ts[idx] = None
            return self._apply_alpha(idx, value, baseline, self.initial_alpha)

        if baseline == 0.0:
            self.stuck_since_ts[idx] = None
            return self._apply_alpha(idx, value, baseline, params["alpha_lo"])

        x = abs((value - baseline) / baseline)
        if x >= params["x_cap"]:
            since = self.stuck_since_ts[idx]
            if since is None or timestamp < since:
                since = timestamp
                self.stuck_since_ts[idx] = timestamp
            stuck_duration = timestamp - since
            alpha = (
                self.initial_alpha
                if stuck_duration >= self.stuck_recovery_duration_s
                else params["alpha_lo"]
            )
        else:
            self.stuck_since_ts[idx] = None
            alpha = compute_baseline_alpha_sigmoid(
                x,
                params["alpha_hi"],
                params["alpha_lo"],
                params["k"],
                params["x_c"],
                params["x_cap"],
            )
        return self._apply_alpha(idx, value, baseline, alpha)

    def reset(self):
        """Clear all channel state, for example after a serial reconnect."""
        self.baseline_values = [None] * self.num_channels
        self.baseline_start_times = [None] * self.num_channels
        self.stuck_since_ts = [None] * self.num_channels
        self.invalid_baseline_recovery_counts = [0] * self.num_channels
        self.system_start_time = None

    def _validated_sensor_idx(self, sensor_idx):
        try:
            idx = int(sensor_idx)
        except (TypeError, ValueError) as exc:
            raise IndexError("sensor_idx is invalid") from exc
        if idx != sensor_idx or idx < 0 or idx >= self.num_channels:
            raise IndexError("sensor_idx is out of range")
        return idx

    @staticmethod
    def _finite_timestamp(timestamp):
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp must be finite") from exc
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        return timestamp

    def _apply_alpha(self, idx, value, baseline, alpha):
        updated = baseline + float(alpha) * (value - baseline)
        self.baseline_values[idx] = updated
        return updated, float(alpha)
