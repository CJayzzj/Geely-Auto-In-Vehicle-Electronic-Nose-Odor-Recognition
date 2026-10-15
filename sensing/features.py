# -*- coding: utf-8 -*-
"""Build response and normalized pattern features from sensor resistances."""

import math


def _is_finite_number(value):
    try:
        return math.isfinite(value)
    except (TypeError, ValueError):
        return False


def build_feature_vectors(resistance_16_compensated, baseline_values):
    """Return signed response and max-absolute normalized pattern vectors.

    Both inputs contain compensated-resistance values in the same domain. A
    channel with an invalid value or baseline contributes ``0.0`` so non-finite
    values cannot propagate to gating, peak detection, or model inference.
    """
    resp_values = []
    for idx, value_r in enumerate(resistance_16_compensated):
        baseline_r = baseline_values[idx]
        if (
            baseline_r is None
            or not _is_finite_number(baseline_r)
            or baseline_r == 0
            or not _is_finite_number(value_r)
        ):
            resp_values.append(0.0)
            continue

        resp = (value_r - baseline_r) / abs(baseline_r)
        resp_values.append(float(resp) if _is_finite_number(resp) else 0.0)

    max_abs_resp = max((abs(value) for value in resp_values), default=0.0)
    if max_abs_resp == 0.0:
        pattern_values = [0.0 for _ in resp_values]
    else:
        pattern_values = [value / max_abs_resp for value in resp_values]

    return resp_values, pattern_values
