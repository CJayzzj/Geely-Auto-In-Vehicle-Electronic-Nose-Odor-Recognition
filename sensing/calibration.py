# -*- coding: utf-8 -*-
"""
sensing/calibration.py

Runtime Conc_Cal.xlsx loading and Conc==1 response gate checks.

This module intentionally keeps only the gate logic needed by perfume
recognition. The legacy alcohol concentration / odor-level interpolation
helpers are not part of enose_app's delivery scope.
"""

import math
import os

import numpy as np
import pandas as pd

from config.constants import (
    CONC2_RESP_GATE_X_CAP,
    INFERENCE_MIN_SENSORS_ABOVE_CONC2,
    RESP_X_CAP_CAL_CONC,
)
from sensing.paths import resolve_device_calibration_file


CONC_CALIBRATION_CATEGORY = "Conc_Cal"
CONC_CALIBRATION_FILENAME = "Conc_Cal.xlsx"
RESP_COLUMNS = [f"resp_Sensor{i}" for i in range(1, 17)]

_BASELINE_SIGMOID_RESP_PARAMS_CACHE = {}


def _default_x_caps():
    return [float(value) for value in CONC2_RESP_GATE_X_CAP]


def _result(x_caps, target_conc, path, loaded):
    return {
        "x_cap": [float(value) for value in x_caps],
        "x_cap_conc": float(target_conc),
        "path": path,
        "loaded": bool(loaded),
    }


def _copy_result(params):
    out = dict(params)
    out["x_cap"] = [float(value) for value in params.get("x_cap", [])]
    return out


def _sanitize_path_segment(text, fallback="default"):
    s = (text or "").strip()
    if not s:
        return fallback
    out = []
    for ch in s:
        if ch in '<>:"/\\|?*\x00' or ch == " ":
            out.append("_")
        else:
            out.append(ch)
    safe = "".join(out).strip("._")
    if not safe or safe in (".", ".."):
        return fallback
    return safe[:80]


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def conc_calibration_xlsx_path(device_segment_sanitized):
    """Return data/calibration/Conc_Cal/<device>/Conc_Cal.xlsx if it exists."""
    return resolve_device_calibration_file(
        CONC_CALIBRATION_CATEGORY,
        device_segment_sanitized,
        CONC_CALIBRATION_FILENAME,
    )


def load_baseline_sigmoid_resp_params(device_id_text, x_cap_conc=None):
    """
    Load per-sensor Conc==1 response gates from Conc_Cal.xlsx.

    x_cap for each channel is mean(abs(resp_SensorN)) over rows where Conc
    equals the target concentration. Missing files, missing columns, unreadable
    workbooks, and invalid channel data fall back to CONC2_RESP_GATE_X_CAP.
    Results are cached by absolute workbook path, mtime, and target Conc.
    """
    x_caps = _default_x_caps()
    target_conc = (
        float(RESP_X_CAP_CAL_CONC) if x_cap_conc is None else float(x_cap_conc)
    )

    raw = (device_id_text or "").strip()
    if not raw:
        return _result(x_caps, target_conc, None, False)

    device_segment = _sanitize_path_segment(raw, "default")
    path = conc_calibration_xlsx_path(device_segment)
    if not path:
        return _result(x_caps, target_conc, None, False)

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return _result(x_caps, target_conc, path, False)

    cache_key = (os.path.abspath(path), mtime, target_conc)
    cached = _BASELINE_SIGMOID_RESP_PARAMS_CACHE.get(cache_key)
    if cached is not None:
        return _copy_result(cached)

    try:
        df = pd.read_excel(path)
    except Exception:
        out = _result(x_caps, target_conc, path, False)
        _BASELINE_SIGMOID_RESP_PARAMS_CACHE[cache_key] = out
        return _copy_result(out)

    if "Conc" not in df.columns:
        out = _result(x_caps, target_conc, path, False)
        _BASELINE_SIGMOID_RESP_PARAMS_CACHE[cache_key] = out
        return _copy_result(out)

    conc = np.asarray(pd.to_numeric(df["Conc"], errors="coerce"), dtype=float)
    rows = df[np.isclose(conc, target_conc, rtol=0.0, atol=1e-9)]
    if rows.empty:
        out = _result(x_caps, target_conc, path, False)
        _BASELINE_SIGMOID_RESP_PARAMS_CACHE[cache_key] = out
        return _copy_result(out)

    loaded_any = False
    for idx, column in enumerate(RESP_COLUMNS):
        if column not in rows.columns:
            continue
        values = np.asarray(pd.to_numeric(rows[column], errors="coerce"), dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        x_cap = float(np.mean(np.abs(values)))
        if not math.isfinite(x_cap) or x_cap <= 0.0:
            continue
        x_caps[idx] = x_cap
        loaded_any = True

    out = _result(x_caps, target_conc, path, loaded_any)
    _BASELINE_SIGMOID_RESP_PARAMS_CACHE[cache_key] = out
    return _copy_result(out)


def count_sensors_above_conc2_resp_threshold(resp_values, device_id_text):
    """Count channels whose finite abs(resp) is greater than that channel's gate."""
    if resp_values is None:
        return 0
    params = load_baseline_sigmoid_resp_params(device_id_text or "")
    x_caps = params.get("x_cap") or _default_x_caps()
    count = 0
    for idx, resp in enumerate(resp_values):
        if idx >= len(x_caps):
            break
        value = _finite_float(resp)
        threshold = _finite_float(x_caps[idx])
        if value is None or threshold is None or threshold <= 0.0:
            continue
        if abs(value) > threshold:
            count += 1
    return count


def should_run_recognition_inference(resp_values, device_id_text):
    """Return True once enough channels are above their Conc==1 gates."""
    return (
        count_sensors_above_conc2_resp_threshold(resp_values, device_id_text)
        >= int(INFERENCE_MIN_SENSORS_ABOVE_CONC2)
    )


def build_conc2_resp_gate_status_fields(resp_values, device_id_text):
    """Fields passed into PeakDetector.update(gate_fields=...)."""
    return {
        "sensors_above_conc2": count_sensors_above_conc2_resp_threshold(
            resp_values, device_id_text
        ),
        "resp_gate_min_sensors": int(INFERENCE_MIN_SENSORS_ABOVE_CONC2),
    }


def conc2_resp_x_caps(device_id_text):
    """Return the per-channel Conc==1 gate list used by PeakDetector."""
    params = load_baseline_sigmoid_resp_params(device_id_text or "")
    return params.get("x_cap") or _default_x_caps()


def clear_calibration_cache():
    """Clear cached workbook-derived gate parameters."""
    _BASELINE_SIGMOID_RESP_PARAMS_CACHE.clear()
