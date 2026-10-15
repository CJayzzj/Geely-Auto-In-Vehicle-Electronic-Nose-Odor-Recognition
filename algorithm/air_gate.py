# -*- coding: utf-8 -*-
"""Stage0 air gate based on the device's concentration calibration workbook."""

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from utils.paths import CONC_CAL_DIR, sanitize_path_segment

from .features import RESP_COLUMNS, SENSOR_COUNT


RESP_X_CAP_CAL_CONC = 1.0
INFERENCE_RESP_GATE_TOP_K = 3
INFERENCE_RESP_GATE_SCORE_THRESHOLD = 1.0
DEFAULT_RESP_GATE_X_CAP = 0.1


def resolve_calibration_path(
    device_id_text: str = "", calibration_path: Optional[str] = None
) -> Optional[Path]:
    if calibration_path:
        candidate = Path(calibration_path)
        return candidate if candidate.is_file() else None
    device = sanitize_path_segment(device_id_text, "")
    candidate = Path(CONC_CAL_DIR) / device / "Conc_Cal.xlsx"
    return candidate if device and candidate.is_file() else None


def load_air_gate_calibration(
    device_id_text: str = "", calibration_path: Optional[str] = None
) -> Dict[str, Any]:
    path = resolve_calibration_path(device_id_text, calibration_path)
    fallback = [DEFAULT_RESP_GATE_X_CAP] * SENSOR_COUNT
    if path is None:
        return {
            "x_cap": fallback,
            "x_cap_conc": RESP_X_CAP_CAL_CONC,
            "path": None,
            "loaded": False,
        }
    try:
        frame = pd.read_excel(path)
    except Exception:
        return {
            "x_cap": fallback,
            "x_cap_conc": RESP_X_CAP_CAL_CONC,
            "path": str(path),
            "loaded": False,
        }
    if "Conc" not in frame.columns:
        return {
            "x_cap": fallback,
            "x_cap_conc": RESP_X_CAP_CAL_CONC,
            "path": str(path),
            "loaded": False,
        }
    concentration = pd.to_numeric(frame["Conc"], errors="coerce").values.astype(float)
    rows = frame.loc[
        np.isclose(concentration, RESP_X_CAP_CAL_CONC, rtol=0.0, atol=1e-9)
    ]
    caps = list(fallback)
    loaded_any = False
    for index, column in enumerate(RESP_COLUMNS):
        if column not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors="coerce").values.astype(float)
        values = values[np.isfinite(values)]
        if values.size:
            cap = float(np.mean(np.abs(values)))
            if np.isfinite(cap) and cap > 0.0:
                caps[index] = cap
                loaded_any = True
    return {
        "x_cap": caps,
        "x_cap_conc": RESP_X_CAP_CAL_CONC,
        "path": str(path),
        "loaded": loaded_any,
    }


def resp_gate_score_from_resp_values(
    resp_values: Iterable[float],
    device_id_text: str = "",
    calibration_path: Optional[str] = None,
) -> float:
    values = np.asarray(list(resp_values), dtype=float)
    calibration = load_air_gate_calibration(device_id_text, calibration_path)
    thresholds = np.asarray(calibration["x_cap"], dtype=float)
    usable = min(values.size, thresholds.size)
    values, thresholds = values[:usable], thresholds[:usable]
    valid = np.isfinite(values) & np.isfinite(thresholds) & (thresholds > 0.0)
    if not np.any(valid):
        return float("nan")
    ratios = np.abs(values[valid]) / thresholds[valid]
    top_k = min(INFERENCE_RESP_GATE_TOP_K, ratios.size)
    return float(np.mean(np.sort(ratios)[-top_k:]))


def should_run_recognition_inference(
    resp_values: Iterable[float],
    device_id_text: str = "",
    calibration_path: Optional[str] = None,
) -> bool:
    score = resp_gate_score_from_resp_values(
        resp_values, device_id_text, calibration_path
    )
    return bool(
        np.isfinite(score) and score >= INFERENCE_RESP_GATE_SCORE_THRESHOLD
    )
