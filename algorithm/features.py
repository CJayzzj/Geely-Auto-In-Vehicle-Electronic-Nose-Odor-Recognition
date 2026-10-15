# -*- coding: utf-8 -*-
"""Feature engineering shared by training, offline tests, and realtime inference."""

from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ADC_MAX = 4096.0
SENSOR_COUNT = 16
SENSOR_COLUMNS = ["Sensor{}".format(i) for i in range(1, SENSOR_COUNT + 1)]
LEGACY_SENSOR_COLUMNS = [
    "Sensor{}_Resistance".format(i) for i in range(1, SENSOR_COUNT + 1)
]
ADC_COLUMNS = ["ADC_Sensor{}".format(i) for i in range(1, SENSOR_COUNT + 1)]
R_RAW_COLUMNS = ["R_raw_Sensor{}".format(i) for i in range(1, SENSOR_COUNT + 1)]
R_COMP_COLUMNS = ["R_comp_Sensor{}".format(i) for i in range(1, SENSOR_COUNT + 1)]
LEGACY_R_SENSOR_COLUMNS = ["R_Sensor{}".format(i) for i in range(1, SENSOR_COUNT + 1)]
BASELINE_COLUMNS = ["baseline_Sensor{}".format(i) for i in range(1, SENSOR_COUNT + 1)]
RESP_COLUMNS = ["resp_Sensor{}".format(i) for i in range(1, SENSOR_COUNT + 1)]
CENTERED_RESP_COLUMNS = [
    "centered_resp_Sensor{}".format(i) for i in range(1, SENSOR_COUNT + 1)
]
PATTERN_COLUMNS = ["pattern_Sensor{}".format(i) for i in range(1, SENSOR_COUNT + 1)]
GROUP_ZSCORE_RESP_COLUMNS = [
    "group_zscore_resp_Sensor{}".format(i) for i in range(1, SENSOR_COUNT + 1)
]
RESPONSE_STRENGTH_SCORE_COLUMN = "score"
RESPONSE_STRENGTH_THRESHOLDS_COLUMN = "score_thresholds"
MATERIAL_RESP_GROUPS = ((0, 4), (4, 8), (8, 12), (12, 16))
DEFAULT_FEATURE_MODE = "material_group_zscore_resp"
DEFAULT_BASELINE_MODE = "online_adaptive_baseline"
BASELINE_FAST_DURATION_S = 5 * 60
BASELINE_FAST_ALPHA = 0.08
BASELINE_SLOW_ALPHA = 0.003

STANDARD_COLUMNS = (
    ["Time"]
    + ADC_COLUMNS
    + R_RAW_COLUMNS
    + R_COMP_COLUMNS
    + BASELINE_COLUMNS
    + RESP_COLUMNS
    + CENTERED_RESP_COLUMNS
    + PATTERN_COLUMNS
    + GROUP_ZSCORE_RESP_COLUMNS
    + [RESPONSE_STRENGTH_SCORE_COLUMN, RESPONSE_STRENGTH_THRESHOLDS_COLUMN]
    + ["Temperature", "Humidity", "Timestamp", "Label"]
)


def adc_to_resistance(values: Any) -> Any:
    if isinstance(values, (pd.DataFrame, pd.Series)):
        clipped = values.astype(float).clip(lower=0, upper=ADC_MAX - 1e-6)
        return 100.0 * clipped / (ADC_MAX - clipped)
    value = float(values)
    if value >= ADC_MAX:
        return float("inf")
    return 100.0 * value / (ADC_MAX - value)


def compute_material_centered_resp_values(
    resp_values: Iterable[float],
) -> Optional[np.ndarray]:
    values = np.asarray(list(resp_values), dtype=float)
    if values.size != SENSOR_COUNT:
        return None
    values = np.where(np.isfinite(values), values, 0.0)
    centered = values.copy()
    for start, end in MATERIAL_RESP_GROUPS:
        group = values[start:end]
        centered[start:end] = group - float(np.mean(group))
    return centered


def compute_material_group_zscore_resp_values(
    resp_values: Iterable[float],
) -> Optional[np.ndarray]:
    values = np.asarray(list(resp_values), dtype=float)
    if values.size != SENSOR_COUNT:
        return None
    values = np.where(np.isfinite(values), values, 0.0)
    transformed = np.zeros_like(values)
    for start, end in MATERIAL_RESP_GROUPS:
        group = values[start:end]
        std = float(np.std(group))
        transformed[start:end] = (
            (group - float(np.mean(group))) / std if std > 1e-12 else 0.0
        )
    return transformed


def compute_pattern_values(resp_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(resp_values), dtype=float)
    if values.size != SENSOR_COUNT:
        raise ValueError("resp特征需要16维，当前为{}维。".format(values.size))
    values = np.where(np.isfinite(values), values, 0.0)
    maximum = float(np.max(np.abs(values)))
    return values / maximum if maximum > 0.0 else np.zeros_like(values)


def build_feature_vector(
    resp_values: Iterable[float], pattern_values: Optional[Iterable[float]] = None
) -> np.ndarray:
    resp = np.asarray(list(resp_values), dtype=float)
    pattern = (
        compute_pattern_values(resp)
        if pattern_values is None
        else np.asarray(list(pattern_values), dtype=float)
    )
    return np.concatenate([resp, pattern]).astype(float)


def normalize_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}
    for sensor, legacy, adc in zip(
        SENSOR_COLUMNS, LEGACY_SENSOR_COLUMNS, ADC_COLUMNS
    ):
        if legacy in out.columns and adc not in out.columns:
            rename_map[legacy] = adc
        elif sensor in out.columns and adc not in out.columns:
            rename_map[sensor] = adc
    for legacy, comp in zip(LEGACY_R_SENSOR_COLUMNS, R_COMP_COLUMNS):
        if legacy in out.columns and comp not in out.columns:
            rename_map[legacy] = comp
    if rename_map:
        out = out.rename(columns=rename_map)
    if "Label" not in out.columns:
        out["Label"] = ""
    return out


def _numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: (
                pd.to_numeric(df[column], errors="coerce")
                if column in df.columns
                else pd.Series(np.nan, index=df.index, dtype=float)
            )
            for column in columns
        },
        index=df.index,
    )


def _elapsed_seconds(df: pd.DataFrame) -> pd.Series:
    for column in ("Timestamp", "Time"):
        if column not in df.columns:
            continue
        parsed = pd.to_datetime(df[column], errors="coerce")
        if parsed.notna().sum() >= 2:
            base = parsed.dropna().iloc[0]
            return (parsed - base).dt.total_seconds().ffill().fillna(0.0)
    return pd.Series(np.arange(len(df), dtype=float), index=df.index)


def _build_online_resp_from_adc(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    adc = _numeric_frame(df, ADC_COLUMNS)
    elapsed = _elapsed_seconds(df)
    baselines = [None] * SENSOR_COUNT
    starts = [None] * SENSOR_COUNT
    resp_rows, baseline_rows = [], []
    for row_pos, (_, adc_row) in enumerate(adc.iterrows()):
        current = adc_row.values.astype(float)
        now = float(elapsed.iloc[row_pos])
        baseline_adc = []
        for sensor_idx, value in enumerate(current):
            if not np.isfinite(value):
                baseline_adc.append(
                    np.nan if baselines[sensor_idx] is None else baselines[sensor_idx]
                )
                continue
            if baselines[sensor_idx] is None:
                baselines[sensor_idx] = float(value)
                starts[sensor_idx] = now
            else:
                age = now - float(starts[sensor_idx] or now)
                alpha = (
                    BASELINE_FAST_ALPHA
                    if age < BASELINE_FAST_DURATION_S
                    else BASELINE_SLOW_ALPHA
                )
                baselines[sensor_idx] += alpha * (
                    float(value) - baselines[sensor_idx]
                )
            baseline_adc.append(float(baselines[sensor_idx]))
        row_resp, row_baseline = [], []
        for value, base_adc in zip(current, baseline_adc):
            if not np.isfinite(value) or not np.isfinite(base_adc):
                row_resp.append(np.nan)
                row_baseline.append(np.nan)
                continue
            current_r = adc_to_resistance(value)
            baseline_r = adc_to_resistance(base_adc)
            row_baseline.append(float(baseline_r))
            row_resp.append(
                np.nan
                if not np.isfinite(current_r)
                or not np.isfinite(baseline_r)
                or baseline_r == 0
                else float((current_r - baseline_r) / abs(baseline_r))
            )
        resp_rows.append(row_resp)
        baseline_rows.append(row_baseline)
    return (
        pd.DataFrame(resp_rows, columns=RESP_COLUMNS, index=df.index),
        pd.DataFrame(baseline_rows, columns=BASELINE_COLUMNS, index=df.index),
    )


def build_training_feature_frame(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = normalize_dataset_columns(df)
    existing_resp = _numeric_frame(out, RESP_COLUMNS)
    existing_baseline = _numeric_frame(out, BASELINE_COLUMNS)
    computed_resp = pd.DataFrame(np.nan, index=out.index, columns=RESP_COLUMNS)
    computed_baseline = pd.DataFrame(
        np.nan, index=out.index, columns=BASELINE_COLUMNS
    )
    if any(column in out.columns for column in ADC_COLUMNS):
        computed_resp, computed_baseline = _build_online_resp_from_adc(out)

    r_comp = _numeric_frame(out, R_COMP_COLUMNS)
    baseline = existing_baseline.where(existing_baseline.notna(), computed_baseline)
    denominator = baseline.abs().replace(0.0, np.nan)
    resistance_resp = (r_comp - baseline.values) / denominator.values
    resistance_resp.columns = RESP_COLUMNS
    resp = existing_resp.where(existing_resp.notna(), resistance_resp)
    resp = resp.where(resp.notna(), computed_resp)
    for column in RESP_COLUMNS:
        out[column] = resp[column]
    for column in BASELINE_COLUMNS:
        out[column] = baseline[column]

    values = np.where(np.isfinite(resp.values.astype(float)), resp.values, 0.0)
    maximum = np.max(np.abs(values), axis=1, keepdims=True)
    pattern = np.divide(
        values, maximum, out=np.zeros_like(values), where=maximum > 0.0
    )
    centered = np.zeros_like(values)
    group_zscore = np.zeros_like(values)
    for start, end in MATERIAL_RESP_GROUPS:
        group = values[:, start:end]
        mean = group.mean(axis=1, keepdims=True)
        std = group.std(axis=1, keepdims=True)
        centered[:, start:end] = group - mean
        group_zscore[:, start:end] = np.divide(
            group - mean,
            std,
            out=np.zeros_like(group),
            where=std > 1e-12,
        )
    valid = resp.notna().any(axis=1).values
    for index, column in enumerate(PATTERN_COLUMNS):
        out[column] = np.where(valid, pattern[:, index], np.nan)
    for index, column in enumerate(CENTERED_RESP_COLUMNS):
        out[column] = np.where(valid, centered[:, index], np.nan)
    for index, column in enumerate(GROUP_ZSCORE_RESP_COLUMNS):
        out[column] = np.where(valid, group_zscore[:, index], np.nan)
    return out, baseline


def extract_model_feature_vector(model_bundle, feature_vector) -> np.ndarray:
    vector = np.asarray(list(feature_vector), dtype=float)
    spec = (model_bundle or {}).get("feature_spec", {}) or {}
    columns = list(spec.get("feature_columns") or [])
    if (
        spec.get("feature_mode") == DEFAULT_FEATURE_MODE
        or columns == GROUP_ZSCORE_RESP_COLUMNS
    ):
        if vector.size == SENSOR_COUNT:
            return vector
        if vector.size >= SENSOR_COUNT * 2:
            converted = compute_material_group_zscore_resp_values(
                vector[:SENSOR_COUNT]
            )
            if converted is not None:
                return converted
    return vector
