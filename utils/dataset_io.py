# -*- coding: utf-8 -*-
"""Dataset loading, table editing, and CSV/XLSX persistence."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from algorithm.features import STANDARD_COLUMNS, build_training_feature_frame


SAMPLE_INDEX_COLUMN = "sample_index"
DATASET_EXTRA_COLUMNS = (SAMPLE_INDEX_COLUMN,)


class DatasetFormatInfo:
    def __init__(self, path=None, use_legacy_sensor_names=False, extension=".csv"):
        self.path = path
        self.use_legacy_sensor_names = use_legacy_sensor_names
        self.extension = extension


def _read_table(path: str) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
            try:
                return pd.read_csv(source, encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
        raise last_error
    if source.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(source)
    raise ValueError("不支持的数据集格式: {}".format(source.suffix))


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    frame, _ = build_training_feature_frame(df)
    for column in STANDARD_COLUMNS:
        if column not in frame.columns:
            frame[column] = "" if column in ("Time", "Timestamp", "Label") else np.nan
    return frame[list(STANDARD_COLUMNS)]


def load_dataset(path: str) -> Tuple[pd.DataFrame, DatasetFormatInfo]:
    frame = _standardize(_read_table(path))
    return frame, DatasetFormatInfo(
        path=path,
        extension=Path(path).suffix.lower(),
    )


def save_dataset(
    df: pd.DataFrame,
    path: str,
    format_info: Optional[DatasetFormatInfo] = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = _standardize(df)
    if target.suffix.lower() == ".csv":
        frame.to_csv(target, index=False, encoding="utf-8-sig")
    elif target.suffix.lower() in (".xlsx", ".xls"):
        frame.to_excel(target, index=False, engine="openpyxl")
    else:
        raise ValueError("不支持的数据集格式: {}".format(target.suffix))


def dataframe_from_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    return (
        pd.DataFrame(columns=STANDARD_COLUMNS)
        if not records
        else _standardize(pd.DataFrame(records))
    )


def pd_isna(value):
    return bool(pd.isna(value))


def dataset_column_order(columns: Iterable[str]):
    available = list(columns)
    ordered = [column for column in STANDARD_COLUMNS if column in available]
    extras = [column for column in available if column not in ordered]
    return list(DATASET_EXTRA_COLUMNS) + [
        column for column in ordered + extras if column not in DATASET_EXTRA_COLUMNS
    ]


def ensure_dataset_extra_columns(df):
    frame = df.copy()
    if SAMPLE_INDEX_COLUMN not in frame.columns:
        frame.insert(0, SAMPLE_INDEX_COLUMN, range(1, len(frame) + 1))
    return frame[dataset_column_order(frame.columns)]


def dataframe_from_records_with_extra(records):
    return ensure_dataset_extra_columns(dataframe_from_records(records))


def dataframe_append(df, records):
    if not records:
        return ensure_dataset_extra_columns(df)
    combined = pd.concat(
        [ensure_dataset_extra_columns(df), dataframe_from_records_with_extra(records)],
        ignore_index=True,
    )
    combined[SAMPLE_INDEX_COLUMN] = range(1, len(combined) + 1)
    return combined


def load_dataset_with_extra_columns(path):
    raw = _read_table(path)
    extras = raw[[column for column in DATASET_EXTRA_COLUMNS if column in raw.columns]]
    standard, info = load_dataset(path)
    for column in extras.columns:
        standard[column] = extras[column].values
    return ensure_dataset_extra_columns(standard), info


def normalize_dataset_save_path(path, selected_filter=None):
    target = str(path or "").strip()
    if not target:
        return ""
    suffix = Path(target).suffix.lower()
    if suffix not in (".xlsx", ".xls", ".csv"):
        target += ".xlsx"
    return target


def save_dataset_with_extra_columns(df, path, format_info=None):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = ensure_dataset_extra_columns(df)
    if target.suffix.lower() == ".csv":
        frame.to_csv(target, index=False, encoding="utf-8-sig")
    else:
        frame.to_excel(target, index=False, engine="openpyxl")
    return str(target)
