# -*- coding: utf-8 -*-
"""One shared LinearSVC implementation for both binary products.

Keep this shared while both tasks use the same feature columns and SVM
hyperparameters. Split it only if one task needs a different C value, a
different feature family, or task-specific algorithm behavior that would
otherwise change the other product.
"""

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

from task_config import validate_task_config

from .features import (
    DEFAULT_BASELINE_MODE,
    DEFAULT_FEATURE_MODE,
    GROUP_ZSCORE_RESP_COLUMNS,
    RESP_COLUMNS,
    build_feature_vector,
    build_training_feature_frame,
    extract_model_feature_vector,
)


ALGORITHM_VERSION = "JL-v4.6-closed-binary-1"
ODOR_LABEL_ALIASES = {
    "臭豆腐": "异味",
    "异常气味": "异味",
    "abnormal odor": "异味",
    "abnormal_odor": "异味",
}


def _clean_label(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _canonical_source_label(value: Any) -> str:
    label = _clean_label(value)
    return ODOR_LABEL_ALIASES.get(label.lower(), ODOR_LABEL_ALIASES.get(label, label))


def apply_task_labels(
    dataset_df: pd.DataFrame,
    task_config: Dict[str, Any],
    keep_only_task_classes: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Apply the task's one-time label merge when a table is loaded."""
    config = validate_task_config(task_config)
    frame = dataset_df.copy()
    if "Label" not in frame.columns:
        frame["Label"] = ""
    original = frame["Label"].apply(_clean_label)
    canonical = frame["Label"].apply(_canonical_source_label)
    merge_rule = config.get("label_merge_rule") or {}
    mapped = canonical.map(lambda label: merge_rule.get(label, label))
    frame["Label"] = mapped
    class_set = set(config["class_labels"])
    stats = {
        "rows": int(len(frame)),
        "merged_rows": int((mapped != original).sum()),
        "air_rows": int(mapped.eq("空气").sum()),
        "ignored_rows": int((~mapped.isin(class_set | {"空气"})).sum()),
    }
    if keep_only_task_classes:
        frame = frame.loc[mapped.isin(class_set)].copy()
    return frame, stats


def train_group_zscore_resp_linear_model(
    train_df: pd.DataFrame,
    task_config: Dict[str, Any],
    test_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    config = validate_task_config(task_config)
    normalized_train, mapping_stats = apply_task_labels(
        train_df, config, keep_only_task_classes=True
    )
    feature_df, baseline_df = build_training_feature_frame(normalized_train)
    labels = feature_df["Label"].apply(_clean_label)
    values = feature_df[GROUP_ZSCORE_RESP_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = labels.isin(config["class_labels"]) & np.isfinite(values.values).all(axis=1)
    feature_df = feature_df.loc[valid].copy()
    baseline_df = baseline_df.loc[valid].copy()
    labels = labels.loc[valid]
    values = values.loc[valid]
    present = set(labels.unique())
    expected = set(config["class_labels"])
    if present != expected:
        raise ValueError(
            "训练数据必须同时包含{}，当前有效标签为{}。".format(
                "/".join(config["class_labels"]),
                "/".join(sorted(present)) if present else "无",
            )
        )

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels.values)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "linear_svm",
                LinearSVC(C=1.0, class_weight="balanced", max_iter=20000),
            ),
        ]
    )
    model.fit(values.values.astype(float), encoded)
    bundle: Dict[str, Any] = {
        "algorithm_version": ALGORITHM_VERSION,
        "task_key": config.get("task_key", ""),
        "task_config": config,
        "svm_model": model,
        "label_encoder": encoder,
        "feature_spec": {
            "feature_mode": DEFAULT_FEATURE_MODE,
            "baseline_mode": DEFAULT_BASELINE_MODE,
            "feature_columns": list(GROUP_ZSCORE_RESP_COLUMNS),
        },
        "sensor_columns": list(RESP_COLUMNS),
        "class_names": list(config["class_labels"]),
        "train_summary": {
            "train_samples": int(len(feature_df)),
            "train_class_count": 2,
            "train_labels": list(config["class_labels"]),
            "mapping_stats": mapping_stats,
            "feature_mode": DEFAULT_FEATURE_MODE,
            "feature_columns": list(GROUP_ZSCORE_RESP_COLUMNS),
        },
        "evaluation": {},
        "train_feature_df": feature_df,
        "train_baseline_df": baseline_df,
    }
    if test_df is not None and not test_df.empty:
        normalized_test, _ = apply_task_labels(
            test_df, config, keep_only_task_classes=True
        )
        test_feature, _ = build_training_feature_frame(normalized_test)
        test_values = test_feature[GROUP_ZSCORE_RESP_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        )
        test_labels = test_feature["Label"].apply(_clean_label)
        valid_test = test_labels.isin(config["class_labels"]) & np.isfinite(
            test_values.values
        ).all(axis=1)
        if valid_test.any():
            truth = encoder.transform(test_labels.loc[valid_test])
            predicted = model.predict(test_values.loc[valid_test].values)
            bundle["evaluation"] = {
                "accuracy": float(accuracy_score(truth, predicted)),
                "test_samples": int(valid_test.sum()),
                "classification_report_text": classification_report(
                    truth,
                    predicted,
                    labels=list(range(2)),
                    target_names=encoder.classes_,
                    digits=4,
                    zero_division=0,
                ),
            }
    return bundle


def _persistent_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    excluded = {"train_feature_df", "train_baseline_df"}
    return {key: value for key, value in bundle.items() if key not in excluded}


def fit_offline_model(
    train_df: pd.DataFrame,
    task_config: Dict[str, Any],
    validation_df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fit the binary classifier; validation is intentionally not used."""
    bundle = train_group_zscore_resp_linear_model(train_df, task_config)
    metadata = {
        "validation_ignored": validation_df is not None,
        "train_summary": bundle.get("train_summary", {}),
    }
    return _persistent_bundle(bundle), metadata


def predict_feature_vector(
    model_bundle: Dict[str, Any], feature_vector: Iterable[float]
) -> Dict[str, Any]:
    vector = extract_model_feature_vector(model_bundle, feature_vector).reshape(1, -1)
    predicted_index = int(model_bundle["svm_model"].predict(vector)[0])
    label = str(
        model_bundle["label_encoder"].inverse_transform([predicted_index])[0]
    )
    allowed = set((model_bundle.get("task_config") or {}).get("class_labels", []))
    if allowed and label not in allowed:
        raise ValueError("模型输出了当前任务未配置的标签: {}".format(label))
    return {
        "predicted_index": predicted_index,
        "predicted_label": label,
        "probabilities": {},
    }


def predict_dataframe(
    model_bundle: Dict[str, Any], dataset_df: pd.DataFrame
) -> pd.DataFrame:
    config = model_bundle.get("task_config") or {
        "class_labels": model_bundle.get("class_names", []),
        "label_merge_rule": None,
    }
    normalized, _ = apply_task_labels(dataset_df, config)
    feature_df, _ = build_training_feature_frame(normalized)
    rows = []
    for index, row in feature_df.iterrows():
        resp = pd.to_numeric(row[RESP_COLUMNS], errors="coerce").values.astype(float)
        if not np.isfinite(resp).all():
            rows.append(
                {
                    "source_index": index,
                    "true_label": _clean_label(row.get("Label")),
                    "predicted_label": "",
                    "inference_error": "resp16存在缺失值",
                }
            )
            continue
        result = predict_feature_vector(model_bundle, build_feature_vector(resp))
        result["source_index"] = index
        result["true_label"] = _clean_label(row.get("Label"))
        rows.append(result)
    return pd.DataFrame(rows).set_index("source_index", drop=False)
