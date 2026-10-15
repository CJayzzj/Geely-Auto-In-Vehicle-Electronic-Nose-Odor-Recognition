# -*- coding: utf-8 -*-
"""Metrics for a symmetric two-class closed-set task."""

from typing import Any, Dict

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from task_config import validate_task_config


def evaluate_predictions(
    dataset_df: pd.DataFrame,
    predictions: pd.DataFrame,
    task_config: Dict[str, Any],
    label_column: str = "Label",
) -> Dict[str, Any]:
    config = validate_task_config(task_config)
    if label_column not in dataset_df.columns:
        raise ValueError("测试数据缺少标签列: {}".format(label_column))
    labels = config["class_labels"]
    merge_rule = config.get("label_merge_rule") or {}
    truth = dataset_df.loc[predictions.index, label_column].astype(str).str.strip()
    truth = truth.map(lambda value: merge_rule.get(value, value))
    predicted = predictions["predicted_label"].astype(str).str.strip()
    valid = truth.isin(labels) & predicted.isin(labels)
    truth, predicted = truth.loc[valid], predicted.loc[valid]
    matrix = confusion_matrix(truth, predicted, labels=labels)
    return {
        "accuracy": float(accuracy_score(truth, predicted)) if len(truth) else 0.0,
        "labels": labels,
        "confusion_matrix": matrix,
        "evaluated_samples": int(len(truth)),
        "classification_report_text": classification_report(
            truth,
            predicted,
            labels=labels,
            digits=4,
            zero_division=0,
        ),
    }
