# -*- coding: utf-8 -*-
"""Shared closed-set binary-classification package."""

from .air_gate import (
    load_air_gate_calibration,
    resp_gate_score_from_resp_values,
    should_run_recognition_inference,
)
from .classifier import (
    apply_task_labels,
    fit_offline_model,
    predict_dataframe,
    predict_feature_vector,
    train_group_zscore_resp_linear_model,
)
from .evaluation import evaluate_predictions
from .features import *
from .model_io import load_model_bundle, save_model_bundle

__all__ = [
    "apply_task_labels",
    "evaluate_predictions",
    "fit_offline_model",
    "load_air_gate_calibration",
    "load_model_bundle",
    "predict_dataframe",
    "predict_feature_vector",
    "resp_gate_score_from_resp_values",
    "save_model_bundle",
    "should_run_recognition_inference",
    "train_group_zscore_resp_linear_model",
]
