# -*- coding: utf-8 -*-
"""Startup-only configuration for the two closed-set binary tasks."""

from copy import deepcopy
from typing import Any, Dict


TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "alcohol_vs_smoke": {
        "task_key": "alcohol_vs_smoke",
        "app_title": "酒精/香烟二分类上位机",
        "class_labels": ["酒精", "香烟"],
        "label_merge_rule": None,
    },
    "alcohol_vs_smoke_odor": {
        "task_key": "alcohol_vs_smoke_odor",
        "app_title": "酒精/香烟-异味二分类上位机",
        "class_labels": ["酒精", "香烟/异味"],
        "label_merge_rule": {"香烟": "香烟/异味", "异味": "香烟/异味"},
    },
}


def get_task_config(task_key: str) -> Dict[str, Any]:
    """Return an isolated config so widgets cannot mutate the global template."""
    if task_key not in TASK_CONFIGS:
        raise KeyError("未知二分类任务: {}".format(task_key))
    return deepcopy(TASK_CONFIGS[task_key])


def validate_task_config(task_config: Dict[str, Any]) -> Dict[str, Any]:
    config = deepcopy(task_config or {})
    labels = [str(value).strip() for value in config.get("class_labels", [])]
    if len(labels) != 2 or len(set(labels)) != 2 or not all(labels):
        raise ValueError("二分类任务必须配置两个不同的非空类别标签。")
    config["class_labels"] = labels
    config["label_merge_rule"] = {
        str(source).strip(): str(target).strip()
        for source, target in (config.get("label_merge_rule") or {}).items()
    } or None
    return config
