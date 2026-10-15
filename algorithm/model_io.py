# -*- coding: utf-8 -*-
"""Persistence for the closed-set binary model bundle."""

from pathlib import Path
from typing import Any, Dict

import joblib


REQUIRED_MODEL_FIELDS = {
    "svm_model",
    "label_encoder",
    "feature_spec",
    "class_names",
    "train_summary",
}


def _persistent_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    excluded = {"train_feature_df", "train_baseline_df"}
    return {key: value for key, value in bundle.items() if key not in excluded}


def save_model_bundle(path: str, bundle: Dict[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_persistent_bundle(dict(bundle)), target)
    return str(target)


def _ensure_joblib_numpy_alignment_compat() -> None:
    numpy_pickle = getattr(joblib, "numpy_pickle", None)
    wrapper = getattr(numpy_pickle, "NumpyArrayWrapper", None)
    if (
        wrapper is None
        or hasattr(wrapper, "safe_get_numpy_array_alignment_bytes")
        or getattr(wrapper, "_jl_alignment_compat_enabled", False)
    ):
        return
    original = wrapper.read_array

    def read_array_with_alignment(self, unpickler):
        alignment = getattr(self, "numpy_array_alignment_bytes", None)
        if not self.dtype.hasobject and alignment is not None:
            padding_byte = unpickler.file_handle.read(1)
            if not padding_byte:
                raise EOFError("模型文件缺少numpy数组对齐信息")
            padding_length = int.from_bytes(padding_byte, byteorder="little")
            if padding_length:
                padding = unpickler.file_handle.read(padding_length)
                if len(padding) != padding_length:
                    raise EOFError("模型文件的numpy数组对齐信息不完整")
        return original(self, unpickler)

    wrapper.read_array = read_array_with_alignment
    wrapper._jl_alignment_compat_enabled = True


def load_model_bundle(path: str) -> Dict[str, Any]:
    _ensure_joblib_numpy_alignment_compat()
    bundle = joblib.load(path)
    missing = REQUIRED_MODEL_FIELDS - set(bundle)
    if missing:
        raise ValueError("模型文件缺少必要字段: {}".format(sorted(missing)))
    return bundle
