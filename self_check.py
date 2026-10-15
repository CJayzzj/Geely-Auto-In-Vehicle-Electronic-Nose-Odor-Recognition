# -*- coding: utf-8 -*-
"""Fast source, algorithm, and offscreen-GUI regression check. Not real unit test,just a quick self-check"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets

from algorithm.classifier import (
    predict_dataframe,
    train_group_zscore_resp_linear_model,
)
from algorithm.features import RESP_COLUMNS
from gui.main_window import MainWindow
from gui.model_center_dialog import ModelCenterDialog
from task_config import get_task_config
from utils.dataset_io import (
    normalize_dataset_save_path,
    save_dataset_with_extra_columns,
)


ROOT = Path(__file__).resolve().parent


def synthetic_dataset():
    rows = []
    patterns = {
        "酒精": np.array([0.0, 1.0, 2.0, 3.0] * 4),
        "香烟": np.array([3.0, 2.0, 1.0, 0.0] * 4),
        "臭豆腐": np.array([2.0, 3.0, 0.0, 1.0] * 4),
        "空气": np.zeros(16),
    }
    rng = np.random.RandomState(46)
    for label, pattern in patterns.items():
        for _ in range(12):
            values = pattern + rng.normal(0.0, 0.03, size=16)
            row = {column: values[index] for index, column in enumerate(RESP_COLUMNS)}
            row["Label"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def check_source():
    algorithm_text = "\n".join(
        path.read_text(encoding="utf-8-sig")
        for path in (ROOT / "algorithm").glob("*.py")
    )
    for forbidden in ("KMeans", "Mahalanobis"):
        assert forbidden not in algorithm_text
    oversized = []
    for path in ROOT.rglob("*.py"):
        count = len(path.read_text(encoding="utf-8-sig").splitlines())
        if count > 2000:
            oversized.append((str(path.relative_to(ROOT)), count))
    assert not oversized, oversized


def check_algorithms():
    data = synthetic_dataset()
    for key in ("alcohol_vs_smoke", "alcohol_vs_smoke_odor"):
        config = get_task_config(key)
        bundle = train_group_zscore_resp_linear_model(data, config)
        predictions = predict_dataframe(bundle, data)
        outputs = set(predictions["predicted_label"].dropna().astype(str))
        assert outputs <= set(config["class_labels"])
        assert set(bundle["class_names"]) == set(config["class_labels"])
        assert not any(
            "distance" in column.lower() or "unknown" in column.lower()
            for column in predictions.columns
        )


def check_dataset_save_contract():
    with TemporaryDirectory() as temp_dir:
        target = normalize_dataset_save_path(
            str(Path(temp_dir) / "dataset_without_extension")
        )
        saved_path = save_dataset_with_extra_columns(
            synthetic_dataset().head(3),
            target,
        )
        assert saved_path == target
        assert Path(saved_path).exists()
        assert len(pd.read_excel(saved_path)) == 3


def check_gui():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    for key in ("alcohol_vs_smoke", "alcohol_vs_smoke_odor"):
        config = get_task_config(key)
        window = MainWindow(config)
        assert window.windowTitle() == config["app_title"]
        dialog = ModelCenterDialog(window, mode="all")
        assert hasattr(dialog, "raw_save_start_btn")
        assert hasattr(dialog, "dataset_table")
        assert hasattr(dialog, "start_training_btn")
        assert hasattr(dialog, "inference_toggle")
        assert " / ".join(config["class_labels"]) in dialog.inference_task_label.text()
        dialog.close()
        window.close()
    app.quit()


def main():
    check_source()
    check_algorithms()
    check_dataset_save_contract()
    check_gui()
    print("SELF-CHECK PASSED")


if __name__ == "__main__":
    main()
