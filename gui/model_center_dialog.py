# -*- coding: utf-8 -*-
"""Lightweight container for acquisition, training, and inference panels.

The public constructor and mode values match V4.6. `data`, `model`, and `all`
select which composed panels are visible; both delivered products open `all`.
"""

from runtime_support import *
from gui.acquisition_panel import DataAcquisitionPanel
from gui.inference_panel import InferencePanel
from gui.training_panel import TrainingPanel


class ModelCenterDialog(QtWidgets.QDialog):
    DATASET_FIXED_FIELDS = ["序号", "Timestamp", "Label", SAMPLE_INDEX_COLUMN]
    DATASET_FIXED_HEADERS = ["序号", "时间戳", "Label", "sample_index"]

    def __init__(self, parent=None, mode="model"):
        super().__init__(parent)
        self.main_window = parent
        self.task_config = validate_task_config(parent.task_config)
        self.dialog_mode = str(mode or "model")
        self.dataset_df = dataframe_from_records_with_extra([])
        self.dataset_path = ""
        self.dataset_format_info = DatasetFormatInfo()
        self.current_view_indices = []
        self.training_thread = None
        self.dataset_table_dirty = False

        self.acquisition_panel = DataAcquisitionPanel(self)
        self.training_panel = TrainingPanel(self)
        self.inference_panel = InferencePanel(self)

        if self.dialog_mode == "data":
            self.setWindowTitle("数据采集")
            self.resize(1280, 820)
        elif self.dialog_mode == "all":
            self.setWindowTitle("模型中心")
            self.resize(1280, 820)
        else:
            self.setWindowTitle("模型训练与推理")
            self.resize(900, 820)

        main_layout = QtWidgets.QVBoxLayout(self)
        tips = {
            "data": "在此窗口中完成原始数据保存、数据集管理和峰值样本记录。",
            "all": "在此窗口中完成数据集管理、训练、模型加载与实时推理监控。",
            "model": "在此窗口中完成模型训练、模型加载与实时推理监控。",
        }
        top_tip = QtWidgets.QLabel(tips.get(self.dialog_mode, tips["model"]))
        top_tip.setStyleSheet('font: 11pt "楷体"; color: rgb(80, 80, 80);')
        main_layout.addWidget(top_tip)

        if self.dialog_mode == "all":
            content_layout = QtWidgets.QHBoxLayout()
            main_layout.addLayout(content_layout)
            left_layout = QtWidgets.QVBoxLayout()
            right_layout = QtWidgets.QVBoxLayout()
            content_layout.addLayout(left_layout, 3)
            content_layout.addLayout(right_layout, 2)
            self.acquisition_panel._build_raw_save_group(left_layout)
            self.acquisition_panel._build_dataset_group(left_layout)
            self.training_panel._build_training_group(right_layout)
            self.inference_panel._build_inference_group(right_layout)
        elif self.dialog_mode == "data":
            self.acquisition_panel._build_raw_save_group(main_layout)
            self.acquisition_panel._build_dataset_group(main_layout)
        else:
            self.training_panel._build_training_group(main_layout)
            self.inference_panel._build_inference_group(main_layout)

        self.inference_panel.sync_model_state()

    def __getattr__(self, name):
        for panel_name in (
            "acquisition_panel",
            "training_panel",
            "inference_panel",
        ):
            panel = self.__dict__.get(panel_name)
            if panel is None:
                continue
            try:
                return object.__getattribute__(panel, name)
            except AttributeError:
                continue
        raise AttributeError(name)

    def closeEvent(self, event):
        if self.training_thread is not None and self.training_thread.isRunning():
            event.ignore()
            self.hide()
            return
        if self.main_window is not None:
            if self.dialog_mode == "data":
                self.main_window.data_collection_dialog = None
            else:
                self.main_window.model_center_dialog = None
        event.accept()
        self.deleteLater()
