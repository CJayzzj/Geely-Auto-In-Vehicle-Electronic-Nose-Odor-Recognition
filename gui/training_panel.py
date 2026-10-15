# -*- coding: utf-8 -*-
"""Training controls composed into ModelCenterDialog.

The visual layout follows V4.6. The worker now receives the immutable startup
configuration so labels are merged once and the shared binary classifier is used.
"""

from runtime_support import *
from gui.panel_base import OwnerPanel
from gui.training_thread import TrainingThread


class TrainingPanel(OwnerPanel):
    def _build_training_group(self, parent_layout):
        group_box = QtWidgets.QGroupBox("训练区", self)
        group_box.setStyleSheet("font: 11pt \"楷体\";")
        layout = QtWidgets.QVBoxLayout(group_box)

        train_file_layout = QtWidgets.QHBoxLayout()
        self.train_path_input = QtWidgets.QLineEdit(self)
        self.train_path_input.setPlaceholderText("训练集文件路径")
        self.train_browse_btn = QtWidgets.QPushButton("训练集")
        train_file_layout.addWidget(self.train_path_input, 1)
        train_file_layout.addWidget(self.train_browse_btn)
        layout.addLayout(train_file_layout)

        validation_file_layout = QtWidgets.QHBoxLayout()
        self.validation_path_input = QtWidgets.QLineEdit(self)
        self.validation_path_input.setPlaceholderText("验证集文件路径（兼容保留，闭集二分类不使用）")
        self.validation_browse_btn = QtWidgets.QPushButton("验证集")
        validation_file_layout.addWidget(self.validation_path_input, 1)
        validation_file_layout.addWidget(self.validation_browse_btn)
        layout.addLayout(validation_file_layout)

        model_file_layout = QtWidgets.QHBoxLayout()
        self.model_output_input = QtWidgets.QLineEdit(self)
        self.model_output_input.setPlaceholderText("模型输出路径（joblib）")
        self.model_output_browse_btn = QtWidgets.QPushButton("模型保存")
        model_file_layout.addWidget(self.model_output_input, 1)
        model_file_layout.addWidget(self.model_output_browse_btn)
        layout.addLayout(model_file_layout)

        # info_layout = QtWidgets.QHBoxLayout()
        # self.algorithm_label = QtWidgets.QLabel("算法: SVM")
        # self.feature_mode_label = QtWidgets.QLabel("特征: resp + pattern")
        # self.baseline_mode_label = QtWidgets.QLabel("baseline: sigmoid(|resp|)")
        # info_layout.addWidget(self.algorithm_label)
        # info_layout.addWidget(self.feature_mode_label)
        # info_layout.addWidget(self.baseline_mode_label)
        # info_layout.addStretch()
        # layout.addLayout(info_layout)

        self.start_training_btn = QtWidgets.QPushButton("开始训练")
        layout.addWidget(self.start_training_btn)

        self.training_result_text = QtWidgets.QTextEdit(self)
        self.training_result_text.setReadOnly(True)
        self.training_result_text.setPlaceholderText("训练日志和评估结果会显示在这里。")
        layout.addWidget(self.training_result_text, 1)

        parent_layout.addWidget(group_box, 1)

        self.train_browse_btn.clicked.connect(lambda: self.browse_training_file("train"))
        self.validation_browse_btn.clicked.connect(lambda: self.browse_training_file("validation"))
        self.model_output_browse_btn.clicked.connect(self.browse_model_output_file)
        self.start_training_btn.clicked.connect(self.start_training)

    def browse_training_file(self, field_name):
        selected_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择训练/验证集文件",
            self._device_datasets_dir(),
            "Dataset Files (*.csv *.xlsx *.xls)",
        )
        if not selected_path:
            return
        if field_name == "train":
            self.train_path_input.setText(selected_path)
        elif field_name == "validation":
            self.validation_path_input.setText(selected_path)

    def browse_model_output_file(self):
        train_path = self.train_path_input.text().strip()
        default_path = self.build_default_model_output_path(train_path if train_path else None)
        selected_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "选择模型保存路径",
            default_path,
            "Joblib Model (*.joblib)",
        )
        if selected_path:
            self.model_output_input.setText(selected_path)

    def start_training(self):
        train_path = self.train_path_input.text().strip()
        if not train_path:
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择训练集文件。")
            return
        validation_path = ""
        if hasattr(self, "validation_path_input"):
            validation_path = self.validation_path_input.text().strip()

        model_path = self.model_output_input.text().strip()
        if not model_path:
            model_path = self.build_default_model_output_path(train_path)
            self.model_output_input.setText(model_path)

        self.training_result_text.setPlainText("训练中，请稍候...")
        self.start_training_btn.setEnabled(False)
        self.training_thread = TrainingThread(
            train_path,
            None,
            model_path,
            validation_path=validation_path or None,
            task_config=self.task_config,
        )
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.training_thread.training_failed.connect(self.on_training_failed)
        self.training_thread.finished.connect(self.on_training_thread_finished)
        self.training_thread.start()

    def _labels_from_current_dataset(self):
        """当前窗口已加载数据集中的非空类别名（去重、排序）。"""
        if self.dataset_df is None or self.dataset_df.empty:
            return []
        if "Label" not in self.dataset_df.columns:
            return []
        merge_rule = self.task_config.get("label_merge_rule") or {}
        class_set = set(self.task_config["class_labels"])
        labels = {
            merge_rule.get(str(label).strip(), str(label).strip())
            for label in self.dataset_df["Label"].tolist()
        }
        return sorted(labels & class_set)

    def build_default_model_output_path(self, train_path=None):
        """默认文件名：时间戳 + 当前数据集类别名组合；当前集无类别时再从训练集文件推断。"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        labels = self._labels_from_current_dataset()
        if not labels and train_path:
            try:
                train_df, _ = load_dataset(train_path)
                merge_rule = self.task_config.get("label_merge_rule") or {}
                class_set = set(self.task_config["class_labels"])
                mapped = {
                    merge_rule.get(str(label).strip(), str(label).strip())
                    for label in train_df["Label"].tolist()
                }
                labels = sorted(mapped & class_set)
            except Exception:
                labels = []

        if labels:
            label_text = "_".join(labels)
        else:
            label_text = "unknown"

        safe_label_text = "".join(
            ch if ch.isalnum() or ch in ("_", "-", ".") else "_"
            for ch in label_text
        )[:80]
        filename = f"{timestamp}_{safe_label_text}.joblib"
        return os.path.join(self._device_models_dir(), filename)

    def on_training_finished(self, result):
        self.training_result_text.setPlainText(result["summary_text"])
        try:
            self.main_window.load_model_bundle_from_path(result["model_path"])
            self.sync_model_state()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "提示", f"训练完成，但自动加载模型失败：{exc}")

    def on_training_failed(self, message):
        self.training_result_text.setPlainText(f"训练失败:\n{message}")
        QtWidgets.QMessageBox.critical(self, "错误", f"训练失败：{message}")

    def on_training_thread_finished(self):
        self.start_training_btn.setEnabled(True)
        self.training_thread = None
