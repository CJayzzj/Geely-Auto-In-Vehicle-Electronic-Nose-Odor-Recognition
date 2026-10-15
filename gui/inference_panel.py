# -*- coding: utf-8 -*-
"""Realtime inference controls for the closed-set binary model.

The former rejection-threshold controls and distance diagnostics are absent by
design. The panel shows only Stage0 gate/peak progress and one of the configured
binary task labels after a peak event is committed.
"""

import numpy as np

from runtime_support import *
from gui.panel_base import OwnerPanel


class InferencePanel(OwnerPanel):
    def _build_inference_group(self, parent_layout):
        group_box = QtWidgets.QGroupBox("实时推理区", self)
        group_box.setStyleSheet('font: 11pt "楷体";')
        layout = QtWidgets.QVBoxLayout(group_box)

        model_layout = QtWidgets.QHBoxLayout()
        self.loaded_model_path_input = QtWidgets.QLineEdit(self)
        self.loaded_model_path_input.setReadOnly(True)
        self.loaded_model_path_input.setPlaceholderText("当前未加载模型")
        self.load_model_btn = QtWidgets.QPushButton("加载模型")
        self.unload_model_btn = QtWidgets.QPushButton("卸载模型")
        model_layout.addWidget(self.loaded_model_path_input, 1)
        model_layout.addWidget(self.load_model_btn)
        model_layout.addWidget(self.unload_model_btn)
        layout.addLayout(model_layout)

        class_text = " / ".join(self.task_config["class_labels"])
        self.inference_task_label = QtWidgets.QLabel("当前任务: " + class_text)
        self.inference_task_label.setStyleSheet(
            'font: 11pt "楷体"; color: rgb(60, 90, 140);'
        )
        layout.addWidget(self.inference_task_label)

        self.inference_toggle = QtWidgets.QCheckBox("开启实时推理")
        layout.addWidget(self.inference_toggle)

        udp_layout = QtWidgets.QHBoxLayout()
        udp_layout.setSpacing(4)
        self.udp_result_toggle = QtWidgets.QCheckBox("UDP服务")
        self.udp_result_host_input = QtWidgets.QLineEdit(self)
        self.udp_result_host_input.setReadOnly(True)
        self.udp_result_host_input.setPlaceholderText("本机局域网IP")
        self.udp_result_host_input.setMinimumWidth(220)
        self.udp_result_port_spin = QtWidgets.QSpinBox(self)
        self.udp_result_port_spin.setRange(1, 65535)
        self.udp_result_port_spin.setValue(int(UDP_RESULT_DEFAULT_PORT))
        self.udp_result_apply_btn = QtWidgets.QPushButton("确认", self)
        udp_layout.addWidget(self.udp_result_toggle)
        udp_layout.addWidget(QtWidgets.QLabel("本机:"))
        udp_layout.addWidget(self.udp_result_host_input)
        udp_layout.addWidget(QtWidgets.QLabel("端口:"))
        udp_layout.addWidget(self.udp_result_port_spin)
        udp_layout.addWidget(self.udp_result_apply_btn)
        udp_layout.addStretch(1)
        layout.addLayout(udp_layout)
        self.udp_result_status_label = QtWidgets.QLabel("UDP: 未开启")
        layout.addWidget(self.udp_result_status_label)

        self.inference_status_label = QtWidgets.QLabel("状态: 未推理")
        self.inference_predict_label = QtWidgets.QLabel("当前识别: --")
        self.inference_odor_level_label = QtWidgets.QLabel("气味等级: --")
        self.inference_predict_label.setStyleSheet(
            'font: 14pt "楷体"; color: rgb(150, 40, 40);'
        )
        layout.addWidget(self.inference_status_label)
        layout.addWidget(self.inference_predict_label)
        layout.addWidget(self.inference_odor_level_label)

        self.inference_probability_text = QtWidgets.QTextEdit(self)
        self.inference_probability_text.setReadOnly(True)
        self.inference_probability_text.setPlaceholderText("峰值与门控详情显示在这里。")
        layout.addWidget(self.inference_probability_text)
        self.inference_history_list = QtWidgets.QListWidget(self)
        layout.addWidget(self.inference_history_list, 1)
        parent_layout.addWidget(group_box, 1)

        self.load_model_btn.clicked.connect(self.load_model_from_dialog)
        self.unload_model_btn.clicked.connect(self.unload_model)
        self.inference_toggle.toggled.connect(self.on_inference_toggled)
        self.udp_result_toggle.toggled.connect(self.mark_udp_result_config_dirty)
        self.udp_result_port_spin.valueChanged.connect(
            self.mark_udp_result_config_dirty
        )
        self.udp_result_apply_btn.clicked.connect(self.apply_udp_result_config)

    def load_model_from_dialog(self):
        selected_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "加载模型文件",
            self._device_models_dir(),
            "Joblib Model (*.joblib)",
        )
        if not selected_path:
            return
        try:
            self.main_window.load_model_bundle_from_path(selected_path)
            self.sync_model_state()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "错误", "加载模型失败：{}".format(exc))

    def unload_model(self):
        self.main_window.clear_loaded_model_bundle()
        self.sync_model_state()

    def _sync_udp_result_controls(self):
        if not hasattr(self, "udp_result_toggle"):
            return
        mw = self.main_window
        self.udp_result_config_dirty = False
        self.udp_result_toggle.blockSignals(True)
        self.udp_result_port_spin.blockSignals(True)
        self.udp_result_toggle.setChecked(bool(getattr(mw, "udp_result_enabled", False)))
        port = int(
            getattr(mw, "udp_result_port", UDP_RESULT_DEFAULT_PORT)
            or UDP_RESULT_DEFAULT_PORT
        )
        self.udp_result_port_spin.setValue(max(1, min(65535, port)))
        self.udp_result_host_input.setText(mw.udp_local_endpoint_text(port))
        self.udp_result_toggle.blockSignals(False)
        self.udp_result_port_spin.blockSignals(False)
        self.sync_udp_result_status()

    def sync_udp_result_status(self):
        if not hasattr(self, "udp_result_status_label"):
            return
        port = int(self.udp_result_port_spin.value())
        self.udp_result_host_input.setText(
            self.main_window.udp_local_endpoint_text(port)
        )
        if getattr(self, "udp_result_config_dirty", False):
            self.udp_result_status_label.setText("UDP: 参数待确认")
        else:
            self.udp_result_status_label.setText(
                str(getattr(self.main_window, "udp_result_status", "UDP: 未开启"))
            )

    def mark_udp_result_config_dirty(self, *args):
        self.udp_result_config_dirty = True
        self.sync_udp_result_status()

    def apply_udp_result_config(self):
        self.main_window.set_udp_result_config(
            enabled=self.udp_result_toggle.isChecked(),
            port=int(self.udp_result_port_spin.value()),
        )
        self.udp_result_config_dirty = False
        self.sync_udp_result_status()

    def on_inference_toggled(self, checked):
        self.main_window.set_inference_enabled(checked)

    def sync_model_state(self):
        if hasattr(self, "loaded_model_path_input"):
            self.loaded_model_path_input.setText(
                self.main_window.loaded_model_path or ""
            )
            self.inference_toggle.setEnabled(
                self.main_window.loaded_model_bundle is not None
            )
            self.inference_toggle.blockSignals(True)
            self.inference_toggle.setChecked(self.main_window.inference_enabled)
            self.inference_toggle.blockSignals(False)
            self._sync_udp_result_controls()
            self.update_inference_display(self.main_window.last_inference_result)
        self.sync_peak_record_state()
        self.refresh_raw_save_status()

    @staticmethod
    def _format_float(value, digits=4, invalid="--"):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return invalid
        return f"{number:.{digits}f}" if np.isfinite(number) else invalid

    def _peak_detail(self, result):
        if not result.get("peak_detector_enabled"):
            return ""
        status = str(result.get("peak_status", "--"))
        sensor = str(result.get("peak_track_sensor", "--"))
        current = self._format_float(result.get("peak_current_signal"))
        return "寻峰: {} | {} | 当前={}".format(status, sensor, current)

    def _refresh_inference_history(self):
        self.inference_history_list.clear()
        for item in reversed(self.main_window.inference_history[-20:]):
            self.inference_history_list.addItem(item)

    def update_inference_display(self, result):
        if not result:
            self.inference_status_label.setText("状态: 未推理")
            self.inference_predict_label.setText("当前识别: --")
            self.inference_odor_level_label.setText("气味等级: --")
            self.inference_probability_text.clear()
            self.inference_history_list.clear()
            return
        self.main_window.maybe_commit_recognition_snapshot(result)
        display = self.main_window.recognition_result_for_display(result)
        label = str(display.get("predicted_label", "--"))
        allowed = set(self.task_config["class_labels"]) | {"空气", "识别中"}
        if label not in allowed:
            label = "--"
        self.inference_status_label.setText(
            "状态: {} | 模型: {}".format(
                "开启" if self.main_window.inference_enabled else "关闭",
                "已加载" if self.main_window.loaded_model_bundle else "未加载",
            )
        )
        self.inference_predict_label.setText("当前识别: " + label)
        self.inference_odor_level_label.setText(
            "气味等级: "
            + odor_level_for_display(display.get("odor_level", "--"))
        )
        details = []
        gate_score = display.get("resp_gate_score")
        if gate_score is not None:
            details.append(
                "空气门控: {}/{}".format(
                    self._format_float(gate_score, 2),
                    self._format_float(
                        display.get(
                            "resp_gate_score_threshold",
                            INFERENCE_RESP_GATE_SCORE_THRESHOLD,
                        ),
                        2,
                    ),
                )
            )
        peak_text = self._peak_detail(result)
        if peak_text:
            details.append(peak_text)
        self.inference_probability_text.setPlainText("\n".join(details))
        self._refresh_inference_history()
