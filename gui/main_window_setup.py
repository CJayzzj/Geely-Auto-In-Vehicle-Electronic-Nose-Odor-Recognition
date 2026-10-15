# -*- coding: utf-8 -*-
"""Functional MainWindow mixin mechanically moved from V4.6.

Keeping these methods in focused modules controls file size while preserving the
original widget layout and signal behavior.
"""

from runtime_support import *
from gui.model_center_dialog import ModelCenterDialog
from gui.plot_widgets import _apply_standard_plot_axes


class MainWindowSetupMixin:
    def _remove_legacy_main_label_controls(self):
        """Remove the old main-window label field; raw saving is owned by Data Collection."""
        for widget_name in ("label_name", "label_input"):
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            try:
                self.horizontalLayout_4.removeWidget(widget)
            except Exception:
                pass
            widget.hide()

    def reload_th_comp_coeffs_from_disk(self):
        """按当前设备号目录加载 th_compensation/<设备>/comp_params.xlsx；缺失或损坏则用全零系数。"""
        path = os.path.join(
            TH_COMP_DIR, self.get_device_segment(), TH_COMP_COEFF_FILENAME
        )
        if os.path.isfile(path):
            try:
                coeffs, _ = load_th_comp_coeffs_xlsx(path)
                self.th_comp_coeffs = coeffs
            except Exception:
                self.th_comp_coeffs = default_th_comp_coeffs()
        else:
            self.th_comp_coeffs = default_th_comp_coeffs()
        if self.th_comp_dialog is not None:
            try:
                self.th_comp_dialog._sync_from_main_window()
                self.th_comp_dialog._refresh_plots()
            except Exception:
                pass

    def showEvent(self, event):
        super().showEvent(event)
        self._connect_screen_change_signal()
        if getattr(self, "_initial_window_geometry_done", False):
            return
        self._initial_window_geometry_done = True
        self._fit_initial_window_to_available_screen()

    def _connect_screen_change_signal(self):
        """连接当前原生窗口的跨屏信号；窗口句柄重建时会自动重新连接。"""
        handle = self.windowHandle()
        if handle is None or handle is self._screen_change_window_handle:
            return
        old_handle = self._screen_change_window_handle
        if old_handle is not None:
            try:
                old_handle.screenChanged.disconnect(self._on_screen_changed)
            except (TypeError, RuntimeError):
                pass
        handle.screenChanged.connect(self._on_screen_changed)
        self._screen_change_window_handle = handle

    def _on_screen_changed(self, screen):
        """跨屏后延迟刷新，避免在窗口仍处于拖动过程中反复改变尺寸。"""
        self._pending_target_screen = screen
        self._screen_change_timer.start()

    def _apply_pending_screen_change(self):
        screen = self._pending_target_screen
        self._pending_target_screen = None
        if screen is None:
            return
        self._fit_window_to_screen(screen, center_window=True)

    def _available_geometry_for_screen(self, screen=None):
        if screen is None:
            handle = self.windowHandle()
            if handle is not None:
                screen = handle.screen()
        if screen is not None:
            try:
                return screen.availableGeometry()
            except RuntimeError:
                pass
        return QtWidgets.QApplication.desktop().availableGeometry(self)

    @staticmethod
    def _window_size_for_available_geometry(available_geometry):
        """使用目标屏幕的完整可用区域；新版四列仪表盘不再套用旧 UI 宽高比。"""
        margin = 16
        max_w = max(320, available_geometry.width() - margin * 2)
        max_h = max(240, available_geometry.height() - margin * 2)
        return QtCore.QSize(max_w, max_h), max_w, max_h

    def _apply_screen_layout_profile(self, available_geometry):
        """矮屏幕降低固定最小高度，保证三排仪表盘能完整显示。"""
        compact = available_geometry.height() < 820
        main_plot_height = 72 if compact else MAIN_SIGNAL_PLOT_HEIGHT
        inference_height = 142 if compact else MAIN_INFERENCE_PANEL_HEIGHT
        button_height = 24 if compact else 28

        for plot in getattr(self, "voc_plot_widgets", {}).values():
            plot.setMinimumHeight(main_plot_height)
        for index in E_NOSE_VISIBLE_CHANNEL_INDICES:
            if index < len(getattr(self, "plot_widgets", [])):
                self.plot_widgets[index].setMinimumHeight(main_plot_height)

        temperature_plot = getattr(self, "temperature_humidity_plot", None)
        if temperature_plot is not None:
            temperature_plot.setMinimumHeight(main_plot_height)
            temperature_plot.setMaximumHeight(16777215)
        inference_group = getattr(self, "main_inference_result_group", None)
        if inference_group is not None:
            inference_group.setMinimumHeight(inference_height)
        timeline_plot = getattr(self, "voc_timeline_plot", None)
        if timeline_plot is not None:
            timeline_plot.setMinimumHeight(68 if compact else 105)
            timeline_plot.setMaximumHeight(100 if compact else 145)
        status_root_layout = getattr(self, "voc_status_root_layout", None)
        if status_root_layout is not None:
            status_root_layout.setSpacing(4 if compact else 6)
            status_root_layout.setContentsMargins(
                6 if compact else 7,
                12 if compact else 15,
                6 if compact else 7,
                5 if compact else 9,
            )
        status_grid = getattr(self, "voc_status_grid", None)
        if status_grid is not None:
            status_grid.setVerticalSpacing(3 if compact else 5)
        for label_map_name in (
            "voc_r_labels",
            "voc_dr_labels",
            "voc_event_labels",
        ):
            for label in getattr(self, label_map_name, {}).values():
                label.setMinimumHeight(24 if compact else 28)
                label.setMaximumHeight(26 if compact else 32)

        for button_name in (
            "comfirm_link",
            "reflesh",
            "clear",
            "baseline_params_confirm_btn",
            "data_collection_btn",
            "model_center_btn",
            "feature_btn",
            "th_comp_btn",
        ):
            button = getattr(self, button_name, None)
            if button is not None:
                button.setMinimumHeight(button_height)

        for index in E_NOSE_VISIBLE_CHANNEL_INDICES:
            if index < len(getattr(self, "value_labels", [])):
                value_label = self.value_labels[index]
                value_label.setMinimumHeight(28 if compact else 34)
                value_label.setMaximumHeight(38 if compact else 44)

        title = getattr(self, "Sys_name", None)
        if title is not None:
            title.setMaximumHeight(34 if compact else 16777215)

    def _fit_window_to_screen(self, screen=None, center_window=True):
        """依据目标屏幕重新设置窗口约束并强制布局重新计算。"""
        available_geometry = self._available_geometry_for_screen(screen)
        target_size, max_w, max_h = self._window_size_for_available_geometry(
            available_geometry
        )
        w = target_size.width()
        h = target_size.height()
        self._apply_screen_layout_profile(available_geometry)
        min_w = min(640, w)
        min_h = min(480, h)
        self.setMinimumSize(min_w, min_h)
        if not self.isMaximized() and not self.isFullScreen():
            self.resize(min(w, max_w), min(h, max_h))
            if center_window:
                frame = self.frameGeometry()
                frame.moveCenter(available_geometry.center())
                self.move(frame.topLeft())

        for widget in (self.centralwidget, self.view, self.vehicle_dashboard):
            if widget is None:
                continue
            widget.updateGeometry()
            widget.layout().invalidate()
            widget.layout().activate()
        self.updateGeometry()
        self.update()

    def _fit_initial_window_to_available_screen(self):
        """按首次所在屏幕调整并居中窗口。"""
        self._fit_window_to_screen(screen=None, center_window=True)

    def adc_to_resistance(self, adc_value):
        """按公式 R = 匹配电阻 * ADC / (4096 - ADC) 将 ADC 转为电阻值。"""
        try:
            adc_value = float(adc_value)
        except (TypeError, ValueError):
            return float("inf")
        if not np.isfinite(adc_value) or adc_value >= 4096:
            return float("inf")
        denominator = 4096.0 - adc_value
        if denominator <= 0.0:
            return float("inf")
        return self.get_matched_resistance() * adc_value / denominator

    @staticmethod
    def adc_to_feature_resistance(adc_value):
        """特征计算专用：将 ADC 真正换算为电阻值 R。"""
        if adc_value is None or adc_value == float("inf"):
            return float("inf")
        if adc_value >= 4096:
            return float("inf")
        return MATCHED_RESISTANCE * adc_value / (4096 - adc_value)

    @staticmethod
    def format_record_time(timestamp=None):
        """将时间戳格式化为带毫秒的年月日时分秒字符串。"""
        if timestamp is None:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    @staticmethod
    def format_elapsed_time(seconds):
        """将秒数格式化为 HH:MM:SS。"""
        total_seconds = max(0, int(seconds))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def show_error_message(self, title, message, throttle_ms=ERROR_DIALOG_THROTTLE_MS):
        """节流重复错误弹窗，避免长时间异常时界面卡顿。"""
        # 先在状态栏显示，再按节流策略决定是否弹窗。
        if hasattr(self, "statusbar"):
            self.statusbar.showMessage(message, 5000)

        now_ms = int(time.time() * 1000)
        if message == self._last_error_text and (now_ms - self._last_error_time_ms) < throttle_ms:
            return

        self._last_error_text = message
        self._last_error_time_ms = now_ms
        QtWidgets.QMessageBox.critical(self, title, message)

    def get_device_segment(self):
        """当前设备号对应的目录名；未填写时为 default。"""
        return sanitize_path_segment(self.get_device_id_text(), "default")

    def get_device_id_text(self):
        if not hasattr(self, "device_id_input"):
            return ""
        return self.device_id_input.text().strip()

    def get_raw_root_dir(self):
        """原始 CSV 根目录：data/raw/<设备号>/"""
        return os.path.join(RAW_DATA_DIR, self.get_device_segment())

    def _update_comfirm_link_enabled(self):
        """填写设备号后即可连接串口。"""
        if not hasattr(self, "comfirm_link") or not hasattr(self, "device_id_input"):
            return
        dev_ok = bool(self.get_device_id_text())
        self.comfirm_link.setEnabled(dev_ok)

    def get_matched_resistance(self):
        """返回当前主界面设置的匹配电阻；异常时回退默认值。"""
        try:
            if hasattr(self, "matched_resistance_input"):
                value = float(self.matched_resistance_input.value())
            else:
                value = float(getattr(self, "matched_resistance_value", MATCHED_RESISTANCE))
        except (TypeError, ValueError):
            value = float(MATCHED_RESISTANCE)
        if not np.isfinite(value) or value <= 0.0:
            value = float(MATCHED_RESISTANCE)
        self.matched_resistance_value = value
        return value

    def _on_matched_resistance_changed(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if np.isfinite(value) and value > 0.0:
            self.matched_resistance_value = value

    def _init_matched_resistance_field(self):
        """在主界面顶部加入匹配电阻设置，用于 ADC 到 R_raw 的换算。"""
        self.matched_resistance_label = QtWidgets.QLabel("匹配电阻", self)
        self.matched_resistance_label.setStyleSheet("font: 12pt \"楷体\";")
        self.matched_resistance_input = QtWidgets.QSpinBox(self)
        self.matched_resistance_input.setRange(1, 1000000)
        self.matched_resistance_input.setSingleStep(10)
        self.matched_resistance_input.setValue(int(round(float(MATCHED_RESISTANCE))))
        self.matched_resistance_input.setMaximumWidth(110)
        self.matched_resistance_input.setStyleSheet("font: 12pt \"楷体\";")
        self.matched_resistance_input.setToolTip(
            "ADC 转原始电阻公式：R_raw = 匹配电阻 * ADC / (4096 - ADC)。"
        )
        try:
            self.horizontalLayout_4.insertWidget(2, self.matched_resistance_label)
            self.horizontalLayout_4.insertWidget(3, self.matched_resistance_input)
        except Exception:
            self.matched_resistance_label.setParent(self.centralwidget)
            self.matched_resistance_input.setParent(self.centralwidget)
        self.matched_resistance_input.valueChanged.connect(
            self._on_matched_resistance_changed
        )
        self._on_matched_resistance_changed(self.matched_resistance_input.value())

    def _init_device_id_field(self):
        """在标签行最左侧插入设备号输入，用于数据与模型分目录。"""
        self.device_id_label = QtWidgets.QLabel("设备号", self)
        self.device_id_label.setStyleSheet("font: 12pt \"楷体\";")
        self.device_id_label.setObjectName("device_id_label")
        self.device_id_input = QtWidgets.QLineEdit(self)
        self.device_id_input.setPlaceholderText("必填，连接串口前填写")
        self.device_id_input.setToolTip(
            "连接串口前必须填写。原始数据、数据集、模型与补偿参数将保存到该设备号子文件夹下。"
        )
        self.device_id_input.setStyleSheet("font: 12pt \"楷体\";")
        self.device_id_input.setMinimumWidth(96)
        self.device_id_input.setMaximumWidth(200)
        self.device_id_input.setObjectName("device_id_input")
        self.device_id_input.setText("V2")
        try:
            self.horizontalLayout_4.insertWidget(0, self.device_id_input)
            self.horizontalLayout_4.insertWidget(0, self.device_id_label)
        except Exception:
            self.device_id_label.setParent(self.centralwidget)
            self.device_id_input.setParent(self.centralwidget)
        self.device_id_input.textChanged.connect(self._update_comfirm_link_enabled)

    def _init_model_center_button(self):
        """在顶部区域插入数据采集与模型训练/推理入口按钮。"""
        self.data_collection_btn = QtWidgets.QPushButton(self)
        self.data_collection_btn.setText("数据采集")
        self.data_collection_btn.setStyleSheet("font: 12pt \"楷体\";")
        self.data_collection_btn.clicked.connect(self.open_data_collection)

        self.model_center_btn = QtWidgets.QPushButton(self)
        self.model_center_btn.setText("模型训练/推理")
        self.model_center_btn.setStyleSheet("font: 12pt \"楷体\";")
        self.model_center_btn.clicked.connect(self.open_model_center)

        try:
            insert_pos = max(0, self.horizontalLayout_4.count() - 1)
            self.horizontalLayout_4.insertWidget(insert_pos, self.data_collection_btn)
            self.horizontalLayout_4.insertWidget(insert_pos + 1, self.model_center_btn)
        except Exception:
            self.data_collection_btn.setParent(self)
            self.data_collection_btn.move(10, 40)
            self.model_center_btn.setParent(self)
            self.model_center_btn.move(110, 40)

    def _init_feature_button(self):
        """在顶部区域插入“特征窗口”按钮。"""
        self.feature_btn = QtWidgets.QPushButton(self)
        self.feature_btn.setText("特征窗口")
        self.feature_btn.setStyleSheet("font: 12pt \"楷体\";")
        self.feature_btn.clicked.connect(self.open_feature_window)

        try:
            insert_pos = max(0, self.horizontalLayout_4.count() - 1)
            self.horizontalLayout_4.insertWidget(insert_pos, self.feature_btn)
        except Exception:
            self.feature_btn.setParent(self)
            self.feature_btn.move(10, 40)

    def _init_th_comp_button(self):
        """在顶部区域插入「温湿度补偿」按钮。"""
        self.th_comp_btn = QtWidgets.QPushButton(self)
        self.th_comp_btn.setText("温湿度补偿")
        self.th_comp_btn.setStyleSheet("font: 12pt \"楷体\";")
        self.th_comp_btn.clicked.connect(self.open_th_compensation)

        try:
            insert_pos = max(0, self.horizontalLayout_4.count() - 1)
            self.horizontalLayout_4.insertWidget(insert_pos, self.th_comp_btn)
        except Exception:
            self.th_comp_btn.setParent(self)
            self.th_comp_btn.move(10, 70)

    def open_th_compensation(self):
        if self.th_comp_dialog is None:
            self.th_comp_dialog = ThCompensationDialog(self)
        self.th_comp_dialog.show()
        self.th_comp_dialog.raise_()
        self.th_comp_dialog.activateWindow()

    def open_data_collection(self):
        """打开数据采集窗口；若已打开则同步状态并置顶。"""
        if self.data_collection_dialog is None:
            self.data_collection_dialog = ModelCenterDialog(self, mode="data")
        self.data_collection_dialog.sync_data_collection_state()
        self.data_collection_dialog.show()
        self.data_collection_dialog.raise_()
        self.data_collection_dialog.activateWindow()

    def open_model_center(self):
        """打开模型训练/推理窗口；若已打开则同步状态并置顶。"""
        if self.model_center_dialog is None:
            self.model_center_dialog = ModelCenterDialog(self, mode="all")
        self.model_center_dialog.sync_model_state()
        self.model_center_dialog.show()
        self.model_center_dialog.raise_()
        self.model_center_dialog.activateWindow()

    def _init_system_info_label(self):
        """在顶部区域显示系统运行时间。"""
        self.system_info_label = QtWidgets.QLabel(self)
        self.system_info_label.setText("运行时间: 00:00:00")
        self.system_info_label.setStyleSheet("font: 11pt \"楷体\"; color: rgb(120, 40, 40);")
        self.system_info_label.setMinimumWidth(320)

        try:
            insert_pos = max(0, self.horizontalLayout_4.count() - 1)
            self.horizontalLayout_4.insertWidget(insert_pos, self.system_info_label)
        except Exception:
            self.system_info_label.setParent(self)
            self.system_info_label.move(320, 10)

    def _populate_dashboard_control_panel(self):
        """把电子鼻控制项整理成多行紧凑布局，避免中央两列被横向撑宽。"""
        layout = getattr(self, "e_nose_control_layout", None)
        if layout is None:
            return

        for index in range(self.verticalLayout.count()):
            item = self.verticalLayout.itemAt(index)
            if item is not None and item.layout() is self.verticalLayout_2:
                self.verticalLayout.takeAt(index)
                break
        self.verticalLayout_2.setParent(None)
        self.verticalLayout.setStretch(0, 0)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 0)

        compact_widths = (
            (self.rate_input, 55, 85),
            (self.reflesh_comboBox, 90, 160),
            (self.device_id_input, 70, 120),
            (self.matched_resistance_input, 60, 90),
            (self.main_udp_endpoint_input, 130, 230),
            (self.main_udp_status_label, 75, 170),
            (self.system_info_label, 130, 16777215),
        )
        for widget, minimum, maximum in compact_widths:
            if widget is None:
                continue
            widget.setMinimumWidth(minimum)
            widget.setMaximumWidth(maximum)

        for button in (
            self.comfirm_link,
            self.reflesh,
            self.clear,
            self.baseline_params_confirm_btn,
            self.data_collection_btn,
            self.model_center_btn,
            self.feature_btn,
            self.th_comp_btn,
        ):
            button.setMinimumHeight(28)
        self.comfirm_link.setMinimumWidth(88)
        self.reflesh.setMinimumWidth(96)
        self.clear.setMinimumWidth(58)

        self.e_nose_port_label = QtWidgets.QLabel(
            "电子鼻串口", self.e_nose_control_group
        )
        self.e_nose_port_label.setStyleSheet('font: 11pt "KaiTi";')

        rows = (
            (
                self.rate_name,
                self.rate_input,
                self.e_nose_port_label,
                self.reflesh_comboBox,
                self.comfirm_link,
                self.reflesh,
                self.clear,
            ),
            (
                self.baseline_xc_label,
                self.baseline_xc_input,
                self.baseline_af_label,
                self.baseline_af_input,
                self.baseline_as_label,
                self.baseline_as_input,
                self.baseline_params_confirm_btn,
                self.system_info_label,
            ),
            (
                self.device_id_label,
                self.device_id_input,
                self.matched_resistance_label,
                self.matched_resistance_input,
                self.feature_btn,
                self.th_comp_btn,
            ),
            (
                self.data_collection_btn,
                self.model_center_btn,
                self.main_udp_endpoint_title,
                self.main_udp_endpoint_input,
                self.main_udp_status_label,
            ),
        )
        for row_index, widgets in enumerate(rows):
            row_layout = QtWidgets.QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            for widget in widgets:
                if widget is not None:
                    row_layout.addWidget(widget)
            layout.addLayout(row_layout, row_index, 0)

        layout.setColumnStretch(0, 1)

    def _init_main_udp_endpoint_display(self):
        """在主界面显示 UDP 服务的本机连接地址。"""
        udp_layout = QtWidgets.QHBoxLayout()
        udp_layout.setSpacing(6)
        udp_layout.setContentsMargins(0, 0, 0, 0)

        self.main_udp_endpoint_title = QtWidgets.QLabel("UDP本机", self.centralwidget)
        self.main_udp_endpoint_title.setStyleSheet("font: 12pt \"楷体\";")
        self.main_udp_endpoint_title.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )

        self.main_udp_endpoint_input = QtWidgets.QLineEdit(self.centralwidget)
        self.main_udp_endpoint_input.setReadOnly(True)
        self.main_udp_endpoint_input.setMinimumWidth(240)
        self.main_udp_endpoint_input.setMaximumWidth(420)
        self.main_udp_endpoint_input.setStyleSheet(
            "font: 11pt \"楷体\"; color: rgb(40, 70, 120);"
        )

        self.main_udp_status_label = QtWidgets.QLabel("UDP: 未开启", self.centralwidget)
        self.main_udp_status_label.setMinimumWidth(140)
        self.main_udp_status_label.setStyleSheet(
            "font: 11pt \"楷体\"; color: rgb(80, 80, 80);"
        )
        self.main_udp_status_label.setWordWrap(True)

        try:
            insert_at = min(2, self.horizontalLayout_2.count())
            self.horizontalLayout_2.insertWidget(insert_at, self.main_udp_endpoint_title)
            self.horizontalLayout_2.insertWidget(insert_at + 1, self.main_udp_endpoint_input)
            self.horizontalLayout_2.insertWidget(insert_at + 2, self.main_udp_status_label)
        except Exception:
            udp_layout.addWidget(self.main_udp_endpoint_title)
            udp_layout.addWidget(self.main_udp_endpoint_input, 2)
            udp_layout.addWidget(self.main_udp_status_label, 3)
            self.verticalLayout_2.insertLayout(2, udp_layout)
            self.main_udp_endpoint_title.setParent(self.centralwidget)
            self.main_udp_endpoint_input.setParent(self.centralwidget)
            self.main_udp_status_label.setParent(self.centralwidget)
        self.refresh_main_udp_endpoint_display()

    def refresh_main_udp_endpoint_display(self):
        """刷新主界面 UDP 地址和服务状态。"""
        if not hasattr(self, "main_udp_endpoint_input") or self.main_udp_endpoint_input is None:
            return
        endpoint_text = self.udp_local_endpoint_text(self.udp_result_port)
        self.main_udp_endpoint_input.setText(endpoint_text)
        self.main_udp_endpoint_input.setToolTip(endpoint_text)
        if self.main_udp_status_label is not None:
            status = str(getattr(self, "udp_result_status", "UDP: 未开启"))
            status = status.replace(f"，本机 {endpoint_text}", "")
            status = status.replace(f", 本机 {endpoint_text}", "")
            self.main_udp_status_label.setText(status)

    def _init_main_inference_result_panel(self):
        """在温湿度曲线右侧空白区显示实时推理结果。"""
        self.main_inference_result_group = QtWidgets.QGroupBox("实时推理结果", self.view)
        self.main_inference_result_group.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.main_inference_result_group.setMinimumHeight(MAIN_INFERENCE_PANEL_HEIGHT)
        self.main_inference_result_group.setStyleSheet(
            """
            QGroupBox {
                font: bold 14pt "楷体";
                color: rgb(45, 45, 45);
                border: 1px solid rgb(150, 150, 150);
                border-radius: 5px;
                margin-top: 12px;
                background-color: rgb(250, 250, 250);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
            }
            """
        )
        layout = QtWidgets.QVBoxLayout(self.main_inference_result_group)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(5)

        self.main_inference_prediction_label = QtWidgets.QLabel("当前识别：--")
        self.main_inference_prediction_label.setAlignment(QtCore.Qt.AlignCenter)
        self.main_inference_prediction_label.setStyleSheet(
            'font: bold 18pt "KaiTi"; color: rgb(170, 45, 45);'
        )
        layout.addWidget(self.main_inference_prediction_label)

        self.main_driver_drinking_label = QtWidgets.QLabel("饮酒位置判断：--")
        self.main_driver_drinking_label.setAlignment(QtCore.Qt.AlignCenter)
        self.main_driver_drinking_label.setStyleSheet(
            'font: bold 14pt "KaiTi"; color: rgb(130, 70, 30);'
        )
        layout.addWidget(self.main_driver_drinking_label)

        self.main_inference_odor_level_label = QtWidgets.QLabel("气味等级：--")
        self.main_inference_odor_level_label.setAlignment(QtCore.Qt.AlignCenter)
        self.main_inference_odor_level_label.setStyleSheet(
            'font: bold 12pt "KaiTi"; color: rgb(60, 80, 130);'
        )
        layout.addWidget(self.main_inference_odor_level_label)

        self.main_inference_score_label = QtWidgets.QLabel("score：--")
        self.main_inference_score_label.setAlignment(QtCore.Qt.AlignCenter)
        self.main_inference_score_label.setStyleSheet(
            'font: bold 12pt "KaiTi"; color: rgb(45, 110, 90);'
        )
        layout.addWidget(self.main_inference_score_label)
        layout.insertStretch(0, 1)
        layout.addStretch(1)

        self.gridLayout.addWidget(self.main_inference_result_group, 9, 2, 2, 2)
        self.update_main_inference_display()
