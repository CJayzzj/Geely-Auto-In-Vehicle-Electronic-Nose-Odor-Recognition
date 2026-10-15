# -*- coding: utf-8 -*-
"""Functional MainWindow mixin mechanically moved from V4.6.

Keeping these methods in focused modules controls file size while preserving the
original widget layout and signal behavior.
"""

from runtime_support import *
from acquisition.peak_detector import VocEventDetector
from acquisition.serial_threads import SingleVocSerialThread
from gui.plot_widgets import _apply_standard_plot_axes, sensor_raw_curve_rgb


class MainWindowVocMixin:
    def _init_voc_event_panel(self):
        """按三列仪表盘布局组织电子鼻与三路单通道 VOC。"""
        panel_style = """
            QGroupBox {
                font: bold 11pt "KaiTi";
                color: rgb(45, 45, 45);
                border: 1px solid rgb(150, 150, 150);
                border-radius: 4px;
                margin-top: 10px;
                background-color: rgb(250, 250, 250);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """

        self.vehicle_dashboard = QtWidgets.QWidget(self.view)
        self.voc_event_group = self.vehicle_dashboard
        dashboard_layout = QtWidgets.QGridLayout(self.vehicle_dashboard)
        dashboard_layout.setContentsMargins(4, 4, 4, 4)
        dashboard_layout.setHorizontalSpacing(8)
        dashboard_layout.setVerticalSpacing(7)

        # 将原电子鼻控制区从主窗口顶部移到仪表盘中上方，保留原控件与连接。
        self.e_nose_control_group = QtWidgets.QGroupBox(
            "电子鼻设备控制", self.vehicle_dashboard
        )
        self.e_nose_control_group.setStyleSheet(panel_style)
        self.e_nose_control_group.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.e_nose_control_layout = QtWidgets.QGridLayout(
            self.e_nose_control_group
        )
        self.e_nose_control_layout.setContentsMargins(6, 12, 6, 6)
        self.e_nose_control_layout.setHorizontalSpacing(4)
        self.e_nose_control_layout.setVerticalSpacing(3)

        self.voc_mapping_group = QtWidgets.QGroupBox(
            "VOC串口映射", self.vehicle_dashboard
        )
        self.voc_mapping_group.setStyleSheet(panel_style)
        self.voc_mapping_group.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        mapping_layout = QtWidgets.QVBoxLayout(self.voc_mapping_group)
        mapping_layout.setContentsMargins(8, 15, 8, 10)
        mapping_layout.setSpacing(10)

        voc_params_layout = QtWidgets.QGridLayout()
        voc_params_layout.setHorizontalSpacing(5)
        voc_params_layout.setVerticalSpacing(8)
        mapping_label_style = 'font: 10pt "KaiTi";'
        baud_label = QtWidgets.QLabel("波特率", self.voc_mapping_group)
        baud_label.setStyleSheet(mapping_label_style)
        voc_params_layout.addWidget(baud_label, 0, 0)
        self.voc_baud_combo = QtWidgets.QComboBox(self.voc_mapping_group)
        self.voc_baud_combo.setMinimumHeight(28)
        self.voc_baud_combo.setStyleSheet(mapping_label_style)
        self.voc_baud_combo.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.voc_baud_combo.setCurrentText(str(VOC_DEFAULT_BAUDRATE))
        voc_params_layout.addWidget(self.voc_baud_combo, 0, 1)

        interval_label = QtWidgets.QLabel("采样间隔(s)", self.voc_mapping_group)
        interval_label.setStyleSheet(mapping_label_style)
        voc_params_layout.addWidget(interval_label, 1, 0)
        self.voc_interval_spin = QtWidgets.QDoubleSpinBox(self.voc_mapping_group)
        self.voc_interval_spin.setRange(0.2, 10.0)
        self.voc_interval_spin.setSingleStep(0.1)
        self.voc_interval_spin.setDecimals(1)
        self.voc_interval_spin.setValue(VOC_DEFAULT_INTERVAL_SEC)
        self.voc_interval_spin.setMinimumHeight(28)
        self.voc_interval_spin.setStyleSheet(mapping_label_style)
        voc_params_layout.addWidget(self.voc_interval_spin, 1, 1)
        mapping_layout.addLayout(voc_params_layout)

        button_layout = QtWidgets.QGridLayout()
        button_layout.setHorizontalSpacing(4)
        button_layout.setVerticalSpacing(8)
        self.voc_refresh_btn = QtWidgets.QPushButton("刷新串口", self.voc_mapping_group)
        self.voc_refresh_btn.setMinimumHeight(30)
        self.voc_refresh_btn.clicked.connect(self.on_refresh_clicked)
        button_layout.addWidget(self.voc_refresh_btn, 0, 0)
        self.voc_connect_btn = QtWidgets.QPushButton("连接3路VOC", self.voc_mapping_group)
        self.voc_connect_btn.setMinimumHeight(30)
        self.voc_connect_btn.clicked.connect(self.toggle_voc_collection)
        button_layout.addWidget(self.voc_connect_btn, 0, 1)
        self.voc_reset_btn = QtWidgets.QPushButton("重置事件", self.voc_mapping_group)
        self.voc_reset_btn.setMinimumHeight(30)
        self.voc_reset_btn.clicked.connect(lambda: self.reset_voc_events(clear_curves=False))
        button_layout.addWidget(self.voc_reset_btn, 1, 0, 1, 2)
        for button in (self.voc_refresh_btn, self.voc_connect_btn, self.voc_reset_btn):
            button.setStyleSheet(mapping_label_style)
        mapping_layout.addLayout(button_layout)

        port_grid = QtWidgets.QGridLayout()
        port_grid.setHorizontalSpacing(8)
        port_grid.setVerticalSpacing(9)
        port_grid.setColumnStretch(0, 1)
        port_grid.setColumnStretch(1, 1)
        port_grid.setRowMinimumHeight(0, 24)
        mapping_header_style = 'font: bold 10pt "KaiTi"; color: rgb(65, 65, 65);'
        position_header = QtWidgets.QLabel("位置", self.voc_mapping_group)
        position_header.setAlignment(QtCore.Qt.AlignCenter)
        position_header.setStyleSheet(mapping_header_style)
        port_header = QtWidgets.QLabel("对应串口", self.voc_mapping_group)
        port_header.setAlignment(QtCore.Qt.AlignCenter)
        port_header.setStyleSheet(mapping_header_style)
        port_grid.addWidget(position_header, 0, 0)
        port_grid.addWidget(port_header, 0, 1)

        self.voc_status_group = QtWidgets.QGroupBox(
            "事件触发状态", self.vehicle_dashboard
        )
        self.voc_status_group.setStyleSheet(panel_style)
        self.voc_status_group.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        status_root_layout = QtWidgets.QVBoxLayout(self.voc_status_group)
        self.voc_status_root_layout = status_root_layout
        status_root_layout.setContentsMargins(7, 15, 7, 9)
        status_root_layout.setSpacing(6)
        status_grid = QtWidgets.QGridLayout()
        self.voc_status_grid = status_grid
        status_grid.setHorizontalSpacing(4)
        status_grid.setVerticalSpacing(5)
        for column, stretch in enumerate((2, 2, 2, 2, 4, 5)):
            status_grid.setColumnStretch(column, stretch)
        status_grid.setRowMinimumHeight(0, 26)
        headers = ("位置", "R", "ΔR", "检测", "事件 / 簇", "融合状态")
        for col, title in enumerate(headers):
            header_label = QtWidgets.QLabel(title, self.voc_status_group)
            header_label.setAlignment(QtCore.Qt.AlignCenter)
            header_label.setStyleSheet(
                'font: bold 10pt "KaiTi"; color: rgb(65, 65, 65);'
            )
            status_grid.addWidget(header_label, 0, col)

        for row, (key, label) in enumerate(VOC_POSITIONS, start=1):
            map_label = QtWidgets.QLabel(label, self.voc_mapping_group)
            map_label.setAlignment(QtCore.Qt.AlignCenter)
            map_label.setStyleSheet('font: bold 10pt "KaiTi";')
            combo = QtWidgets.QComboBox(self.voc_mapping_group)
            combo.setMinimumWidth(105)
            combo.setMinimumHeight(30)
            combo.setStyleSheet(mapping_label_style)
            self.voc_port_combos[key] = combo
            port_grid.addWidget(map_label, row, 0)
            port_grid.addWidget(combo, row, 1)
            port_grid.setRowMinimumHeight(row, 30)

            pos_label = QtWidgets.QLabel(label, self.voc_status_group)
            pos_label.setAlignment(QtCore.Qt.AlignCenter)
            pos_label.setStyleSheet('font: bold 10pt "KaiTi";')
            status_grid.addWidget(pos_label, row, 0)

            r_label = QtWidgets.QLabel("--", self.voc_status_group)
            dr_label = QtWidgets.QLabel("--", self.voc_status_group)
            event_label = QtWidgets.QLabel("", self.voc_status_group)
            event_id_label = QtWidgets.QLabel("-- / --", self.voc_status_group)
            time_label = QtWidgets.QLabel("--", self.voc_status_group)
            fusion_state_label = QtWidgets.QLabel("--", self.voc_status_group)
            status_label = QtWidgets.QLabel("未连接", self.voc_status_group)
            for value_label in (r_label, dr_label, event_label):
                value_label.setAlignment(QtCore.Qt.AlignCenter)
                value_label.setMinimumWidth(38)
                value_label.setMinimumHeight(28)
                value_label.setMaximumHeight(32)
                value_label.setStyleSheet('font: 10pt "KaiTi";')
            event_label.setStyleSheet(
                "background: rgb(204, 238, 207); "
                "border: 1px solid rgb(126, 177, 132);"
            )
            event_label.setToolTip("未触发事件")

            event_cell = QtWidgets.QWidget(self.voc_status_group)
            event_cell_layout = QtWidgets.QVBoxLayout(event_cell)
            event_cell_layout.setContentsMargins(0, 0, 0, 0)
            event_cell_layout.setSpacing(2)
            event_id_label.setAlignment(QtCore.Qt.AlignCenter)
            time_label.setAlignment(QtCore.Qt.AlignCenter)
            event_id_label.setStyleSheet(
                'font: bold 9pt "KaiTi"; color: rgb(45, 75, 115);'
            )
            time_label.setStyleSheet('font: 8pt "KaiTi"; color: rgb(100, 70, 50);')
            event_cell_layout.addWidget(event_id_label)
            event_cell_layout.addWidget(time_label)

            fusion_cell = QtWidgets.QWidget(self.voc_status_group)
            fusion_layout = QtWidgets.QVBoxLayout(fusion_cell)
            fusion_layout.setContentsMargins(0, 0, 0, 0)
            fusion_layout.setSpacing(2)
            fusion_state_label.setAlignment(QtCore.Qt.AlignCenter)
            status_label.setAlignment(QtCore.Qt.AlignCenter)
            fusion_state_label.setStyleSheet(
                'font: bold 9pt "KaiTi"; color: rgb(115, 85, 35);'
            )
            status_label.setStyleSheet('font: 8pt "KaiTi"; color: rgb(70, 70, 70);')
            fusion_layout.addWidget(fusion_state_label)
            fusion_layout.addWidget(status_label)

            self.voc_r_labels[key] = r_label
            self.voc_dr_labels[key] = dr_label
            self.voc_event_labels[key] = event_label
            self.voc_time_labels[key] = time_label
            self.voc_event_id_labels[key] = event_id_label
            self.voc_fusion_state_labels[key] = fusion_state_label
            self.voc_status_labels[key] = status_label
            status_grid.addWidget(r_label, row, 1)
            status_grid.addWidget(dr_label, row, 2)
            status_grid.addWidget(event_label, row, 3)
            status_grid.addWidget(event_cell, row, 4)
            status_grid.addWidget(fusion_cell, row, 5)
            status_grid.setRowMinimumHeight(row, 46)

        mapping_layout.addLayout(port_grid)
        status_root_layout.addLayout(status_grid)
        self.voc_decision_label = QtWidgets.QLabel("融合判断：--", self.voc_status_group)
        self.voc_decision_label.setAlignment(QtCore.Qt.AlignCenter)
        self.voc_decision_label.setWordWrap(True)
        self.voc_decision_label.setMinimumHeight(24)
        self.voc_decision_label.setStyleSheet(
            'font: bold 11pt "KaiTi"; color: rgb(130, 70, 30);'
        )
        status_root_layout.addWidget(self.voc_decision_label)

        self.voc_timeline_summary_label = QtWidgets.QLabel(
            "事件簇时间轴：等待电子鼻 T_onset", self.voc_status_group
        )
        self.voc_timeline_summary_label.setAlignment(
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        self.voc_timeline_summary_label.setWordWrap(True)
        self.voc_timeline_summary_label.setStyleSheet(
            'font: 9pt "KaiTi"; color: rgb(65, 65, 65);'
        )
        status_root_layout.addWidget(self.voc_timeline_summary_label)

        self.voc_timeline_plot = pg.PlotWidget(self.voc_status_group)
        self.voc_timeline_plot.setBackground("w")
        self.voc_timeline_plot.setMinimumHeight(105)
        self.voc_timeline_plot.setMaximumHeight(145)
        self.voc_timeline_plot.setMouseEnabled(x=False, y=False)
        self.voc_timeline_plot.setMenuEnabled(False)
        self.voc_timeline_plot.showGrid(x=True, y=False, alpha=0.25)
        self.voc_timeline_plot.setXRange(
            -VOC_ONSET_LOOKBACK_SEC,
            VOC_ONSET_LOOKAHEAD_SEC,
            padding=0,
        )
        self.voc_timeline_plot.setYRange(-0.45, 2.45, padding=0)
        self.voc_timeline_plot.setLabel("bottom", "相对 T_onset (s)")
        self.voc_timeline_plot.getAxis("left").setTicks(
            [[(2, "主驾"), (1, "副驾"), (0, "后排")]]
        )
        self.voc_timeline_plot.getAxis("left").setWidth(42)
        self.voc_timeline_anchor_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            movable=False,
            pen=pg.mkPen((190, 55, 45), width=1.5, style=QtCore.Qt.DashLine),
        )
        self.voc_timeline_anchor_line.setToolTip("电子鼻开始响应时间 T_onset")
        self.voc_timeline_plot.addItem(
            self.voc_timeline_anchor_line, ignoreBounds=True
        )
        status_root_layout.addWidget(self.voc_timeline_plot, 1)

        position_panels = {}
        for idx, (key, label) in enumerate(VOC_POSITIONS):
            panel = QtWidgets.QFrame(self.vehicle_dashboard)
            panel.setObjectName("vocPositionPanel")
            panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
            panel.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
            )
            panel.setStyleSheet(
                "QFrame#vocPositionPanel { "
                "background: white; border: 1px solid rgb(185, 185, 185); }"
            )
            panel_layout = QtWidgets.QVBoxLayout(panel)
            panel_layout.setContentsMargins(5, 4, 5, 5)
            panel_layout.setSpacing(3)

            header_layout = QtWidgets.QHBoxLayout()
            header_layout.setContentsMargins(0, 0, 0, 0)
            position_label = QtWidgets.QLabel(label, panel)
            position_label.setStyleSheet(
                'font: bold 11pt "KaiTi"; color: rgb(45, 45, 45); border: none;'
            )
            event_badge = QtWidgets.QLabel("", panel)
            event_badge.setAlignment(QtCore.Qt.AlignCenter)
            event_badge.setFixedSize(42, 20)
            event_badge.setStyleSheet(
                "background: rgb(204, 238, 207); "
                "border: 1px solid rgb(126, 177, 132);"
            )
            event_badge.setToolTip("未触发事件")
            self.voc_plot_event_labels[key] = event_badge
            header_layout.addWidget(position_label)
            header_layout.addStretch(1)
            header_layout.addWidget(event_badge)
            panel_layout.addLayout(header_layout)

            plot = pg.PlotWidget(panel)
            plot.setMinimumHeight(MAIN_SIGNAL_PLOT_HEIGHT)
            plot.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
            )
            plot.setBackground("w")
            plot.setClipToView(True)
            plot.setDownsampling(auto=True, mode="peak")
            _apply_standard_plot_axes(plot)
            plot.setLabel("left", f"{label} R", color="black")
            plot.setLabel("bottom", "Time", color="black")
            curve = plot.plot(
                pen=pg.mkPen(color=(40 + idx * 70, 90, 210 - idx * 45), width=2)
            )
            tighten_plot_layout(plot, left_axis_width=48, bottom_axis_height=24)
            self.voc_plot_widgets[key] = plot
            self.voc_curves[key] = curve
            panel_layout.addWidget(plot, 1)
            position_panels[key] = panel

        dashboard_layout.addWidget(position_panels["driver"], 0, 0)
        dashboard_layout.addWidget(self.e_nose_control_group, 0, 1, 1, 2)
        dashboard_layout.addWidget(position_panels["passenger"], 0, 3)

        mapping_slot = QtWidgets.QWidget(self.vehicle_dashboard)
        mapping_slot_layout = QtWidgets.QVBoxLayout(mapping_slot)
        mapping_slot_layout.setContentsMargins(0, 0, 0, 0)
        mapping_slot_layout.addWidget(self.voc_mapping_group)
        dashboard_layout.addWidget(mapping_slot, 1, 0)

        status_slot = QtWidgets.QWidget(self.vehicle_dashboard)
        status_slot_layout = QtWidgets.QVBoxLayout(status_slot)
        status_slot_layout.setContentsMargins(0, 0, 0, 0)
        status_slot_layout.addWidget(self.voc_status_group)
        dashboard_layout.addWidget(status_slot, 1, 3)

        if hasattr(self, "e_nose_display_group"):
            self.e_nose_display_group.setParent(self.vehicle_dashboard)
            dashboard_layout.addWidget(self.e_nose_display_group, 1, 1, 1, 2)

        if hasattr(self, "temperature_humidity_plot"):
            aux_panel = QtWidgets.QFrame(self.vehicle_dashboard)
            aux_panel.setObjectName("vocPositionPanel")
            aux_panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
            aux_panel.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
            )
            aux_panel.setStyleSheet(
                "QFrame#vocPositionPanel { "
                "background: white; border: 1px solid rgb(185, 185, 185); }"
            )
            aux_layout = QtWidgets.QVBoxLayout(aux_panel)
            # Sensor17 原始 PlotWidget 自带边框占位，少留 2px 底边使曲线区与单通道等高。
            aux_layout.setContentsMargins(5, 4, 5, 3)
            aux_layout.setSpacing(3)
            aux_header_layout = QtWidgets.QHBoxLayout()
            aux_header_layout.setContentsMargins(0, 0, 0, 0)
            aux_title = QtWidgets.QLabel("温湿度", aux_panel)
            aux_title.setStyleSheet(
                'font: bold 11pt "KaiTi"; color: rgb(45, 45, 45); border: none;'
            )
            aux_header_layout.addWidget(aux_title)
            aux_header_layout.addStretch(1)
            if hasattr(self, "temperature_humidity_value_label"):
                self.temperature_humidity_value_label.show()
                self.temperature_humidity_value_label.setAlignment(
                    QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
                )
                self.temperature_humidity_value_label.setMinimumHeight(18)
                self.temperature_humidity_value_label.setMaximumHeight(20)
                aux_header_layout.addWidget(self.temperature_humidity_value_label)
            aux_layout.addLayout(aux_header_layout)
            self.temperature_humidity_plot.show()
            self.temperature_humidity_plot.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
            )
            aux_layout.addWidget(self.temperature_humidity_plot, 1)
            dashboard_layout.addWidget(aux_panel, 2, 0)

        if hasattr(self, "main_inference_result_group"):
            self.main_inference_result_group.setParent(self.vehicle_dashboard)
            dashboard_layout.addWidget(self.main_inference_result_group, 2, 3)

        rear_slot = QtWidgets.QWidget(self.vehicle_dashboard)
        rear_layout = QtWidgets.QHBoxLayout(rear_slot)
        rear_layout.setContentsMargins(0, 0, 0, 0)
        rear_layout.setSpacing(0)
        rear_layout.addStretch(1)
        rear_layout.addWidget(position_panels["rear"], 2)
        rear_layout.addStretch(1)
        dashboard_layout.addWidget(rear_slot, 2, 1, 1, 2)

        for column in range(4):
            dashboard_layout.setColumnStretch(column, 1)
        dashboard_layout.setRowStretch(0, 2)
        dashboard_layout.setRowStretch(1, 5)
        dashboard_layout.setRowStretch(2, 2)

        self.gridLayout.addWidget(self.vehicle_dashboard, 0, 0, 1, 4)
        try:
            self.gridLayout.setRowStretch(0, 1)
            self.gridLayout.setRowStretch(1, 0)
            self.gridLayout.setRowStretch(2, 0)
        except Exception:
            pass
        self.refresh_voc_port_combos()
        self.update_driver_drinking_display()

    @staticmethod
    def _combo_current_port(combo):
        if combo is None:
            return ""
        data = combo.currentData()
        if data:
            return str(data)
        text = combo.currentText().strip()
        if " - " in text:
            return text.split(" - ", 1)[0].strip()
        return text

    def refresh_voc_port_combos(self, ports=None):
        if not getattr(self, "voc_port_combos", None):
            return
        if ports is None:
            ports = [port.device for port in serial.tools.list_ports.comports()]
        for combo in self.voc_port_combos.values():
            current = self._combo_current_port(combo)
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("未选择", "")
            for port in ports:
                combo.addItem(str(port), str(port))
            if current in ports:
                combo.setCurrentIndex(ports.index(current) + 1)
            combo.blockSignals(False)

    def set_voc_controls_enabled(self, enabled):
        for combo in getattr(self, "voc_port_combos", {}).values():
            combo.setEnabled(enabled)
        for attr in ("voc_baud_combo", "voc_interval_spin", "voc_refresh_btn"):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.setEnabled(enabled)

    def _set_voc_event_flag(self, key, triggered):
        """同步物理检测色块；颜色不再表示事件是否已被融合消费。"""
        triggered = bool(triggered)
        for labels in (self.voc_event_labels, self.voc_plot_event_labels):
            label = labels.get(key)
            if label is None:
                continue
            label.setText("")
            label.setToolTip(
                "传感器当前处于事件响应状态"
                if triggered
                else "传感器当前未处于事件响应状态"
            )
            if triggered:
                label.setStyleSheet(
                    "background: rgb(205, 65, 55); "
                    "border: 1px solid rgb(165, 42, 36);"
                )
            else:
                label.setStyleSheet(
                    "background: rgb(204, 238, 207); "
                    "border: 1px solid rgb(126, 177, 132);"
                )

    @staticmethod
    def _voc_event_state_style(state):
        styles = {
            "待匹配": ((150, 100, 25), QtCore.Qt.DashLine),
            "融合中": ((35, 95, 165), QtCore.Qt.SolidLine),
            "候选待定": ((75, 105, 145), QtCore.Qt.DashLine),
            "已采用": ((35, 135, 75), QtCore.Qt.SolidLine),
            "已排除": ((125, 125, 125), QtCore.Qt.DashLine),
            "已失效": ((165, 75, 65), QtCore.Qt.DotLine),
            "已过期": ((150, 150, 150), QtCore.Qt.DotLine),
        }
        return styles.get(str(state or ""), ((90, 90, 90), QtCore.Qt.SolidLine))

    @staticmethod
    def _voc_event_prefix(key):
        return {
            "driver": "D",
            "passenger": "P",
            "rear": "R",
        }.get(key, "E")

    def _register_voc_event(self, key, event):
        if not isinstance(event, dict):
            return None
        if not event.get("event_uid"):
            event["event_uid"] = "{}{:03d}".format(
                self._voc_event_prefix(key),
                int(event.get("event_id", 0)),
            )
        event["position_key"] = key
        event["position_label"] = dict(VOC_POSITIONS).get(key, key)
        event.setdefault("cluster_id", None)
        event.setdefault("fusion_state", "待匹配")
        event.setdefault("fusion_reason", "等待电子鼻酒精响应")
        event.setdefault("fusion_updated_at", float(event["confirm_timestamp"]))
        return event

    def _set_voc_event_lifecycle(
        self,
        event,
        state,
        reason,
        now_timestamp=None,
        cluster_id=None,
    ):
        if not isinstance(event, dict):
            return
        event["fusion_state"] = str(state or "--")
        event["fusion_reason"] = str(reason or "")
        event["fusion_updated_at"] = (
            time.time() if now_timestamp is None else float(now_timestamp)
        )
        if cluster_id is not None:
            event["cluster_id"] = cluster_id

    @staticmethod
    def _voc_event_clock_text(timestamp):
        try:
            return datetime.fromtimestamp(float(timestamp)).strftime("%H:%M:%S")
        except (TypeError, ValueError, OSError):
            return "--"

    def _latest_voc_event(self, key):
        detector = self.voc_event_detectors.get(key)
        if detector is None or not detector.events:
            return None
        event = detector.events[-1]
        return self._register_voc_event(key, event)

    def _expire_unmatched_voc_events(self, now_timestamp):
        """超过 T_onset 回看范围且未进入融合的事件保留记录并标为过期。"""
        now_timestamp = float(now_timestamp)
        for key, _ in VOC_POSITIONS:
            detector = self.voc_event_detectors.get(key)
            if detector is None:
                continue
            for event in detector.events:
                self._register_voc_event(key, event)
                if event.get("used_by_recognition") is not None:
                    continue
                if event.get("fusion_state") not in (None, "", "待匹配", "已过期"):
                    continue
                age = now_timestamp - float(event["start_timestamp"])
                expire_sec = float(VOC_UNMATCHED_EVENT_EXPIRE_SEC)
                if age > expire_sec:
                    self._set_voc_event_lifecycle(
                        event,
                        "已过期",
                        "超过未匹配保留窗口 {:.0f}s，未匹配到酒精响应".format(
                            expire_sec
                        ),
                        now_timestamp,
                    )

    def _refresh_voc_event_row(self, key):
        event = self._latest_voc_event(key)
        event_id_label = self.voc_event_id_labels.get(key)
        time_label = self.voc_time_labels.get(key)
        fusion_label = self.voc_fusion_state_labels.get(key)
        if event is None:
            if event_id_label is not None:
                event_id_label.setText("-- / --")
                event_id_label.setToolTip("")
            if time_label is not None:
                time_label.setText("--")
                time_label.setToolTip("")
            if fusion_label is not None:
                fusion_label.setText("--")
                fusion_label.setToolTip("")
            return

        event_uid = str(event.get("event_uid") or "--")
        cluster_id = str(event.get("cluster_id") or "--")
        state = str(event.get("fusion_state") or "待匹配")
        reason = str(event.get("fusion_reason") or "")
        start_text = self.format_record_time(event.get("start_timestamp"))
        confirm_text = self.format_record_time(event.get("confirm_timestamp"))
        detail = (
            f"事件：{event_uid}\n事件簇：{cluster_id}\n"
            f"开始：{start_text}\n确认：{confirm_text}\n"
            f"状态：{state}\n原因：{reason or '--'}"
        )
        if event_id_label is not None:
            event_id_label.setText(f"{event_uid} / {cluster_id}")
            event_id_label.setToolTip(detail)
        if time_label is not None:
            time_label.setText(self._voc_event_clock_text(event.get("start_timestamp")))
            time_label.setToolTip(detail)
        if fusion_label is not None:
            color, _ = self._voc_event_state_style(state)
            color_css = "rgb({}, {}, {})".format(*color)
            fusion_label.setText(state)
            fusion_label.setStyleSheet(
                f'font: bold 9pt "KaiTi"; color: {color_css};'
            )
            fusion_label.setToolTip(detail)

    def _clear_voc_curve_event_markers(self, key):
        plot = self.voc_plot_widgets.get(key)
        for item in self.voc_curve_event_marker_items.get(key, []):
            if plot is not None:
                try:
                    plot.removeItem(item)
                except Exception:
                    pass
        self.voc_curve_event_marker_items[key] = []

    def _refresh_voc_curve_event_markers(self, key):
        plot = self.voc_plot_widgets.get(key)
        detector = self.voc_event_detectors.get(key)
        if plot is None or detector is None:
            return
        x_values = self.voc_xdata.get(key, [])
        min_visible_x = float(x_values[0]) if x_values else float("-inf")
        events = []
        for event in detector.events:
            self._register_voc_event(key, event)
            if float(event.get("start_sample_index", -1)) >= min_visible_x:
                events.append(event)
        events = events[-6:]
        signature = tuple(
            (
                event.get("event_uid"),
                event.get("cluster_id"),
                event.get("fusion_state"),
                event.get("start_sample_index"),
            )
            for event in events
        )
        if signature == self.voc_curve_event_marker_signatures.get(key):
            return
        self._clear_voc_curve_event_markers(key)
        marker_items = []
        for event in events:
            state = str(event.get("fusion_state") or "待匹配")
            color, pen_style = self._voc_event_state_style(state)
            marker = pg.InfiniteLine(
                pos=float(event["start_sample_index"]),
                angle=90,
                movable=False,
                pen=pg.mkPen(color, width=1.4, style=pen_style),
            )
            marker.setZValue(20)
            marker.setToolTip(
                "{} / {}：{}\n{}".format(
                    event.get("event_uid", "--"),
                    event.get("cluster_id") or "--",
                    state,
                    event.get("fusion_reason") or "",
                )
            )
            plot.addItem(marker, ignoreBounds=True)
            marker_items.append(marker)
        self.voc_curve_event_marker_items[key] = marker_items
        self.voc_curve_event_marker_signatures[key] = signature

    def _refresh_voc_event_indicators(self, now_timestamp=None):
        """刷新物理检测、事件生命周期文字、曲线标记和事件簇时间轴。"""
        now_timestamp = time.time() if now_timestamp is None else float(now_timestamp)
        self._expire_unmatched_voc_events(now_timestamp)
        for key, _ in VOC_POSITIONS:
            detector = self.voc_event_detectors.get(key)
            if detector is None:
                continue
            detector.prune_events(now_timestamp)
            self._set_voc_event_flag(key, detector.indicator_active(now_timestamp))
            self._refresh_voc_event_row(key)
            self._refresh_voc_curve_event_markers(key)
        self._refresh_voc_timeline()

    def toggle_voc_collection(self):
        if self.voc_threads:
            self.stop_voc_collection()
        else:
            self.start_voc_collection()

    def start_voc_collection(self):
        ports_by_key = {}
        missing = []
        for key, label in VOC_POSITIONS:
            port = self._combo_current_port(self.voc_port_combos.get(key))
            if not port:
                missing.append(label)
            ports_by_key[key] = port
        if missing:
            QtWidgets.QMessageBox.warning(
                self,
                "提示",
                "请为以下位置选择单通道VOC串口：" + "、".join(missing),
            )
            return

        selected_ports = [port for port in ports_by_key.values() if port]
        if len(set(selected_ports)) != len(selected_ports):
            QtWidgets.QMessageBox.warning(self, "提示", "3路VOC不能使用重复串口。")
            return

        nose_port = str(self.reflesh_comboBox.currentText() or "").strip()
        if self.serial_thread and self.serial_thread.isRunning() and nose_port in selected_ports:
            QtWidgets.QMessageBox.warning(
                self,
                "提示",
                "单通道VOC串口不能与16通道电子鼻串口重复。",
            )
            return

        try:
            baudrate = int(self.voc_baud_combo.currentText())
        except (TypeError, ValueError):
            baudrate = VOC_DEFAULT_BAUDRATE
        interval = float(self.voc_interval_spin.value())

        self.reset_voc_events(clear_curves=True)
        self.voc_error_keys.clear()
        for key, label in VOC_POSITIONS:
            thread = SingleVocSerialThread(
                key,
                label,
                ports_by_key[key],
                baudrate,
                interval,
            )
            thread.sample_received.connect(self.on_voc_sample_received)
            thread.status_changed.connect(self.on_voc_status_changed)
            thread.error_occurred.connect(self.on_voc_error)
            thread.finished.connect(
                lambda key=key, thread=thread: self.on_voc_thread_finished(key, thread)
            )
            self.voc_threads[key] = thread
            self.voc_status_labels[key].setText("正在连接")
            thread.start()

        self.set_voc_controls_enabled(False)
        self.voc_connect_btn.setText("断开3路VOC")
        if hasattr(self, "statusbar"):
            self.statusbar.showMessage("3路单通道VOC采集已启动。", 3000)

    def stop_voc_collection(self):
        threads = dict(getattr(self, "voc_threads", {}))
        self.voc_threads.clear()
        self.voc_error_keys.clear()
        for key, thread in threads.items():
            try:
                thread.stop()
            except Exception:
                pass
            if key in self.voc_status_labels:
                self.voc_status_labels[key].setText("已断开")
        self.set_voc_controls_enabled(True)
        if hasattr(self, "voc_connect_btn"):
            self.voc_connect_btn.setText("连接3路VOC")

    def on_voc_thread_finished(self, key, thread=None):
        if thread is None:
            thread = self.sender()
        if self.voc_threads.get(key) is thread:
            self.voc_threads.pop(key, None)
            if key in self.voc_status_labels:
                if key in self.voc_error_keys:
                    self.voc_error_keys.discard(key)
                else:
                    self.voc_status_labels[key].setText("已断开")
        if not self.voc_threads:
            self.set_voc_controls_enabled(True)
            if hasattr(self, "voc_connect_btn"):
                self.voc_connect_btn.setText("连接3路VOC")

    def reset_voc_events(self, clear_curves=True):
        for key, _ in VOC_POSITIONS:
            self._clear_voc_curve_event_markers(key)
            self.voc_curve_event_marker_signatures[key] = None
            detector = self.voc_event_detectors.get(key)
            if detector is not None:
                detector.reset()
            if clear_curves:
                self.voc_sample_index[key] = 0
                self.voc_xdata[key].clear()
                self.voc_rdata[key].clear()
                curve = self.voc_curves.get(key)
                if curve is not None:
                    curve.clear()
            if key in self.voc_r_labels:
                self.voc_r_labels[key].setText("--")
            if key in self.voc_dr_labels:
                self.voc_dr_labels[key].setText("--")
            self._set_voc_event_flag(key, False)
            if key in self.voc_time_labels:
                self.voc_time_labels[key].setText("--")
                self.voc_time_labels[key].setToolTip("")
            if key in self.voc_event_id_labels:
                self.voc_event_id_labels[key].setText("-- / --")
                self.voc_event_id_labels[key].setToolTip("")
            if key in self.voc_fusion_state_labels:
                self.voc_fusion_state_labels[key].setText("--")
                self.voc_fusion_state_labels[key].setToolTip("")
        self.driver_drinking_decision = "--"
        self.voc_fusion_results.clear()
        self.voc_fusion_first_seen_at.clear()
        self.voc_active_recognition_id = None
        self.voc_visual_recognition_id = None
        self.voc_visual_anchor_timestamp = None
        self.voc_visual_anchor_source = None
        self.voc_visual_fusion_state = None
        self.voc_cluster_sequence = 0
        self.voc_cluster_records.clear()
        self.voc_timeline_signature = None
        self._clear_voc_timeline_items()
        self.update_driver_drinking_display()

    def on_voc_status_changed(self, key, message):
        label = self.voc_status_labels.get(key)
        if label is not None:
            message_text = str(message or "--")
            label.setText(message_text)
            if message_text.startswith(("已连接", "已重连")):
                self.voc_error_keys.discard(key)
                label.setToolTip("")

    def on_voc_error(self, key, message):
        self.voc_error_keys.add(key)
        label = self.voc_status_labels.get(key)
        if label is not None:
            label.setText("错误")
            label.setToolTip(str(message or ""))
        if hasattr(self, "statusbar"):
            self.statusbar.showMessage(str(message or ""), 5000)

    def _append_voc_raw_save_sample(self, key, sample, r_value):
        """保存开启时按各路VOC自身到数节奏缓存原始电阻。"""
        if not self.is_saving:
            return
        writer = self.voc_csv_writers.get(key)
        if writer is None:
            self.abort_saving_due_to_error(
                f"{dict(VOC_POSITIONS).get(key, key)} VOC保存文件未就绪。"
            )
            return
        buffer = self.voc_save_buffers.setdefault(key, [])
        if len(buffer) >= SAVE_BUFFER_MAX_ROWS:
            self.abort_saving_due_to_error(
                f"{dict(VOC_POSITIONS).get(key, key)} VOC保存缓存超过 "
                f"{SAVE_BUFFER_MAX_ROWS} 行，已停止保存。"
            )
            return

        timestamp = sample.get("timestamp", time.time())
        time_text = str(sample.get("time", "") or "").strip()
        if not time_text:
            time_text = self.format_record_time(timestamp)
        port = str(sample.get("port", "") or "").strip()
        if not port:
            port = self._combo_current_port(self.voc_port_combos.get(key))
        next_sequence = (
            int(self.voc_save_rows_written.get(key, 0))
            + len(buffer)
            + 1
        )
        buffer.append(
            [
                next_sequence,
                time_text,
                dict(VOC_POSITIONS).get(key, key),
                port,
                r_value if np.isfinite(r_value) else "",
            ]
        )
        self.update_data_collection_save_status()

    def on_voc_sample_received(self, key, sample):
        if key not in self.voc_event_detectors:
            return
        idx = int(self.voc_sample_index.get(key, 0))
        timestamp = float(sample.get("timestamp", time.time()))
        r_value = sample.get("r_value", float("nan"))
        try:
            r_float = float(r_value)
        except (TypeError, ValueError):
            r_float = float("nan")

        self._append_limited(self.voc_xdata[key], idx)
        self._append_limited(self.voc_rdata[key], r_float)
        curve = self.voc_curves.get(key)
        if curve is not None:
            curve.setData(self.voc_xdata[key], self.voc_rdata[key])

        self._append_voc_raw_save_sample(key, sample, r_float)
        state = self.voc_event_detectors[key].update(idx, timestamp, r_float)
        if state.get("event") is not None:
            self._register_voc_event(key, state["event"])
        self.voc_sample_index[key] = idx + 1

        if key in self.voc_r_labels:
            self.voc_r_labels[key].setText(self._format_number(r_float, digits=2))
        if key in self.voc_dr_labels:
            self.voc_dr_labels[key].setText(self._format_number(state.get("dr"), digits=2))

        self._set_voc_event_flag(key, state.get("indicator_active"))
        if key in self.voc_status_labels:
            self.voc_status_labels[key].setText(f"已采集 {idx + 1} 条")
            self.voc_status_labels[key].setToolTip("")
        self.voc_error_keys.discard(key)

        if state.get("triggered_now") and hasattr(self, "statusbar"):
            position_label = dict(VOC_POSITIONS).get(key, key)
            self.statusbar.showMessage(f"{position_label} VOC事件已触发。", 3000)

        self.update_driver_drinking_display()

    @staticmethod
    def _recognition_timestamp_seconds(result):
        if not isinstance(result, dict):
            return None
        try:
            timestamp_ms = float(result.get("timestamp_ms"))
        except (TypeError, ValueError):
            timestamp_ms = float("nan")
        if np.isfinite(timestamp_ms):
            return timestamp_ms / 1000.0
        return None

    @staticmethod
    def _response_onset_timestamp_seconds(result):
        if not isinstance(result, dict):
            return None
        try:
            timestamp_ms = float(result.get("response_onset_timestamp_ms"))
        except (TypeError, ValueError):
            timestamp_ms = float("nan")
        if np.isfinite(timestamp_ms):
            return timestamp_ms / 1000.0
        return None

    def _recognition_result_for_voc_fusion(self):
        """融合固定使用最近一次完整识别，避免寻峰等待帧更新时间导致窗口漂移。"""
        live_result = getattr(self, "last_inference_result", None)
        if not live_result:
            return None
        if (
            self.is_inference_recognition_wait_frame(live_result)
            and self.last_committed_recognition_result
        ):
            return self.last_committed_recognition_result
        return live_result

    def _voc_events_for_recognition(
        self,
        recognition_timestamp,
        recognition_id,
        now_timestamp,
        response_onset_timestamp=None,
    ):
        if response_onset_timestamp is None:
            anchor_timestamp = float(recognition_timestamp)
            window_start = anchor_timestamp - VOC_FUSION_LOOKBACK_SEC
            window_end = anchor_timestamp + VOC_FUSION_LOOKAHEAD_SEC
            require_confirmation_in_window = True
        else:
            anchor_timestamp = float(response_onset_timestamp)
            window_start = anchor_timestamp - VOC_ONSET_LOOKBACK_SEC
            window_end = anchor_timestamp + VOC_ONSET_LOOKAHEAD_SEC
            require_confirmation_in_window = False
        candidates = []
        labels_by_key = dict(VOC_POSITIONS)
        for key, _ in VOC_POSITIONS:
            detector = self.voc_event_detectors.get(key)
            if detector is None:
                continue
            detector.prune_events(now_timestamp)
            for event in detector.events:
                used_by = event.get("used_by_recognition")
                if used_by not in (None, recognition_id):
                    continue
                start_timestamp = float(event["start_timestamp"])
                confirm_timestamp = float(event["confirm_timestamp"])
                if not (window_start <= start_timestamp <= window_end):
                    continue
                if require_confirmation_in_window and confirm_timestamp > window_end:
                    continue
                candidates.append(
                    {
                        "position_key": key,
                        "position_label": labels_by_key.get(key, key),
                        "event": event,
                    }
                )
        return candidates

    @staticmethod
    def _split_voc_event_clusters(candidates):
        """按相邻事件间隔切簇，并限制单簇最大时间跨度。"""
        if not candidates:
            return []
        ordered = sorted(
            candidates,
            key=lambda item: float(item["event"]["start_timestamp"]),
        )
        clusters = []
        current = []
        for item in ordered:
            start_timestamp = float(item["event"]["start_timestamp"])
            if current:
                previous_timestamp = float(
                    current[-1]["event"]["start_timestamp"]
                )
                cluster_start_timestamp = float(
                    current[0]["event"]["start_timestamp"]
                )
                if (
                    start_timestamp - previous_timestamp > VOC_EVENT_CLUSTER_GAP_SEC
                    or start_timestamp - cluster_start_timestamp
                    > VOC_EVENT_GROUP_MAX_SPAN_SEC
                ):
                    clusters.append(current)
                    current = []
            current.append(item)
        if current:
            clusters.append(current)
        return clusters

    @classmethod
    def _voc_event_cluster_details(cls, candidates, anchor_timestamp):
        """返回每个时间簇、每路最优事件及其相对响应起点的评分。"""
        details = []
        for cluster in cls._split_voc_event_clusters(candidates):
            by_position = {}
            for item in cluster:
                start_timestamp = float(item["event"]["start_timestamp"])
                key = item["position_key"]
                current = by_position.get(key)
                distance = abs(start_timestamp - anchor_timestamp)
                strength = float(item["event"].get("max_dr", 0.0))
                if current is None:
                    by_position[key] = item
                    continue
                current_distance = abs(
                    float(current["event"]["start_timestamp"]) - anchor_timestamp
                )
                current_strength = float(current["event"].get("max_dr", 0.0))
                if (distance, -strength) < (current_distance, -current_strength):
                    by_position[key] = item

            group = list(by_position.values())
            group_times = [
                float(item["event"]["start_timestamp"]) for item in group
            ]
            if not group_times:
                continue
            center_time = float(np.median(group_times))
            total_strength = sum(
                float(item["event"].get("max_dr", 0.0)) for item in group
            )
            score = (
                abs(center_time - anchor_timestamp),
                -len(group),
                -total_strength,
                center_time,
            )
            details.append(
                {
                    "items": list(cluster),
                    "group": group,
                    "center_timestamp": center_time,
                    "score": score,
                }
            )
        return details

    @classmethod
    def _best_voc_event_group(cls, candidates, anchor_timestamp):
        """优先选择中心时刻最接近电子鼻响应起点的事件簇。"""
        details = cls._voc_event_cluster_details(candidates, anchor_timestamp)
        if not details:
            return []
        return list(min(details, key=lambda detail: detail["score"])["group"])

    def _sync_voc_cluster_records(
        self,
        recognition_id,
        cluster_details,
        anchor_timestamp,
    ):
        """按事件重叠沿用簇编号，避免收集期内新增事件造成编号跳变。"""
        previous_records = list(
            self.voc_cluster_records.get(recognition_id, [])
        )
        available_indices = set(range(len(previous_records)))
        current_records = []
        for detail in cluster_details:
            for item in detail["items"]:
                self._register_voc_event(item["position_key"], item["event"])
            event_uids = {
                str(item["event"].get("event_uid"))
                for item in detail["items"]
            }
            best_index = None
            best_overlap = 0
            for index in sorted(available_indices):
                previous_uids = set(
                    previous_records[index].get("event_uids", set())
                )
                overlap = len(event_uids & previous_uids)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_index = index
            if best_index is None:
                self.voc_cluster_sequence += 1
                record = {
                    "cluster_id": f"C{self.voc_cluster_sequence:03d}",
                }
            else:
                available_indices.discard(best_index)
                record = previous_records[best_index]
            record.update(
                {
                    "recognition_id": recognition_id,
                    "anchor_timestamp": float(anchor_timestamp),
                    "items": list(detail["items"]),
                    "group": list(detail["group"]),
                    "event_uids": event_uids,
                    "center_timestamp": float(detail["center_timestamp"]),
                    "score": detail["score"],
                }
            )
            detail["cluster_record"] = record
            detail["cluster_id"] = record["cluster_id"]
            current_records.append(record)
        self.voc_cluster_records[recognition_id] = current_records
        if len(self.voc_cluster_records) > 20:
            removable_ids = [
                item
                for item in sorted(self.voc_cluster_records)
                if item not in (recognition_id, self.voc_visual_recognition_id)
            ]
            while len(self.voc_cluster_records) > 20 and removable_ids:
                self.voc_cluster_records.pop(removable_ids.pop(0), None)
        return cluster_details

    def _apply_voc_cluster_lifecycle(
        self,
        recognition_id,
        cluster_details,
        selected_detail,
        anchor_timestamp,
        final,
        now_timestamp,
    ):
        """把分簇结果写回事件，供状态表、曲线标记和时间轴共同显示。"""
        for detail in cluster_details:
            record = detail["cluster_record"]
            cluster_id = record["cluster_id"]
            selected = detail is selected_detail
            offset = float(detail["center_timestamp"]) - float(anchor_timestamp)
            selected_event_uids = {
                item["event"].get("event_uid") for item in detail["group"]
            }
            if final and selected:
                cluster_state = "已采用"
                cluster_reason = "距离 T_onset 最近，已用于饮酒位置判断"
            elif final:
                cluster_state = "已排除"
                cluster_reason = "存在更接近 T_onset 的事件簇"
            elif selected:
                cluster_state = "融合中"
                cluster_reason = "当前距离 T_onset 最近，等待收集窗结束"
            else:
                cluster_state = "候选待定"
                cluster_reason = "当前不是距离 T_onset 最近的事件簇"
            record.update(
                {
                    "state": cluster_state,
                    "reason": cluster_reason,
                    "selected": bool(selected),
                    "final": bool(final),
                    "offset_seconds": offset,
                    "updated_at": float(now_timestamp),
                }
            )

            for item in detail["items"]:
                event = item["event"]
                event_uid = event.get("event_uid")
                detector = self.voc_event_detectors.get(item["position_key"])
                if detector is not None:
                    detector.claim_event(event, recognition_id)
                if final and selected and event_uid in selected_event_uids:
                    state = "已采用"
                    reason = f"属于 {cluster_id}，已用于饮酒位置判断"
                elif final and selected:
                    state = "已排除"
                    reason = "同一位置存在更接近 T_onset 的事件"
                elif final:
                    state = "已排除"
                    reason = f"{cluster_id} 不是距离 T_onset 最近的事件簇"
                elif selected and event_uid in selected_event_uids:
                    state = "融合中"
                    reason = f"属于 {cluster_id}，等待收集窗结束"
                else:
                    state = "候选待定"
                    reason = f"属于 {cluster_id}，等待最终选簇"
                self._set_voc_event_lifecycle(
                    event,
                    state,
                    reason,
                    now_timestamp,
                    cluster_id,
                )

    def _invalidate_active_voc_fusion(self, reason, now_timestamp=None):
        recognition_id = self.voc_active_recognition_id
        if recognition_id is None:
            return
        now_timestamp = time.time() if now_timestamp is None else float(now_timestamp)
        for record in self.voc_cluster_records.get(recognition_id, []):
            if record.get("final"):
                continue
            record.update(
                {
                    "state": "已失效",
                    "reason": str(reason or "融合流程已中止"),
                    "selected": False,
                    "final": True,
                    "updated_at": now_timestamp,
                }
            )
            for item in record.get("items", []):
                event = item.get("event", {})
                if event.get("used_by_recognition") != recognition_id:
                    continue
                if event.get("fusion_state") in ("融合中", "候选待定"):
                    self._set_voc_event_lifecycle(
                        event,
                        "已失效",
                        str(reason or "融合流程已中止"),
                        now_timestamp,
                        record.get("cluster_id"),
                    )
        self.voc_active_recognition_id = None
        self.voc_fusion_first_seen_at.pop(recognition_id, None)
        if self.voc_visual_recognition_id == recognition_id:
            self.voc_visual_fusion_state = "已失效"
        self.voc_timeline_signature = None
        for key, _ in VOC_POSITIONS:
            self.voc_curve_event_marker_signatures[key] = None

    def _clear_voc_timeline_items(self):
        plot = getattr(self, "voc_timeline_plot", None)
        for item in getattr(self, "voc_timeline_dynamic_items", []):
            if plot is not None:
                try:
                    plot.removeItem(item)
                except Exception:
                    pass
        self.voc_timeline_dynamic_items = []

    def _refresh_voc_timeline(self):
        plot = getattr(self, "voc_timeline_plot", None)
        summary_label = getattr(self, "voc_timeline_summary_label", None)
        if plot is None:
            return
        recognition_id = self.voc_visual_recognition_id
        anchor_timestamp = self.voc_visual_anchor_timestamp
        records = list(self.voc_cluster_records.get(recognition_id, []))
        signature = (
            recognition_id,
            anchor_timestamp,
            self.voc_visual_anchor_source,
            self.voc_visual_fusion_state,
            tuple(
                (
                    record.get("cluster_id"),
                    record.get("state"),
                    tuple(
                        (
                            item["event"].get("event_uid"),
                            item["event"].get("fusion_state"),
                            item["event"].get("start_timestamp"),
                        )
                        for item in record.get("items", [])
                    ),
                )
                for record in records
            ),
        )
        if signature == self.voc_timeline_signature:
            return
        self.voc_timeline_signature = signature
        self._clear_voc_timeline_items()

        anchor_line = getattr(self, "voc_timeline_anchor_line", None)
        if anchor_timestamp is None:
            if anchor_line is not None:
                anchor_line.setVisible(False)
            if summary_label is not None:
                summary_label.setText("事件簇时间轴：等待电子鼻 T_onset")
            return
        if anchor_line is not None:
            anchor_line.setVisible(True)

        lookback = (
            VOC_FUSION_LOOKBACK_SEC
            if self.voc_visual_anchor_source == "recognition_timestamp"
            else VOC_ONSET_LOOKBACK_SEC
        )
        lookahead = (
            VOC_FUSION_LOOKAHEAD_SEC
            if self.voc_visual_anchor_source == "recognition_timestamp"
            else VOC_ONSET_LOOKAHEAD_SEC
        )
        plot.setXRange(-lookback, lookahead, padding=0)
        anchor_clock = self._voc_event_clock_text(anchor_timestamp)
        fusion_state = self.voc_visual_fusion_state or "--"
        if records:
            cluster_summary = " · ".join(
                "{} {}".format(
                    record.get("cluster_id", "--"),
                    record.get("state", "--"),
                )
                for record in records
            )
        else:
            if fusion_state == "已失效":
                cluster_summary = "未匹配到事件，本次融合已失效"
            elif fusion_state == "已完成":
                cluster_summary = "未匹配到事件，本次融合已完成"
            else:
                cluster_summary = "等待 VOC 事件"
        if summary_label is not None:
            summary_label.setText(
                f"事件簇时间轴：T_onset {anchor_clock} | "
                f"{fusion_state} | {cluster_summary}"
            )

        y_by_position = {"driver": 2.0, "passenger": 1.0, "rear": 0.0}
        dynamic_items = []
        for record in records:
            cluster_id = str(record.get("cluster_id") or "--")
            cluster_state = str(record.get("state") or "候选待定")
            cluster_color, cluster_pen_style = self._voc_event_state_style(
                cluster_state
            )
            points = []
            for item in record.get("items", []):
                event = item["event"]
                x_value = float(event["start_timestamp"]) - float(anchor_timestamp)
                y_value = y_by_position.get(item["position_key"], 0.0)
                points.append((x_value, y_value, event))
            points.sort(key=lambda point: point[0])
            if len(points) >= 2:
                connector = plot.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    pen=pg.mkPen(
                        cluster_color,
                        width=1.8,
                        style=cluster_pen_style,
                    ),
                )
                connector.setToolTip(
                    f"{cluster_id}：{cluster_state}\n{record.get('reason') or ''}"
                )
                dynamic_items.append(connector)
            for x_value, y_value, event in points:
                event_state = str(event.get("fusion_state") or "候选待定")
                event_color, _ = self._voc_event_state_style(event_state)
                symbol = {
                    "已采用": "o",
                    "融合中": "d",
                    "候选待定": "t",
                    "待匹配": "t",
                    "已排除": "x",
                    "已失效": "x",
                    "已过期": "x",
                }.get(event_state, "o")
                scatter = pg.ScatterPlotItem(
                    [x_value],
                    [y_value],
                    symbol=symbol,
                    size=10,
                    pen=pg.mkPen(event_color, width=1.5),
                    brush=pg.mkBrush(event_color),
                )
                scatter.setToolTip(
                    "{} / {}：{}\n{}".format(
                        event.get("event_uid", "--"),
                        cluster_id,
                        event_state,
                        event.get("fusion_reason") or "",
                    )
                )
                plot.addItem(scatter)
                dynamic_items.append(scatter)
                event_text = pg.TextItem(
                    str(event.get("event_uid") or "--"),
                    color=event_color,
                    anchor=(0.5, 1.25),
                )
                event_text.setPos(x_value, y_value)
                plot.addItem(event_text, ignoreBounds=True)
                dynamic_items.append(event_text)
            if points:
                cluster_text = pg.TextItem(
                    f"{cluster_id} {cluster_state}",
                    color=cluster_color,
                    anchor=(0.5, 0.0),
                )
                cluster_text.setPos(
                    float(np.median([point[0] for point in points])),
                    min(2.25, max(point[1] for point in points) + 0.22),
                )
                plot.addItem(cluster_text, ignoreBounds=True)
                dynamic_items.append(cluster_text)
        self.voc_timeline_dynamic_items = dynamic_items

    def evaluate_driver_drinking_decision(self, now_timestamp=None):
        now_timestamp = time.time() if now_timestamp is None else float(now_timestamp)
        recognition_result = self._recognition_result_for_voc_fusion()
        predicted_label = str(
            (recognition_result or {}).get("predicted_label", "")
        ).strip()
        empty_result = {
            "decision": "--",
            "first_key": None,
            "first_label": "--",
            "second_key": None,
            "second_label": "--",
            "matched_positions": 0,
            "first_response_gap_seconds": None,
            "position_ambiguous": False,
            "position_ambiguity_reason": "",
            "final": False,
        }
        if predicted_label != "酒精":
            self._invalidate_active_voc_fusion(
                "电子鼻识别已离开酒精状态，本次融合中止",
                now_timestamp,
            )
            return empty_result

        recognition_timestamp = self._recognition_timestamp_seconds(
            recognition_result
        )
        if recognition_timestamp is None:
            result = dict(empty_result)
            result["decision"] = "酒精-缺少识别时间"
            return result
        response_onset_timestamp = self._response_onset_timestamp_seconds(
            recognition_result
        )
        fusion_anchor_timestamp = (
            recognition_timestamp
            if response_onset_timestamp is None
            else response_onset_timestamp
        )
        fusion_anchor_source = (
            "recognition_timestamp"
            if response_onset_timestamp is None
            else "response_onset"
        )
        recognition_id = int(round(recognition_timestamp * 1000))
        if (
            self.voc_active_recognition_id is not None
            and self.voc_active_recognition_id != recognition_id
        ):
            self._invalidate_active_voc_fusion(
                "出现新的酒精识别，本次融合已被后续识别替代",
                now_timestamp,
            )
        self.voc_visual_recognition_id = recognition_id
        self.voc_visual_anchor_timestamp = float(fusion_anchor_timestamp)
        self.voc_visual_anchor_source = fusion_anchor_source
        cached = self.voc_fusion_results.get(recognition_id)
        if cached is not None:
            self.voc_active_recognition_id = None
            self.voc_visual_fusion_state = "已完成"
            return dict(cached)
        self.voc_active_recognition_id = recognition_id
        self.voc_visual_fusion_state = "收集中"

        candidates = self._voc_events_for_recognition(
            recognition_timestamp,
            recognition_id,
            now_timestamp,
            response_onset_timestamp,
        )
        cluster_details = self._voc_event_cluster_details(
            candidates,
            fusion_anchor_timestamp,
        )
        self._sync_voc_cluster_records(
            recognition_id,
            cluster_details,
            fusion_anchor_timestamp,
        )
        selected_detail = (
            min(cluster_details, key=lambda detail: detail["score"])
            if cluster_details
            else None
        )
        group = [] if selected_detail is None else selected_detail["group"]
        position_order = {
            key: index for index, (key, _) in enumerate(VOC_POSITIONS)
        }
        ordered_group = sorted(
            group,
            key=lambda item: (
                float(item["event"]["start_timestamp"]),
                float(item["event"]["confirm_timestamp"]),
                position_order.get(item["position_key"], len(position_order)),
            ),
        )
        matched_positions = len(ordered_group)
        first_item = ordered_group[0] if ordered_group else None
        second_item = ordered_group[1] if len(ordered_group) >= 2 else None
        first_key = first_item["position_key"] if first_item else None
        first_label = first_item["position_label"] if first_item else "--"
        second_key = second_item["position_key"] if second_item else None
        second_label = second_item["position_label"] if second_item else "--"
        first_response_gap_seconds = None
        if first_item is not None and second_item is not None:
            first_response_gap_seconds = max(
                0.0,
                float(second_item["event"]["start_timestamp"])
                - float(first_item["event"]["start_timestamp"]),
            )
        response_times_too_close = bool(
            first_response_gap_seconds is not None
            and first_response_gap_seconds <= VOC_POSITION_AMBIGUITY_SEC
        )
        first_seen_at = self.voc_fusion_first_seen_at.setdefault(
            recognition_id,
            now_timestamp,
        )
        deadline_reached = (
            now_timestamp
            >= first_seen_at + VOC_FUSION_LOOKAHEAD_SEC
        )
        all_positions_matched = matched_positions == len(VOC_POSITIONS)
        # 起点分簇模式等待完整收集期，避免较早的三路干扰簇抢先锁定结果。
        final = bool(
            deadline_reached
            or (
                response_onset_timestamp is None
                and all_positions_matched
            )
        )
        self._apply_voc_cluster_lifecycle(
            recognition_id,
            cluster_details,
            selected_detail,
            fusion_anchor_timestamp,
            final,
            now_timestamp,
        )
        self.voc_timeline_signature = None

        position_ambiguous = False
        position_ambiguity_reason = ""
        if final and matched_positions == 0:
            position_ambiguous = True
            position_ambiguity_reason = "规定时间内未匹配到定位传感器事件"
        elif final and response_times_too_close:
            position_ambiguous = True
            position_ambiguity_reason = (
                f"最早两路响应仅相差 {first_response_gap_seconds:.2f} 秒，"
                f"不超过 {VOC_POSITION_AMBIGUITY_SEC:.2f} 秒阈值"
            )

        if not final:
            decision = (
                "酒精-等待VOC触发"
                if matched_positions == 0
                else "酒精-正在确认位置"
            )
        elif position_ambiguous:
            decision = "酒精-无法确认方位"
        elif first_key == "driver":
            decision = "酒精-主驾"
        else:
            decision = "酒精-非主驾"

        result = {
            "decision": decision,
            "first_key": first_key,
            "first_label": first_label,
            "second_key": second_key,
            "second_label": second_label,
            "matched_positions": matched_positions,
            "first_response_gap_seconds": first_response_gap_seconds,
            "position_ambiguity_threshold_seconds": float(
                VOC_POSITION_AMBIGUITY_SEC
            ),
            "position_ambiguous": position_ambiguous,
            "position_ambiguity_reason": position_ambiguity_reason,
            "final": final,
            "recognition_id": recognition_id,
            "fusion_anchor_timestamp_ms": int(
                round(fusion_anchor_timestamp * 1000)
            ),
            "fusion_anchor_source": fusion_anchor_source,
            "candidate_event_count": len(candidates),
            "cluster_count": len(cluster_details),
            "selected_cluster_id": (
                selected_detail["cluster_id"]
                if selected_detail is not None
                else None
            ),
        }
        if final:
            self.voc_active_recognition_id = None
            self.voc_visual_fusion_state = "已完成"
            self.voc_fusion_results[recognition_id] = dict(result)
            if len(self.voc_fusion_results) > 20:
                oldest_id = sorted(self.voc_fusion_results)[0]
                self.voc_fusion_results.pop(oldest_id, None)
                self.voc_fusion_first_seen_at.pop(oldest_id, None)
                if oldest_id != self.voc_visual_recognition_id:
                    self.voc_cluster_records.pop(oldest_id, None)
        return result

    def first_voc_trigger_position(self):
        fusion = self.evaluate_driver_drinking_decision()
        return fusion.get("first_key"), fusion.get("first_label", "")

    def driver_drinking_decision_text(self, predicted_label):
        label = str(predicted_label or "").strip()
        if label != "酒精":
            return "--"
        return self.evaluate_driver_drinking_decision().get("decision", "--")

    def update_driver_drinking_display(self):
        live_result = getattr(self, "last_inference_result", None)
        disp = self.recognition_result_for_display(live_result) if live_result else None
        predicted_label = str((disp or {}).get("predicted_label", "")).strip()
        fusion = self.evaluate_driver_drinking_decision()
        self._refresh_voc_event_indicators()
        decision = (
            fusion.get("decision", "--")
            if predicted_label == "酒精"
            else "--"
        )
        self.driver_drinking_decision = decision
        first_text = fusion.get("first_label", "--")
        if fusion.get("position_ambiguous"):
            gap_seconds = fusion.get("first_response_gap_seconds")
            second_text = fusion.get("second_label", "--")
            if gap_seconds is None:
                trigger_summary = "定位事件：无"
            else:
                trigger_summary = (
                    f"前两路：{first_text}/{second_text}，"
                    f"相差 {float(gap_seconds):.2f}s"
                )
        else:
            trigger_summary = f"最先触发：{first_text}"

        if hasattr(self, "main_driver_drinking_label"):
            self.main_driver_drinking_label.setText(f"饮酒位置判断：{decision}")
            if decision == "酒精-主驾":
                color = "rgb(185, 45, 35)"
            elif decision == "酒精-非主驾":
                color = "rgb(40, 95, 150)"
            elif decision.startswith("酒精-"):
                color = "rgb(130, 90, 30)"
            else:
                color = "rgb(90, 90, 90)"
            self.main_driver_drinking_label.setStyleSheet(
                f'font: bold 14pt "KaiTi"; color: {color};'
            )

        if hasattr(self, "voc_decision_label"):
            cluster_id = fusion.get("selected_cluster_id") or "--"
            cluster_title = "采用簇" if fusion.get("final") else "当前簇"
            self.voc_decision_label.setText(
                f"融合判断：{decision} | {cluster_title}：{cluster_id} | "
                f"{trigger_summary}"
            )

    def _load_default_realtime_model(self):
        """启动时加载并开启指定的默认实时推理模型。"""
        model_path = os.path.join(MODELS_DIR, DEFAULT_REALTIME_MODEL_RELATIVE_PATH)
        if not os.path.isfile(model_path):
            self.update_main_inference_display()
            if hasattr(self, "statusbar"):
                self.statusbar.showMessage(f"默认模型不存在: {model_path}", 8000)
            return
        try:
            self.load_model_bundle_from_path(model_path)
            self.set_inference_enabled(True)
        except Exception as exc:
            self.loaded_model_bundle = None
            self.loaded_model_path = ""
            self.inference_enabled = False
            self.update_main_inference_display()
            if hasattr(self, "statusbar"):
                self.statusbar.showMessage(f"默认模型加载失败: {exc}", 8000)
