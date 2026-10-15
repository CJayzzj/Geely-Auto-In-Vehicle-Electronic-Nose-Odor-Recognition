# -*- coding: utf-8 -*-
"""Functional MainWindow mixin mechanically moved from V4.6.

Keeping these methods in focused modules controls file size while preserving the
original widget layout and signal behavior.
"""

from runtime_support import *
from algorithm.classifier import predict_feature_vector
from gui.plot_widgets import _apply_standard_plot_axes, sensor_raw_curve_rgb


class MainWindowSensorMixin:
    def init_ui_logic(self):
        # 绑定按钮事件，并设置默认波特率。
        self.comfirm_link.clicked.connect(self.on_confirm_link_clicked)
        self.reflesh.clicked.connect(self.on_refresh_clicked)
        self.on_refresh_clicked()
        self.rate_input.setText("115200")
        self.clear.clicked.connect(self.clear_plot)
        self._init_baseline_params_ui()

    def _init_baseline_params_ui(self):
        """初始化 baseline 参数；默认值立即生效，确认按钮用于应用手动修改。"""
        self.baseline_sig_x_c = float(BASELINE_SIGMOID_X_C)
        self.baseline_sig_hi = float(BASELINE_SIGMOID_ALPHA_HI_DEFAULT)
        self.baseline_sig_lo = float(BASELINE_SIGMOID_ALPHA_LO_DEFAULT)

        self.baseline_xc_label = QtWidgets.QLabel("x_c", self.centralwidget)
        self.baseline_xc_label.setStyleSheet("font: 12pt \"楷体\";")
        self.baseline_xc_input = QtWidgets.QLineEdit(self.centralwidget)
        self.baseline_xc_input.setText(str(BASELINE_SIGMOID_X_C))
        self.baseline_xc_input.setMaximumWidth(64)
        self.baseline_xc_input.setStyleSheet("font: 12pt \"楷体\";")
        self.baseline_xc_input.setToolTip("logistic 中心点 x_c：|resp| 接近该值时 α 开始快速接近 α_lo")

        self.baseline_af_label = QtWidgets.QLabel("α_hi", self.centralwidget)
        self.baseline_af_label.setStyleSheet("font: 12pt \"楷体\";")
        self.baseline_af_input = QtWidgets.QLineEdit(self.centralwidget)
        self.baseline_af_input.setText(str(BASELINE_SIGMOID_ALPHA_HI_DEFAULT))
        self.baseline_af_input.setMaximumWidth(72)
        self.baseline_af_input.setStyleSheet("font: 12pt \"楷体\";")
        self.baseline_af_input.setToolTip("小残差时 EMA 系数上限 α_hi")

        self.baseline_as_label = QtWidgets.QLabel("α_lo", self.centralwidget)
        self.baseline_as_label.setStyleSheet("font: 12pt \"楷体\";")
        self.baseline_as_input = QtWidgets.QLineEdit(self.centralwidget)
        self.baseline_as_input.setText(str(BASELINE_SIGMOID_ALPHA_LO_DEFAULT))
        self.baseline_as_input.setMaximumWidth(80)
        self.baseline_as_input.setStyleSheet("font: 12pt \"楷体\";")
        self.baseline_as_input.setToolTip("大残差 / x≥截断 时的 EMA 系数下限 α_lo")

        self.baseline_params_confirm_btn = QtWidgets.QPushButton("确认", self.centralwidget)
        self.baseline_params_confirm_btn.setStyleSheet("font: 12pt \"楷体\";")
        self.baseline_params_confirm_btn.setToolTip("应用修改后的 x_c、α_hi、α_lo 参数")
        self.baseline_params_confirm_btn.clicked.connect(self.on_baseline_params_confirmed)
        try:
            insert_at = 2
            self.horizontalLayout_3.insertWidget(insert_at, self.baseline_params_confirm_btn)
            self.horizontalLayout_3.insertWidget(insert_at, self.baseline_as_input)
            self.horizontalLayout_3.insertWidget(insert_at, self.baseline_as_label)
            self.horizontalLayout_3.insertWidget(insert_at, self.baseline_af_input)
            self.horizontalLayout_3.insertWidget(insert_at, self.baseline_af_label)
            self.horizontalLayout_3.insertWidget(insert_at, self.baseline_xc_input)
            self.horizontalLayout_3.insertWidget(insert_at, self.baseline_xc_label)
        except Exception:
            for w in (
                self.baseline_xc_label,
                self.baseline_xc_input,
                self.baseline_af_label,
                self.baseline_af_input,
                self.baseline_as_label,
                self.baseline_as_input,
                self.baseline_params_confirm_btn,
            ):
                w.setParent(self.centralwidget)

        self._update_comfirm_link_enabled()

    def on_baseline_params_confirmed(self):
        def _parse_float(s):
            return float(str(s).strip().replace(",", "."))

        try:
            x_c = _parse_float(self.baseline_xc_input.text())
            if not (0.0 < x_c < BASELINE_SIGMOID_X_CAP):
                raise ValueError("x_c")

            hi = _parse_float(self.baseline_af_input.text())
            lo = _parse_float(self.baseline_as_input.text())
            if not (0.0 < lo < 1.0) or not (lo < hi <= 1.0):
                raise ValueError("alpha_range")
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self,
                "提示",
                "请填写有效参数：\n"
                f"· x_c：0～{BASELINE_SIGMOID_X_CAP:g} 之间的小数，建议 {BASELINE_SIGMOID_X_C:g}\n"
                "· α_hi、α_lo：0～1 之间的小数，且 α_lo < α_hi",
            )
            return

        self.baseline_sig_x_c = x_c
        self.baseline_sig_hi = hi
        self.baseline_sig_lo = lo
        self._update_comfirm_link_enabled()
        if hasattr(self, "statusbar"):
            self.statusbar.showMessage(
                f"Baseline(sigmoid) 参数已应用：x_c={x_c:g}，α_hi={hi:g}，α_lo={lo:g}；"
                f"k={BASELINE_SIGMOID_K_DEFAULT:g}，x≥{BASELINE_SIGMOID_X_CAP} 时 α=α_lo。"
                f"{' 已可连接串口。' if self.get_device_id_text() else ' 请填写设备号后再连接串口。'}",
                8000,
            )

    def init_plot(self):
        # 初始化 18 个图窗，其中前 16 路额外叠加 baseline 曲线。
        pg.setConfigOptions(antialias=True)
        plot_widgets = [
            self.Sensor1_view, self.Sensor2_view, self.Sensor3_view, self.Sensor4_view,
            self.Sensor5_view, self.Sensor6_view, self.Sensor7_view, self.Sensor8_view,
            self.Sensor9_view, self.Sensor10_view, self.Sensor11_view, self.Sensor12_view,
            self.Sensor13_view, self.Sensor14_view, self.Sensor15_view, self.Sensor16_view,
            self.Sensor17_view, self.Sensor18_view
        ]
        self.plot_widgets = plot_widgets
        for i, pw in enumerate(plot_widgets):
            pw.setBackground('w')
            pw.setClipToView(True)
            pw.setDownsampling(auto=True, mode='peak')
            _apply_standard_plot_axes(pw)
            # 最后两个为温度/湿度，label 改成对应名称
            if i < 16:
                pw.setLabel('left', f'Sensor {i + 1}', color='black')
            elif i == 16:
                pw.setLabel('left', 'Temperature', color='black')
            else:
                pw.setLabel('left', 'Humidity', color='black')

            pw.setLabel('bottom', 'Time', color='black')
            if i < 16:
                baseline_curve = pw.plot(
                    pen=pg.mkPen(
                        color=SENSOR_CURVE_BL_RGB, width=1.5, style=QtCore.Qt.DashLine
                    ),
                )
                self.baseline_curves.append(baseline_curve)
                curve = pw.plot(pen=pg.mkPen(color=sensor_raw_curve_rgb(i), width=2))
                self.curves.append(curve)
                comp_curve = pw.plot(
                    pen=pg.mkPen(
                        color=SENSOR_CURVE_COMP_RGB, width=1.5, style=QtCore.Qt.DotLine
                    ),
                )
                self.comp_curves.append(comp_curve)
            else:
                curve = pw.plot(pen=pg.mkPen(color=(i * 15 % 255, 100, 200), width=2))
                self.curves.append(curve)

            if i < 16:
                tighten_plot_layout(pw, left_axis_width=50, bottom_axis_height=28)
            else:
                tighten_plot_layout(pw, left_axis_width=50, bottom_axis_height=28)

    def _configure_visible_e_nose_plots(self):
        """只在主界面显示 1/5/9/13 号电子鼻通道和温湿度，保留 16 路数据处理。"""
        plot_widgets = list(getattr(self, "plot_widgets", []))
        value_labels = list(getattr(self, "value_labels", []))
        if len(plot_widgets) < 18 or len(value_labels) < 18:
            return

        visible_indices = set(E_NOSE_VISIBLE_CHANNEL_INDICES)
        for idx, (plot_widget, value_label) in enumerate(zip(plot_widgets, value_labels)):
            try:
                self.gridLayout.removeWidget(plot_widget)
            except Exception:
                pass
            try:
                self.gridLayout.removeWidget(value_label)
            except Exception:
                pass
            if idx in visible_indices:
                plot_widget.show()
                value_label.show()
            else:
                plot_widget.hide()
                value_label.hide()

        self.temperature_humidity_plot = plot_widgets[16]
        self.temperature_humidity_value_label = value_labels[16]
        self.temperature_humidity_plot.setLabel("left", "Temp / Humidity", color="black")
        self.temperature_humidity_plot.setMinimumHeight(MAIN_SIGNAL_PLOT_HEIGHT)
        self.temperature_humidity_plot.setMaximumHeight(16777215)
        if len(self.curves) > 17:
            self.temperature_humidity_curve = self.temperature_humidity_plot.plot(
                pen=pg.mkPen(color=(40, 150, 70), width=1.8)
            )
            self.curves[17] = self.temperature_humidity_curve

        self.e_nose_display_group = QtWidgets.QGroupBox(
            "电子鼻信号（Sensor 1 / 5 / 9 / 13）",
            self.view,
        )
        self.e_nose_display_group.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.e_nose_display_group.setStyleSheet(
            """
            QGroupBox {
                font: bold 12pt "楷体";
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
        display_layout = QtWidgets.QGridLayout(self.e_nose_display_group)
        display_layout.setContentsMargins(8, 14, 8, 6)
        display_layout.setHorizontalSpacing(8)
        display_layout.setVerticalSpacing(3)

        self.e_nose_title_labels = []
        for order, idx in enumerate(E_NOSE_VISIBLE_CHANNEL_INDICES):
            row_base = (order // 2) * 2
            col = order % 2

            value_label = value_labels[idx]
            value_label.setAlignment(QtCore.Qt.AlignCenter)
            value_label.setMinimumHeight(34)
            value_label.setMaximumHeight(44)
            value_label.setText(
                f'<span style="font-weight: 700;">Sensor {idx + 1}</span><br/>--'
            )
            plot_widget = plot_widgets[idx]
            plot_widget.setMinimumHeight(MAIN_SIGNAL_PLOT_HEIGHT)
            plot_widget.setMaximumHeight(16777215)
            plot_widget.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
            )

            display_layout.addWidget(value_label, row_base, col)
            display_layout.addWidget(plot_widget, row_base + 1, col)
            display_layout.setColumnStretch(col, 1)
            display_layout.setRowStretch(row_base + 1, 3)

        if hasattr(self, "main_inference_result_group"):
            try:
                self.gridLayout.removeWidget(self.main_inference_result_group)
            except Exception:
                pass

    @staticmethod
    def _append_limited(series, value):
        """追加一个绘图点，并按 MAX_POINTS 原地裁剪，保证各曲线长度稳定。"""
        series.append(value)
        if MAX_POINTS is not None and len(series) > MAX_POINTS:
            del series[:-MAX_POINTS]

    @staticmethod
    def _format_number(value, digits=2, invalid="--", inf_text="--"):
        """把实时显示中的数值格式化为固定小数；无效值给出界面占位。"""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return invalid
        if value == float("inf"):
            return inf_text
        if not np.isfinite(value):
            return invalid
        return f"{value:.{digits}f}"

    def _baseline_alpha_params(self):
        """返回当前 baseline sigmoid 参数；未确认时使用默认值兜底。"""
        return {
            "k": float(BASELINE_SIGMOID_K_DEFAULT),
            "x_c": (
                self.baseline_sig_x_c
                if self.baseline_sig_x_c is not None
                else float(BASELINE_SIGMOID_X_C)
            ),
            "hi": (
                self.baseline_sig_hi
                if self.baseline_sig_hi is not None
                else float(BASELINE_SIGMOID_ALPHA_HI_DEFAULT)
            ),
            "lo": (
                self.baseline_sig_lo
                if self.baseline_sig_lo is not None
                else float(BASELINE_SIGMOID_ALPHA_LO_DEFAULT)
            ),
        }

    def _initial_baseline_alpha_active(self, timestamp):
        """连接后的初始稳定期使用较大的 alpha，让 baseline 快速贴近环境。"""
        return (
            self.system_start_time is not None
            and np.isfinite(timestamp)
            and (timestamp - self.system_start_time) < BASELINE_INITIAL_ALPHA_DURATION_S
        )

    def _baseline_sigmoid_x(self, sensor_idx, value, baseline):
        """Return abs(resp_i / x_cap_i) for baseline alpha mapping only."""
        if not (
            np.isfinite(value)
            and np.isfinite(baseline)
            and baseline != 0.0
            and baseline != float("inf")
        ):
            return float("nan")
        resp = (float(value) - float(baseline)) / abs(float(baseline))
        x = abs(resp)
        try:
            params = load_baseline_sigmoid_resp_params(
                self.get_device_id_text(),
                BASELINE_SIGMOID_X_CAP_CAL_CONC,
            )
            x_caps = params.get("x_cap") or CONC2_RESP_GATE_X_CAP
            x_cap = float(x_caps[sensor_idx])
        except Exception:
            x_cap = float("nan")
        if np.isfinite(x_cap) and x_cap > 0.0:
            return x / x_cap
        return x

    def _format_sensor_value_label(
        self, sensor_idx, raw_value, compensated_value, baseline_value, alpha_used
    ):
        """用两行显示通道名、alpha、baseline、原始电阻和补偿电阻。"""
        alpha_text = self._format_number(alpha_used, digits=6)
        baseline_text = self._format_number(baseline_value, digits=2)
        raw_text = self._format_number(raw_value, digits=2, inf_text="inf")
        comp_text = self._format_number(compensated_value, digits=2)

        br, bg, bb = SENSOR_CURVE_BL_RGB
        rr, rg, rb = sensor_raw_curve_rgb(sensor_idx)
        cr, cg, cb = SENSOR_CURVE_COMP_RGB
        ar, ag, ab = 80, 60, 120
        return (
            f'<span style="font-weight: 700; color: rgb(45,45,45);">Sensor {sensor_idx + 1}</span>'
            f'&nbsp;&nbsp;<span style="color: rgb({ar},{ag},{ab}); font-weight: 600;">α: {alpha_text}</span>'
            f'&nbsp;&nbsp;<span style="color: rgb({br},{bg},{bb})">BL: {baseline_text}</span><br/>'
            f'<span style="color: rgb({rr},{rg},{rb})">R_raw: {raw_text}</span>'
            f'&nbsp;&nbsp;'
            f'<span style="color: rgb({cr},{cg},{cb})">R_comp: {comp_text}</span>'
        )

    def update_sensor_baseline(self, sensor_idx, value, timestamp):
        """按 |resp| 经 logistic 得到 α，对补偿电阻域 baseline 做指数平滑。

        程序启动后前一分钟 α 固定为 0.5；之后切换为 sigmoid 规则持续更新。
        """
        # 遇到 nan/inf 等异常值时不更新基线，直接沿用上一时刻的 baseline，
        # 避免无效值把基线拉偏。
        if not np.isfinite(value):
            return self.baseline_values[sensor_idx], None

        # 第一次收到该传感器数据时，直接把当前值作为初始 baseline，
        # 同时记录该传感器 baseline 开始自适应更新的起始时间。
        if self.baseline_values[sensor_idx] is None:
            self.baseline_values[sensor_idx] = value
            self.baseline_start_times[sensor_idx] = timestamp
            return value, float(BASELINE_INITIAL_ALPHA)

        params = self._baseline_alpha_params()

        baseline = self.baseline_values[sensor_idx]
        if (
            not np.isfinite(baseline)
            or baseline == 0.0
            or baseline == float("inf")
        ):
            self.baseline_values[sensor_idx] = value
            self.baseline_start_times[sensor_idx] = timestamp
            return value, None

        if self._initial_baseline_alpha_active(timestamp):
            alpha = float(BASELINE_INITIAL_ALPHA)
        elif value > baseline:
            alpha = float(BASELINE_RISE_FAST_ALPHA)
        else:
            x = self._baseline_sigmoid_x(sensor_idx, value, baseline)
            alpha = compute_baseline_alpha_sigmoid(
                x,
                params["hi"],
                params["lo"],
                params["k"],
                params["x_c"],
                BASELINE_SIGMOID_X_CAP,
            )

        # 新 baseline = 旧 baseline + alpha * (当前值 - 旧 baseline)
        baseline += alpha * (value - baseline)
        self.baseline_values[sensor_idx] = baseline
        return baseline, float(alpha)

    def get_current_alpha(self, sensor_idx, timestamp, current_value=None):
        """返回当前传感器 baseline 更新所使用的 alpha（与 update_sensor_baseline 规则一致）。"""
        if self._initial_baseline_alpha_active(timestamp):
            return float(BASELINE_INITIAL_ALPHA)

        params = self._baseline_alpha_params()
        if self.baseline_values[sensor_idx] is None:
            return float(params["hi"])
        baseline = self.baseline_values[sensor_idx]
        v = current_value
        if v is None and sensor_idx < len(self.compensated_ydata) and self.compensated_ydata[sensor_idx]:
            v = self.compensated_ydata[sensor_idx][-1]
        if v is None or not np.isfinite(v) or v == float("inf"):
            return float(params["lo"])
        if (
            not np.isfinite(baseline)
            or baseline == 0.0
            or baseline == float("inf")
        ):
            return float(params["lo"])
        if v > baseline:
            return float(BASELINE_RISE_FAST_ALPHA)
        x = self._baseline_sigmoid_x(sensor_idx, v, baseline)
        return compute_baseline_alpha_sigmoid(
            x,
            params["hi"],
            params["lo"],
            params["k"],
            params["x_c"],
            BASELINE_SIGMOID_X_CAP,
        )

    def update_system_info_display(self, timestamp):
        """刷新顶部系统运行时间。"""
        if not self.system_info_label:
            return

        if self.system_start_time is None:
            runtime_text = "00:00:00"
        else:
            runtime_text = self.format_elapsed_time(timestamp - self.system_start_time)

        self.system_info_label.setText(f"运行时间: {runtime_text}")

    @staticmethod
    def build_dataset_row(record):
        """按统一列顺序把样本字典转成一行数据。"""
        return [record.get(column, "") for column in STANDARD_COLUMNS]

    def _score_threshold_entries(self):
        return response_strength_threshold_entries(
            self.get_device_id_text(),
            self.task_config["class_labels"],
        )

    def _apply_score_threshold_entries_to_rows(self, rows, start_index=0):
        if RESPONSE_STRENGTH_THRESHOLDS_COLUMN not in STANDARD_COLUMNS:
            return rows
        column_index = list(STANDARD_COLUMNS).index(RESPONSE_STRENGTH_THRESHOLDS_COLUMN)
        entries = self._score_threshold_entries()
        out_rows = []
        for offset, row in enumerate(rows or []):
            row_values = list(row)
            if len(row_values) < len(STANDARD_COLUMNS):
                row_values.extend([""] * (len(STANDARD_COLUMNS) - len(row_values)))
            entry_index = int(start_index) + int(offset)
            row_values[column_index] = (
                entries[entry_index] if entry_index < len(entries) else ""
            )
            out_rows.append(row_values)
        return out_rows

    def _dataset_with_score_threshold_entries(self, df):
        out = dataframe_from_records([]) if df is None else df.copy()
        if RESPONSE_STRENGTH_THRESHOLDS_COLUMN not in out.columns:
            out[RESPONSE_STRENGTH_THRESHOLDS_COLUMN] = ""
        out[RESPONSE_STRENGTH_THRESHOLDS_COLUMN] = ""
        entries = self._score_threshold_entries()
        for idx, entry in enumerate(entries):
            if idx >= len(out):
                break
            out.at[out.index[idx], RESPONSE_STRENGTH_THRESHOLDS_COLUMN] = entry
        return out

    def build_dataset_record(
        self,
        time_str,
        adc_values,
        resistance_raw_values,
        resistance_comp_values,
        baseline_values,
        resp_values,
        pattern_values,
        temp_value,
        humid_value,
        label="",
    ):
        """构造一条包含 ADC、R_raw、R_comp、baseline、resp、centered_resp、pattern 的数据记录。"""
        record = {
            "Time": time_str,
            "Temperature": temp_value,
            "Humidity": humid_value,
            "Timestamp": time_str,
            "Label": label,
        }
        for idx, value in enumerate(adc_values):
            record[ADC_COLUMNS[idx]] = value
        for idx, value in enumerate(resistance_raw_values):
            record[R_RAW_COLUMNS[idx]] = value
        for idx, value in enumerate(resistance_comp_values):
            record[R_COMP_COLUMNS[idx]] = value
        for idx, value in enumerate(baseline_values):
            record[BASELINE_COLUMNS[idx]] = value
        return self.apply_feature_values_to_record(record, resp_values, pattern_values)

    def apply_feature_values_to_record(self, record, resp_values, pattern_values):
        """Overwrite feature columns with an event-level peak summary vector."""
        out = dict(record or {})
        resp_values = [] if resp_values is None else list(resp_values)
        pattern_values = [] if pattern_values is None else list(pattern_values)
        if len(resp_values) < len(RESP_COLUMNS):
            resp_values.extend([0.0] * (len(RESP_COLUMNS) - len(resp_values)))
        if len(pattern_values) < len(PATTERN_COLUMNS):
            pattern_values.extend([0.0] * (len(PATTERN_COLUMNS) - len(pattern_values)))

        resp_values = resp_values[: len(RESP_COLUMNS)]
        pattern_values = pattern_values[: len(PATTERN_COLUMNS)]

        for idx, value in enumerate(resp_values):
            out[RESP_COLUMNS[idx]] = value
        centered_resp_values = compute_material_centered_resp_values(resp_values)
        if centered_resp_values is None:
            centered_resp_values = [0.0 for _ in CENTERED_RESP_COLUMNS]
        for idx, value in enumerate(centered_resp_values):
            out[CENTERED_RESP_COLUMNS[idx]] = value
        for idx, value in enumerate(pattern_values):
            out[PATTERN_COLUMNS[idx]] = value
        group_zscore_resp_values = compute_material_group_zscore_resp_values(resp_values)
        if group_zscore_resp_values is None:
            group_zscore_resp_values = [0.0 for _ in GROUP_ZSCORE_RESP_COLUMNS]
        for idx, value in enumerate(group_zscore_resp_values):
            out[GROUP_ZSCORE_RESP_COLUMNS[idx]] = value

        device_id_text = self.get_device_id_text()
        out[RESPONSE_STRENGTH_SCORE_COLUMN] = response_strength_score_from_resp_values(
            resp_values,
            device_id_text,
        )
        out[RESPONSE_STRENGTH_THRESHOLDS_COLUMN] = ""
        return out

    def build_feature_vectors(self, resistance_16_compensated, baseline_values=None):
        """根据补偿后电阻 R 与补偿域 baseline（同域平滑电阻）计算 resp / pattern。"""
        if baseline_values is None:
            baseline_values = self.baseline_values
        resp_values = []
        for i, value_r in enumerate(resistance_16_compensated):
            baseline_r = baseline_values[i] if i < len(baseline_values) else None
            if (
                baseline_r is None
                or not np.isfinite(baseline_r)
                or baseline_r == 0
                or baseline_r == float("inf")
                or not np.isfinite(value_r)
                or value_r == float("inf")
            ):
                resp_values.append(0.0)
            else:
                resp_values.append((value_r - baseline_r) / abs(baseline_r))

        max_abs_resp = max((abs(v) for v in resp_values), default=0.0)
        if max_abs_resp == 0:
            pattern_values = [0.0 for _ in resp_values]
        else:
            pattern_values = [v / max_abs_resp for v in resp_values]

        return resp_values, pattern_values

    def update_feature_window_data(self):
        """若特征窗口正在显示，则同步刷新其中的曲线数据。"""
        if self.feature_window is not None and self.feature_window.isVisible():
            self.feature_window.set_feature_data(
                self.xdata,
                self.resp_ydata,
                self.group_zscore_resp_ydata,
            )

    def reset_peak_detector(self):
        """重置局部峰值触发识别状态。"""
        self.peak_detector.reset()

    def build_peak_wait_result(
        self,
        timestamp,
        signal,
        status_text,
        predicted_label=None,
        gate_fields=None,
    ):
        """未触发峰值识别时，给界面一个可解释的等待状态。"""
        label = str(predicted_label or "").strip()
        recognition_pending = not label or label == "识别中"
        base_result = {}
        if recognition_pending and isinstance(self.last_committed_recognition_result, dict):
            base_result = copy.deepcopy(self.last_committed_recognition_result)
            label = str(base_result.get("predicted_label", "")).strip()
        if recognition_pending and (not label or label == "识别中"):
            label = INFERENCE_RESP_GATE_LABEL

        peak_fields = self.peak_detector.snapshot_fields()
        result = dict(base_result)
        result.update(
            {
                "predicted_index": base_result.get("predicted_index"),
                "predicted_label": label,
                "probabilities": base_result.get("probabilities", {}),
                "odor_level": base_result.get("odor_level", "--"),
                "timestamp": self.format_record_time(timestamp),
                "timestamp_ms": int(round(float(timestamp) * 1000)),
                "recognition_pending": bool(recognition_pending),
                "peak_detector_enabled": True,
                "peak_triggered": False,
                "peak_status": status_text,
                "peak_current_signal": float(signal),
                "resp_gate_score": float("nan"),
                "resp_gate_score_threshold": float(INFERENCE_RESP_GATE_SCORE_THRESHOLD),
                "resp_gate_top_k": int(INFERENCE_RESP_GATE_TOP_K),
            }
        )
        result.update(peak_fields)
        if gate_fields:
            result.update(gate_fields)
        return result

    def _merge_odor_preview_into_result(
        self, result_dict, feature_vector, device_id_text=""
    ):
        """闭集二分类等待帧不运行分类器，仅保留寻峰状态。"""
        return result_dict

    def build_model_inference_result(self, feature_vector, resp_values, timestamp):
        """对指定帧特征做一次模型识别，并补充酒精等级信息。"""
        result = predict_feature_vector(self.loaded_model_bundle, feature_vector)
        result["timestamp"] = self.format_record_time(timestamp)
        result["timestamp_ms"] = int(round(float(timestamp) * 1000))
        predicted_label = str(result.get("predicted_label", "")).strip()
        if predicted_label == "酒精":
            result["odor_level"] = self.calculate_alcohol_odor_level(resp_values)
        elif predicted_label == self.task_config["class_labels"][1]:
            strength_score, level = self.calculate_ethanol_equiv_odor_level(
                predicted_label,
                resp_values,
            )
            result["odor_strength_score"] = strength_score
            result["odor_strength_metric"] = (
                f"top3_resp_over_{RESP_X_CAP_CAL_CONC:g}ppm_gate"
            )
            result["odor_strength_reference"] = "ethanol_equivalent_10_50_100ppm"
            result["odor_level"] = level
        else:
            result["odor_level"] = "--"
        return result

    def _apply_peak_detector_event(self, event, live_feature_vector, dev_raw, timestamp):
        """把 PeakDetector 的纯状态机事件转换成 UI/模型结果。"""
        if event.get("kind") == "wait":
            self.last_inference_result = self._merge_odor_preview_into_result(
                self.build_peak_wait_result(
                    event["timestamp"],
                    event["signal"],
                    event["status_text"],
                    event.get("predicted_label"),
                    event.get("gate_fields"),
                ),
                live_feature_vector,
                dev_raw,
            )
            self.update_model_center_inference_display()
            self.update_system_info_display(timestamp)
            return

        if event.get("kind") != "trigger":
            return

        try:
            response_onset_timestamp = float(
                event.get("response_onset_timestamp")
            )
        except (TypeError, ValueError):
            response_onset_timestamp = float("nan")
        if not np.isfinite(response_onset_timestamp):
            response_onset_timestamp = float(event["candidate_timestamp"])

        result = self.build_model_inference_result(
            event["candidate_feature_vector"],
            event["candidate_resp_values"],
            event["candidate_timestamp"],
        )
        result.update(
            {
                "peak_detector_enabled": True,
                "peak_triggered": True,
                "response_onset_timestamp_ms": int(
                    round(response_onset_timestamp * 1000)
                ),
                "peak_status": "16路峰值汇总触发识别",
                "peak_current_signal": float(event["peak_current_signal"]),
                "peak_signal": float(event["peak_signal"]),
                "peak_prominence": float(event["peak_prominence"]),
                "peak_min_prominence": float(
                    event.get(
                        "peak_min_prominence",
                        self.peak_detector.current_min_prominence(),
                    )
                ),
                "peak_sample_mode": event.get("peak_sample_mode"),
                "peak_event_trigger_reason": event.get("peak_event_trigger_reason"),
                **event.get("gate_fields", {}),
                "peak_track_sensor_idx": event.get("peak_track_sensor_idx"),
                "peak_track_sensor": event.get("peak_track_sensor", "--"),
                "peak_decline_frames": int(event.get("peak_decline_frames", 0)),
                "peak_required_decline_frames": int(
                    self.peak_detector.decline_frames_required
                ),
            }
        )
        self.last_inference_result = result
        history_text = f"{result['timestamp']} | 峰值 | {result['predicted_label']}"
        self.inference_history.append(history_text)
        self.inference_history = self.inference_history[-50:]
        self._record_model_peak_sample_if_active(event)
        self._record_pre_peak_window_sample_if_active(event)
        self.update_model_center_inference_display()
        self.update_system_info_display(timestamp)
        if hasattr(self, "statusbar"):
            self.statusbar.showMessage(f"实时识别: {result['predicted_label']}", 1500)

    def run_realtime_inference(self, resp_values, pattern_values, timestamp):
        """top3 门控连续确认后锁定主导传感器；锁定通道平滑 |resp| 过峰回落后识别一次。

        寻峰状态机由 PeakDetector 管理；主窗口只负责把 wait/trigger 事件转成界面和模型结果。
        """
        if not self.inference_enabled or self.loaded_model_bundle is None:
            return

        try:
            dev_raw = self.get_device_id_text()
            feature_vector = build_feature_vector(resp_values, pattern_values)
            gate_fields = build_top3_resp_gate_status_fields(resp_values, dev_raw)
            event = self.peak_detector.update(
                resp_values,
                pattern_values,
                feature_vector,
                timestamp,
                dev_raw,
                gate_fields,
                self.latest_dataset_record,
            )
            self._apply_peak_detector_event(event, feature_vector, dev_raw, timestamp)
        except Exception as exc:
            self.inference_enabled = False
            self.update_model_center_inference_display()
            self.update_system_info_display(time.time())
            self.show_error_message("错误", f"实时推理失败：{exc}")
