# -*- coding: utf-8 -*-
"""Functional MainWindow mixin mechanically moved from V4.6.

Keeping these methods in focused modules controls file size while preserving the
original widget layout and signal behavior.
"""

from runtime_support import *
from acquisition.peak_detector import PeakDetector
from acquisition.serial_threads import SerialThread


class MainWindowAcquisitionMixin:
    def _task_label(self, label_text):
        label = str(label_text or "").strip()
        return (self.task_config.get("label_merge_rule") or {}).get(label, label)

    def _sync_peak_record_ui(self):
        if (
            self.data_collection_dialog is not None
            and hasattr(self.data_collection_dialog, "sync_peak_record_state")
        ):
            self.data_collection_dialog.sync_peak_record_state()

    def _set_next_peak_record_status(self, status_text):
        status_text = str(status_text or "未启动")
        if status_text == self.next_peak_record_status:
            return
        self.next_peak_record_status = status_text
        self._sync_peak_record_ui()

    def is_next_peak_record_armed(self):
        return bool(self.next_peak_record_armed)

    def arm_next_peak_record(self, label_text):
        label = self._task_label(label_text)
        if not label:
            return False, "请先输入峰值样本标签。"

        self.next_peak_record_label = label
        self.next_peak_record_armed = True
        self.next_peak_record_count = 0
        self._set_next_peak_record_status(f"正在记录峰值: {label} | 已采集 0 条")
        self._sync_peak_record_ui()
        if hasattr(self, "statusbar"):
            self.statusbar.showMessage(f"已开始记录峰值样本: {label}", 3000)
        return True, self.next_peak_record_status

    def cancel_next_peak_record(self):
        label = self.next_peak_record_label
        count = int(getattr(self, "next_peak_record_count", 0) or 0)
        self.next_peak_record_armed = False
        self.next_peak_record_label = ""
        self.next_peak_record_count = 0
        if label:
            self._set_next_peak_record_status(f"已结束记录峰值: {label} | 共采集 {count} 条")
        else:
            self._set_next_peak_record_status("未启动")
        self._sync_peak_record_ui()
        if hasattr(self, "statusbar"):
            self.statusbar.showMessage("已结束峰值记录。", 2000)

    def _set_pre_peak_window_record_status(self, status_text):
        status_text = str(status_text or "未启动")
        if status_text == self.pre_peak_window_record_status:
            return
        self.pre_peak_window_record_status = status_text
        self._sync_peak_record_ui()

    def is_pre_peak_window_record_armed(self):
        return bool(self.pre_peak_window_record_armed)

    def arm_pre_peak_window_record(self, label_text):
        label = self._task_label(label_text)
        if not label:
            return False, "请先输入峰前连续数据标签。"

        self.pre_peak_window_record_label = label
        self.pre_peak_window_record_armed = True
        self.pre_peak_window_record_count = 0
        self.pre_peak_window_sample_count = 0
        frames = max(0, int(PEAK_PRE_PEAK_SAVE_FRAMES))
        self._set_pre_peak_window_record_status(
            f"正在记录: {label} | 窗口 {frames}+1 帧 | 已采集 0 组/0 行"
        )
        self._sync_peak_record_ui()
        if hasattr(self, "statusbar"):
            self.statusbar.showMessage(f"已开始记录峰前连续数据: {label}", 3000)
        return True, self.pre_peak_window_record_status

    def cancel_pre_peak_window_record(self):
        label = self.pre_peak_window_record_label
        row_count = int(getattr(self, "pre_peak_window_record_count", 0) or 0)
        sample_count = int(getattr(self, "pre_peak_window_sample_count", 0) or 0)
        self.pre_peak_window_record_armed = False
        self.pre_peak_window_record_label = ""
        self.pre_peak_window_record_count = 0
        self.pre_peak_window_sample_count = 0
        if label:
            self._set_pre_peak_window_record_status(
                f"已结束: {label} | 共采集 {sample_count} 组/{row_count} 行"
            )
        else:
            self._set_pre_peak_window_record_status("未启动")
        self._sync_peak_record_ui()
        if hasattr(self, "statusbar"):
            self.statusbar.showMessage("已结束峰前连续数据记录。", 2000)

    def update_next_peak_recording(self, resp_values, pattern_values, timestamp):
        """记录开关开启后，等待主模型寻峰触发；不再独立跑第二套寻峰器。"""
        if not self.next_peak_record_armed:
            return

        label = self.next_peak_record_label.strip()
        if not label:
            self.cancel_next_peak_record()
            return

    def _record_model_peak_sample_if_active(self, event):
        """把主模型实际使用的峰值样本追加到数据集，确保保存样本与推理样本一致。"""
        if not self.next_peak_record_armed:
            return

        label = self.next_peak_record_label.strip()
        if not label:
            self.cancel_next_peak_record()
            return

        peak_record = event.get("candidate_dataset_record")
        if peak_record:
            peak_record = dict(peak_record)
        else:
            peak_record = dict(self.latest_dataset_record or {})
        if not peak_record:
            self._set_next_peak_record_status(f"记录失败: 当前没有可保存的峰值样本 | {label}")
            return

        peak_record = self.apply_feature_values_to_record(
            peak_record,
            event.get("candidate_resp_values", []),
            event.get("candidate_pattern_values", []),
        )

        candidate_timestamp = event.get("candidate_timestamp")
        peak_candidate_timestamp = event.get("peak_candidate_timestamp", candidate_timestamp)
        peak_record_timestamp = peak_candidate_timestamp or candidate_timestamp
        if peak_record_timestamp is not None:
            time_text = self.format_record_time(peak_record_timestamp)
            peak_record["Time"] = time_text
            peak_record["Timestamp"] = time_text
        peak_record["Label"] = label

        if self.data_collection_dialog is not None:
            self.data_collection_dialog.append_peak_record(peak_record)
            self.next_peak_record_count = int(getattr(self, "next_peak_record_count", 0) or 0) + 1
            status_text = (
                f"正在记录峰值: {label} | 已采集 {self.next_peak_record_count} 条"
                f" | 最近 {peak_record.get('Timestamp', '--')}"
            )
        else:
            status_text = "已检测到峰值，但数据采集窗口未打开，样本未追加。"

        self._set_next_peak_record_status(status_text)
        self._sync_peak_record_ui()
        if hasattr(self, "statusbar"):
            self.statusbar.showMessage(status_text, 4000)

    @staticmethod
    def _event_dataset_entries(event):
        entries = []
        for item in (event.get("event_dataset_records") or event.get("pre_peak_dataset_records") or []):
            if not isinstance(item, dict):
                continue
            record = item.get("record")
            if not record:
                continue
            timestamp = item.get("timestamp")
            try:
                timestamp_value = float(timestamp)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(timestamp_value):
                continue
            entries.append(
                {
                    "timestamp": timestamp_value,
                    "record": copy.deepcopy(record),
                }
            )
        entries.sort(key=lambda item: item["timestamp"])
        return entries

    @staticmethod
    def _records_up_to_timestamp(entries, timestamp, window_len):
        if not entries:
            return []
        window_len = max(1, int(window_len))
        if len(entries) < window_len:
            return []
        try:
            timestamp_value = float(timestamp)
        except (TypeError, ValueError):
            timestamp_value = float(entries[-1]["timestamp"])
        if np.isfinite(timestamp_value):
            candidates = [
                item for item in entries if float(item["timestamp"]) <= timestamp_value + 1e-9
            ]
        else:
            candidates = []
        if len(candidates) < window_len:
            return []
        if len(candidates) > window_len:
            candidates = candidates[-window_len:]
        return candidates

    @staticmethod
    def _raw_peak_timestamp_for_sensor(entries, sensor_idx, window_len=1):
        if sensor_idx < 0 or sensor_idx >= len(RESP_COLUMNS):
            return None
        column = RESP_COLUMNS[sensor_idx]
        best_timestamp = None
        best_abs_value = -1.0
        start_index = max(0, int(window_len) - 1)
        for item in entries[start_index:]:
            record = item.get("record") or {}
            try:
                value = float(record.get(column, 0.0))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue
            abs_value = abs(value)
            if abs_value > best_abs_value:
                best_abs_value = abs_value
                best_timestamp = item.get("timestamp")
        return best_timestamp

    def _build_channel_aligned_pre_peak_window_records(self, event, label, sample_index):
        entries = self._event_dataset_entries(event)
        pre_frames = max(0, int(PEAK_PRE_PEAK_SAVE_FRAMES))
        window_len = pre_frames + 1
        if not entries:
            return [], "事件历史为空，无法保存真实连续窗口"

        peak_timestamps = list(event.get("event_peak_timestamps") or [])

        sensor_windows = []
        for sensor_idx in range(len(RESP_COLUMNS)):
            sensor_peak_timestamp = self._raw_peak_timestamp_for_sensor(
                entries,
                sensor_idx,
                window_len,
            )
            if sensor_peak_timestamp is None:
                sensor_peak_timestamp = (
                    peak_timestamps[sensor_idx]
                    if sensor_idx < len(peak_timestamps)
                    else entries[-1]["timestamp"]
                )
            sensor_window = self._records_up_to_timestamp(
                entries,
                sensor_peak_timestamp,
                window_len,
            )
            if len(sensor_window) < window_len:
                sensor_window = entries[-window_len:] if len(entries) >= window_len else []
            if len(sensor_window) < window_len:
                return [], f"事件历史不足 {window_len} 帧"
            sensor_windows.append(sensor_window)

        reference_timestamp = event.get(
            "peak_candidate_timestamp", event.get("candidate_timestamp")
        )
        reference_window = self._records_up_to_timestamp(
            entries, reference_timestamp, window_len
        )
        if len(reference_window) < window_len:
            reference_window = sensor_windows[0]

        out_records = []
        sensor_column_groups = (
            ADC_COLUMNS,
            R_RAW_COLUMNS,
            R_COMP_COLUMNS,
            BASELINE_COLUMNS,
            RESP_COLUMNS,
        )
        for frame_idx in range(window_len):
            row = copy.deepcopy(reference_window[frame_idx]["record"])
            for sensor_idx in range(len(RESP_COLUMNS)):
                source_record = sensor_windows[sensor_idx][frame_idx]["record"]
                for columns in sensor_column_groups:
                    column = columns[sensor_idx]
                    if column in source_record:
                        row[column] = source_record.get(column)

            resp_values = []
            for column in RESP_COLUMNS:
                try:
                    value = float(row.get(column, 0.0))
                except (TypeError, ValueError):
                    value = 0.0
                resp_values.append(value if np.isfinite(value) else 0.0)
            pattern_values = PeakDetector._pattern_values_from_resp(resp_values)
            row = self.apply_feature_values_to_record(row, resp_values, pattern_values)
            row["Label"] = label
            row[SAMPLE_INDEX_COLUMN] = sample_index
            out_records.append(row)

        return out_records, ""

    def _record_pre_peak_window_sample_if_active(self, event):
        if not self.pre_peak_window_record_armed:
            return

        label = self.pre_peak_window_record_label.strip()
        if not label:
            self.cancel_pre_peak_window_record()
            return

        next_sample_number = int(getattr(self, "pre_peak_window_sample_count", 0) or 0) + 1
        sample_index = f"{label}_{next_sample_number:04d}"
        records, error_text = self._build_channel_aligned_pre_peak_window_records(
            event,
            label,
            sample_index,
        )
        if not records:
            self._set_pre_peak_window_record_status(
                f"记录失败: {error_text or '无可保存连续数据'} | {label}"
            )
            return

        if self.data_collection_dialog is not None:
            for record in records:
                self.data_collection_dialog.append_peak_record(record)
            self.pre_peak_window_sample_count = next_sample_number
            self.pre_peak_window_record_count = (
                int(getattr(self, "pre_peak_window_record_count", 0) or 0)
                + len(records)
            )
            status_text = (
                f"正在记录: {label} | 已采集 "
                f"{self.pre_peak_window_sample_count} 组/"
                f"{self.pre_peak_window_record_count} 行"
                f" | 最近 {sample_index}"
            )
        else:
            status_text = "已检测到峰值，但数据采集窗口未打开，连续数据未追加。"

        self._set_pre_peak_window_record_status(status_text)
        self._sync_peak_record_ui()
        if hasattr(self, "statusbar"):
            self.statusbar.showMessage(status_text, 4000)

    def get_current_dataset_record(self, label_text):
        """把当前实时值组织成标准化数据集记录。"""
        if not self.latest_dataset_record:
            return None

        label = self._task_label(label_text)
        if not label:
            return None

        record = dict(self.latest_dataset_record)
        record["Label"] = label
        record["Timestamp"] = self.format_record_time()
        return record

    def get_buffer_dataset_records(self):
        """把当前缓存区数据组织成标准化数据集记录列表。"""
        records = []
        for row in self.data_buffer:
            if len(row) < len(STANDARD_COLUMNS):
                continue
            records.append(dict(zip(STANDARD_COLUMNS, row)))
        return records

    def init_value_labels(self):
        # 将界面中的 18 个数值标签集中到列表中，便于统一更新。
        self.value_labels = [
            self.Sensor1_value, self.Sensor2_value, self.Sensor3_value, self.Sensor4_value,
            self.Sensor5_value, self.Sensor6_value, self.Sensor7_value, self.Sensor8_value,
            self.Sensor9_value, self.Sensor10_value, self.Sensor11_value, self.Sensor12_value,
            self.Sensor13_value, self.Sensor14_value, self.Sensor15_value, self.Sensor16_value,
            self.Sensor17_value, self.Sensor18_value
        ]
        for idx, lbl in enumerate(self.value_labels):
            if idx < 16:
                lbl.setTextFormat(QtCore.Qt.RichText)
            lbl.setText("--")

    def on_refresh_clicked(self):
        # 刷新当前电脑可用串口列表。
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.reflesh_comboBox.clear()
        self.reflesh_comboBox.addItems(ports)
        self.refresh_voc_port_combos(ports)
        self.refresh_main_udp_endpoint_display()

    def on_confirm_link_clicked(self):
        if not self.get_device_id_text():
            QtWidgets.QMessageBox.warning(
                self,
                "提示",
                "请先输入设备号后再连接串口。",
            )
            return
        # 若已有线程运行，先安全停止，避免重复打开串口
        if self.serial_thread and self.serial_thread.isRunning():
            if self.serial_thread.stop():
                self.serial_thread = None
            else:
                self.show_error_message("错误", "旧串口线程停止超时，请检查串口设备连接。")
                return

        port = self.reflesh_comboBox.currentText()
        baudrate = int(self.rate_input.text())
        if not port:
            QtWidgets.QMessageBox.warning(self, "警告", "请先选择串口号！")
            return

        # 创建并启动串口读取线程。
        self.serial_thread = SerialThread(port, baudrate)
        self.serial_thread.data_received.connect(self.update_plot)
        self.serial_thread.error_occurred.connect(self.on_serial_error)
        self.serial_thread.start()

        self.system_start_time = time.time()
        if self.system_info_label:
            self.update_system_info_display(self.system_start_time)

        self.set_udp_result_config(enabled=True, port=self.udp_result_port)
        udp_endpoint = self.udp_local_endpoint_text(self.udp_result_port)

        QtWidgets.QMessageBox.information(
            self,
            "提示",
            f"已连接到 {port}，波特率 {baudrate}\nUDP服务正在启动：{udp_endpoint}",
        )

    def update_plot(self, values):
        # 这里values是字典，包含键："adc", "temp", "humid"
        # timestamp没有在串口数据里传输，假如你需要时间戳，自己生成
        timestamp = time.time()
        time_str = self.format_record_time(timestamp)

        # 取出 16 路 ADC 和温湿度，并整理成统一的 18 通道数组。
        adc_values = values.get("adc", [0] * 16)
        resistance_values = [self.adc_to_resistance(adc) for adc in adc_values]
        temp_value = values.get("temp", 0)
        humid_value = values.get("humid", 0)
        # 补偿后电阻：系数全零时与原始相同；特征/数据集/baseline 均在此域计算。
        r_comp = apply_th_comp_resistances(
            resistance_values,
            temp_value,
            humid_value,
            self.th_comp_coeffs,
            dlog_beta=TH_COMP_DLOG_BETA_DEFAULT,
        )
        baseline_values_for_resp = list(self.baseline_values)

        # 组合成18通道数据，前16为原始电阻（曲线展示），后2为温湿度
        sensor_values = resistance_values + [temp_value, humid_value]
        self.latest_sensor_values = sensor_values
        self._append_limited(self.xdata, self.sample_index)
        self.sample_index += 1

        # 逐通道刷新主曲线、baseline 曲线以及界面标签。
        for i, value in enumerate(sensor_values):
            self._append_limited(self.ydata[i], value)
            self.curves[i].setData(self.xdata, self.ydata[i])

            if i < 16:
                # 当前帧 resp 使用进入本帧时的 baseline；这里更新的是供下一帧使用的 baseline。
                baseline_value, alpha_used = self.update_sensor_baseline(i, r_comp[i], timestamp)
                self._append_limited(self.baseline_ydata[i], baseline_value)
                self.baseline_curves[i].setData(self.xdata, self.baseline_ydata[i])
                self._append_limited(self.compensated_ydata[i], r_comp[i])
                self.comp_curves[i].setData(self.xdata, self.compensated_ydata[i])
                self.value_labels[i].setText(
                    self._format_sensor_value_label(
                        i, value, r_comp[i], baseline_value, alpha_used
                    )
                )
            else:
                self.value_labels[i].setText(f"{value:.2f}")

        if hasattr(self, "temperature_humidity_value_label"):
            self.temperature_humidity_value_label.setText(
                f"T: {temp_value:.2f}  H: {humid_value:.2f}"
            )

        resp_values, pattern_values = self.build_feature_vectors(
            r_comp,
            baseline_values_for_resp,
        )
        group_zscore_resp_values = compute_material_group_zscore_resp_values(resp_values)
        if group_zscore_resp_values is None:
            group_zscore_resp_values = [0.0 for _ in GROUP_ZSCORE_RESP_COLUMNS]
        self.latest_resp_values = list(resp_values)
        for i in range(16):
            self._append_limited(self.resp_ydata[i], resp_values[i])
            self._append_limited(self.pattern_ydata[i], pattern_values[i])
            self._append_limited(
                self.group_zscore_resp_ydata[i],
                group_zscore_resp_values[i],
            )

        baseline_resistance_values = [
            np.nan if br is None else br for br in baseline_values_for_resp
        ]

        self.latest_dataset_record = self.build_dataset_record(
            time_str,
            adc_values,
            resistance_values,
            r_comp,
            baseline_resistance_values,
            resp_values,
            pattern_values,
            temp_value,
            humid_value,
        )

        self.update_next_peak_recording(resp_values, pattern_values, timestamp)
        self.run_realtime_inference(resp_values, pattern_values, timestamp)
        self.update_system_info_display(timestamp)
        self.update_feature_window_data()

        # 保存时缓存写入
        if self.is_saving:
            if len(self.data_buffer) >= SAVE_BUFFER_MAX_ROWS:
                self.abort_saving_due_to_error(
                    f"保存缓存超过 {SAVE_BUFFER_MAX_ROWS} 行，已停止保存，避免长时间运行占用过多内存。"
                )
            else:
                label = self.current_raw_save_label
                row_record = dict(self.latest_dataset_record)
                row_record["Label"] = label
                self.data_buffer.append(self.build_dataset_row(row_record))
                self.update_data_collection_save_status()

    def _close_raw_save_streams(self):
        """关闭电子鼻和三路VOC保存文件。"""
        try:
            if self.save_file and not self.save_file.closed:
                self.save_file.close()
        except Exception:
            pass
        self.save_file = None
        self.csv_writer = None

        for save_file in list(self.voc_save_files.values()):
            try:
                if save_file and not save_file.closed:
                    save_file.close()
            except Exception:
                pass
        self.voc_save_files = {}
        self.voc_csv_writers = {}

    def _clear_raw_save_buffers(self):
        self.data_buffer.clear()
        for key, _ in VOC_POSITIONS:
            self.voc_save_buffers.setdefault(key, []).clear()

    def abort_saving_due_to_error(self, message):
        """保存链路异常时停止采集缓存，避免后台持续占用内存。"""
        self.save_error_stopped = True
        self.is_saving = False
        self.raw_save_status_text = "保存失败"
        if self.save_timer.isActive():
            self.save_timer.stop()
        self._clear_raw_save_buffers()
        self._close_raw_save_streams()
        self.show_error_message("错误", message)
        self.update_data_collection_save_status()

    def flush_buffer_to_file(self, force=False):
        # force=True 允许在停止保存时也写入最后残留的数据
        if not self.is_saving and not force:
            return True

        has_voc_data = any(
            self.voc_save_buffers.get(key) for key, _ in VOC_POSITIONS
        )
        if not self.data_buffer and not has_voc_data:
            return True

        try:
            # 批量写入缓存，减少频繁 IO 对界面的影响。
            if self.data_buffer:
                if not self.csv_writer or not self.save_file:
                    raise RuntimeError("电子鼻保存文件未就绪")
                rows_to_write = self._apply_score_threshold_entries_to_rows(
                    self.data_buffer,
                    self.raw_save_rows_written,
                )
                self.csv_writer.writerows(rows_to_write)
                self.save_file.flush()
                if force:
                    os.fsync(self.save_file.fileno())
                self.raw_save_rows_written += len(rows_to_write)
                self.data_buffer.clear()

            for key, label in VOC_POSITIONS:
                buffer = self.voc_save_buffers.get(key, [])
                if not buffer:
                    continue
                writer = self.voc_csv_writers.get(key)
                save_file = self.voc_save_files.get(key)
                if writer is None or save_file is None:
                    raise RuntimeError(f"{label} VOC保存文件未就绪")
                writer.writerows(buffer)
                save_file.flush()
                if force:
                    os.fsync(save_file.fileno())
                self.voc_save_rows_written[key] = (
                    int(self.voc_save_rows_written.get(key, 0))
                    + len(buffer)
                )
                buffer.clear()
            self.update_data_collection_save_status()
            return True
        except Exception as e:
            self.abort_saving_due_to_error(f"自动写入失败，已停止保存：{e}")
            return False

    def clear_plot(self):
        # 清空时间轴、主曲线、baseline 曲线和界面显示。
        self.xdata.clear()
        self.sample_index = 0
        for i in range(18):
            self.ydata[i].clear()
            self.curves[i].clear()
            if i in E_NOSE_VISIBLE_CHANNEL_INDICES:
                self.value_labels[i].setText(
                    f'<span style="font-weight: 700;">Sensor {i + 1}</span><br/>--'
                )
            else:
                self.value_labels[i].setText("--")
        if hasattr(self, "temperature_humidity_value_label"):
            self.temperature_humidity_value_label.setText("T: --  H: --")
        for i in range(16):
            self.baseline_ydata[i].clear()
            self.compensated_ydata[i].clear()
            self.resp_ydata[i].clear()
            self.pattern_ydata[i].clear()
            self.group_zscore_resp_ydata[i].clear()
            self.baseline_values[i] = None
            self.baseline_start_times[i] = None
            self.baseline_curves[i].clear()
            self.comp_curves[i].clear()
        if self.system_info_label:
            self.update_system_info_display(time.time())
        if self.feature_window is not None:
            self.feature_window.clear_data()
        self.last_inference_result = None
        self.last_committed_recognition_result = None
        self.inference_history = []
        self.reset_peak_detector()
        if self.next_peak_record_armed:
            self.cancel_next_peak_record()
        else:
            self.peak_record_detector.reset()
        if self.pre_peak_window_record_armed:
            self.cancel_pre_peak_window_record()
        self.latest_dataset_record = None
        self.latest_resp_values = None
        self.reset_voc_events(clear_curves=True)
        self.update_model_center_inference_display()
        self.update_system_info_display(time.time())

    def start_saving(self, label_text=None):
        # 以“标签_时间戳”的格式创建新的原始数据 csv 文件。
        if self.is_saving:
            QtWidgets.QMessageBox.information(self, "提示", "当前已在保存数据。")
            return
        if label_text is None:
            label = self.current_raw_save_label
        else:
            label = str(label_text or "").strip()
        label = (self.task_config.get("label_merge_rule") or {}).get(label, label)
        if not label:
            QtWidgets.QMessageBox.warning(self, "警告", "请先输入标签名！")
            return
        self.current_raw_save_label = label

        # 数据按日期分目录，目录名形如 20251224
        date_dir = time.strftime("%Y%m%d", time.localtime())
        save_dir = os.path.join(self.get_raw_root_dir(), date_dir)
        os.makedirs(save_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        safe_label = sanitize_path_segment(label, "unlabeled")
        filename = os.path.join(save_dir, f"{safe_label}_{timestamp}.csv")
        stem_path = os.path.splitext(filename)[0]
        voc_paths = {
            key: f"{stem_path}_VOC_{position_label}.csv"
            for key, position_label in VOC_POSITIONS
        }

        opened_files = []
        created_paths = []
        try:
            save_file = open(filename, "w", newline="", encoding="utf-8-sig")
            opened_files.append(save_file)
            created_paths.append(filename)
            csv_writer = csv.writer(save_file)
            # CSV 头：Time + ADC + R_raw + R_comp + baseline + resp + centered_resp + pattern + 温湿度 + 时间戳 + Label
            headers = STANDARD_COLUMNS
            csv_writer.writerow(headers)
            save_file.flush()

            voc_save_files = {}
            voc_csv_writers = {}
            for key, position_label in VOC_POSITIONS:
                voc_file = open(
                    voc_paths[key],
                    "w",
                    newline="",
                    encoding="utf-8-sig",
                )
                opened_files.append(voc_file)
                created_paths.append(voc_paths[key])
                voc_writer = csv.writer(voc_file)
                voc_writer.writerow(["序号", "时间", "位置", "串口", "R"])
                voc_file.flush()
                voc_save_files[key] = voc_file
                voc_csv_writers[key] = voc_writer

            self.save_file = save_file
            self.csv_writer = csv_writer
            self.voc_save_files = voc_save_files
            self.voc_csv_writers = voc_csv_writers
            self.current_raw_save_paths = {
                "electronic_nose": filename,
                **voc_paths,
            }
            self._clear_raw_save_buffers()
            self.save_error_stopped = False
            self.is_saving = True
            self.raw_save_rows_written = 0
            self.voc_save_rows_written = {
                key: 0 for key, _ in VOC_POSITIONS
            }
            self.raw_save_status_text = "正在保存"

            self.save_timer.start(5000)  # 5秒自动保存
            self.update_data_collection_save_status()

            QtWidgets.QMessageBox.information(
                self,
                "提示",
                "已开始保存电子鼻和3路VOC数据：\n"
                f"{os.path.abspath(save_dir)}",
            )
        except Exception as e:
            for opened_file in opened_files:
                try:
                    opened_file.close()
                except Exception:
                    pass
            for path in created_paths:
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except OSError:
                    pass
            self._close_raw_save_streams()
            self.raw_save_status_text = "文件创建失败"
            self.update_data_collection_save_status()
            self.show_error_message("错误", f"文件创建失败：{e}")

    def stop_saving(self):
        # 停止定时保存，并把缓存中的最后一批数据写入磁盘。
        if not self.is_saving:
            QtWidgets.QMessageBox.information(self, "提示", "当前没有在保存数据。")
            return

        self.save_timer.stop()

        # 最后一次写入所有数据
        if not self.flush_buffer_to_file(force=True):
            return
        self.is_saving = False

        self._close_raw_save_streams()
        voc_summary = " / ".join(
            f"{label} {int(self.voc_save_rows_written.get(key, 0))} 条"
            for key, label in VOC_POSITIONS
        )
        self.raw_save_status_text = (
            f"已结束 | 电子鼻 {self.raw_save_rows_written} 条"
            f" | VOC {voc_summary}"
        )
        self.update_data_collection_save_status()

        QtWidgets.QMessageBox.information(
            self,
            "提示",
            "数据保存已结束。\n"
            f"电子鼻 {self.raw_save_rows_written} 条；VOC {voc_summary}",
        )

    def on_serial_error(self, message):
        """统一处理串口错误并关闭线程"""
        self.show_error_message("错误", message)
        if self.serial_thread and self.serial_thread.isRunning():
            if self.serial_thread.stop():
                self.serial_thread = None
            else:
                if hasattr(self, "statusbar"):
                    self.statusbar.showMessage("串口线程停止超时，请检查串口设备连接。", 5000)

    def closeEvent(self, event):
        # 关闭窗口前停止线程、停止定时器，并尽量完成最后的数据落盘。
        self.stop_voc_collection()
        if self.serial_thread and not self.serial_thread.stop():
            event.ignore()
            if hasattr(self, "statusbar"):
                self.statusbar.showMessage("串口线程停止超时，请检查串口设备连接后再关闭。", 5000)
            return
        self.serial_thread = None
        self.stop_udp_registration_server()
        if self.data_collection_dialog is not None:
            self.data_collection_dialog.close()
        if self.model_center_dialog is not None:
            self.model_center_dialog.close()
        if self.feature_window is not None:
            self.feature_window.close()
        if self.save_timer.isActive():
            self.save_timer.stop()
        if self.is_saving:
            self.flush_buffer_to_file(force=True)
            self.is_saving = False
        self._close_raw_save_streams()
        event.accept()
