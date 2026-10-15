# -*- coding: utf-8 -*-
"""Functional MainWindow mixin mechanically moved from V4.6.

Keeping these methods in focused modules controls file size while preserving the
original widget layout and signal behavior.
"""

from runtime_support import *
from acquisition.serial_threads import UDPRegistrationThread
from gui.feature_window import FeatureWindow


class MainWindowNetworkMixin:
    def open_feature_window(self):
        """打开特征窗口；若已打开则直接刷新并置顶。"""
        if self.feature_window is None:
            self.feature_window = FeatureWindow(self)

        self.feature_window.set_feature_data(
            self.xdata,
            self.resp_ydata,
            self.group_zscore_resp_ydata,
        )
        self.feature_window.show()
        self.feature_window.raise_()
        self.feature_window.activateWindow()

    def load_model_bundle_from_path(self, model_path):
        """加载模型文件并同步到界面状态。"""
        bundle = load_model_bundle(model_path)
        model_labels = set(bundle.get("class_names") or [])
        task_labels = set(self.task_config["class_labels"])
        if model_labels != task_labels:
            raise ValueError(
                "模型标签{}与当前任务{}不一致，请使用本工具重新训练。".format(
                    sorted(model_labels), self.task_config["class_labels"]
                )
            )
        bundle["task_config"] = dict(self.task_config)
        self.loaded_model_bundle = bundle
        self.loaded_model_path = model_path
        self.last_inference_result = None
        self.last_committed_recognition_result = None
        self.inference_history = []
        self.voc_fusion_results.clear()
        self.voc_fusion_first_seen_at.clear()
        self.reset_peak_detector()
        self.update_model_center_inference_display()
        if hasattr(self, "statusbar"):
            self.statusbar.showMessage(f"已加载模型: {model_path}", 5000)

    def clear_loaded_model_bundle(self):
        """卸载当前模型，并关闭实时推理。"""
        self.loaded_model_bundle = None
        self.loaded_model_path = ""
        self.inference_enabled = False
        self.last_inference_result = None
        self.last_committed_recognition_result = None
        self.inference_history = []
        self.voc_fusion_results.clear()
        self.voc_fusion_first_seen_at.clear()
        self.reset_peak_detector()
        self.update_model_center_inference_display()
        self.update_system_info_display(time.time())

    def set_inference_enabled(self, enabled):
        """切换实时推理开关。"""
        self.inference_enabled = bool(enabled) and self.loaded_model_bundle is not None
        self.reset_peak_detector()
        if not self.inference_enabled:
            self.last_committed_recognition_result = None
        self.update_model_center_inference_display()
        self.update_system_info_display(time.time())

    def udp_local_endpoint_text(self, port=None):
        try:
            port = int(port if port is not None else self.udp_result_port)
        except (TypeError, ValueError):
            port = int(UDP_RESULT_DEFAULT_PORT)
        ips = get_lan_ips()
        port = max(1, min(65535, port))
        if ips:
            return " / ".join(f"{ip}:{port}" for ip in ips)
        return f"未检测到本机局域网IP:{port}"

    def set_udp_result_config(self, enabled=None, host=None, port=None):
        if port is not None:
            try:
                port = int(port)
            except (TypeError, ValueError):
                port = int(UDP_RESULT_DEFAULT_PORT)
            self.udp_result_port = max(1, min(65535, port))

        desired_enabled = self.udp_result_enabled if enabled is None else bool(enabled)
        if desired_enabled:
            self.start_udp_registration_server(self.udp_result_port)
        else:
            self.stop_udp_registration_server()
            self.udp_result_enabled = False
            self.udp_registered_client_addr = None
            self.udp_registered_payload = None
            self.last_udp_result_payload = None
            self.udp_result_status = "UDP: 未开启"
            self.update_model_center_udp_status()

    def start_udp_registration_server(self, port):
        port = max(1, min(65535, int(port)))
        if (
            self.udp_server_thread is not None
            and self.udp_server_thread.isRunning()
            and int(getattr(self.udp_server_thread, "port", 0)) == port
        ):
            self.udp_result_enabled = True
            if self.udp_registered_client_addr:
                host, client_port = self.udp_registered_client_addr
                self.udp_result_status = f"UDP: 车机已注册 {host}:{client_port}"
            else:
                self.udp_result_status = (
                    f"UDP: 服务已开启，本机 {self.udp_local_endpoint_text(port)}，等待车机注册"
                )
            self.update_model_center_udp_status()
            return

        self.stop_udp_registration_server()
        self.udp_result_port = port
        self.udp_registered_client_addr = None
        self.udp_registered_payload = None
        self.last_udp_result_payload = None
        self.udp_server_thread = UDPRegistrationThread(port, self)
        self.udp_server_thread.registered.connect(self.on_udp_client_registered)
        self.udp_server_thread.status_changed.connect(self.on_udp_server_status)
        self.udp_server_thread.failed.connect(self.on_udp_server_failed)
        self.udp_server_thread.finished.connect(self.on_udp_server_finished)
        self.udp_server_thread.start()
        self.udp_result_enabled = True
        self.udp_result_status = (
            f"UDP: 服务启动中，本机 {self.udp_local_endpoint_text(port)}"
        )
        self.update_model_center_udp_status()

    def stop_udp_registration_server(self):
        thread = self.udp_server_thread
        self.udp_server_thread = None
        if thread is not None:
            try:
                thread.stop()
            except Exception:
                pass

    def on_udp_server_status(self, status):
        self.udp_result_status = str(status or "")
        self.update_model_center_udp_status()

    def on_udp_server_failed(self, message):
        if self.sender() is self.udp_server_thread:
            self.udp_server_thread = None
        self.udp_result_enabled = False
        self.udp_registered_client_addr = None
        self.udp_registered_payload = None
        self.last_udp_result_payload = None
        self.udp_result_status = f"UDP: 服务启动失败 {message}"
        self.update_model_center_udp_status()
        if hasattr(self, "statusbar"):
            self.statusbar.showMessage(self.udp_result_status, 5000)

    def on_udp_server_finished(self):
        if self.sender() is not self.udp_server_thread:
            return
        self.udp_server_thread = None
        if not self.udp_result_enabled:
            return
        self.udp_registered_client_addr = None
        self.udp_registered_payload = None
        self.last_udp_result_payload = None
        self.udp_result_status = "UDP: 服务中断，重新开启并等待车机注册"
        self.update_model_center_udp_status()
        QtCore.QTimer.singleShot(
            0,
            lambda: (
                self.start_udp_registration_server(self.udp_result_port)
                if self.udp_result_enabled and self.udp_server_thread is None
                else None
            ),
        )

    def on_udp_client_registered(self, host, port, payload):
        self.udp_registered_client_addr = (str(host), int(port))
        self.udp_registered_payload = dict(payload or {})
        self.last_udp_result_payload = None
        self.udp_result_status = f"UDP: 车机已注册 {host}:{int(port)}"
        self.update_model_center_udp_status()
        if hasattr(self, "statusbar"):
            self.statusbar.showMessage(self.udp_result_status, 5000)

    def _udp_registration_server_running(self):
        thread = self.udp_server_thread
        return bool(thread is not None and thread.isRunning())

    def _restart_udp_registration_after_interruption(self, reason=""):
        if not self.udp_result_enabled:
            return
        self.stop_udp_registration_server()
        self.udp_registered_client_addr = None
        self.udp_registered_payload = None
        self.last_udp_result_payload = None
        suffix = f"（{reason}）" if reason else ""
        self.udp_result_status = f"UDP: 服务中断{suffix}，重新开启并等待车机注册"
        self.update_model_center_udp_status()
        self.start_udp_registration_server(self.udp_result_port)

    def _ensure_udp_registration_server_running(self):
        if not self.udp_result_enabled:
            return False
        if self._udp_registration_server_running():
            return True
        self._restart_udp_registration_after_interruption("UDP服务未开启")
        return False

    def update_model_center_udp_status(self):
        self.refresh_main_udp_endpoint_display()
        if (
            self.model_center_dialog is not None
            and hasattr(self.model_center_dialog, "sync_udp_result_status")
        ):
            self.model_center_dialog.sync_udp_result_status()

    def update_data_collection_save_status(self):
        if (
            self.data_collection_dialog is not None
            and hasattr(self.data_collection_dialog, "refresh_raw_save_status")
        ):
            self.data_collection_dialog.refresh_raw_save_status()

    @staticmethod
    def udp_class_from_label_static(label, mapping):
        label = str(label or "").strip()
        return mapping.get(label, "")

    def udp_class_from_label(self, label):
        return self.udp_class_from_label_static(
            label, udp_class_label_map(self.task_config)
        )

    @staticmethod
    def udp_level_from_result(result):
        level = str((result or {}).get("odor_level", "")).strip()
        return normalize_odor_level_output(level)

    def udp_position_from_result(self, result):
        """UDP 位置：0不明确、1主驾、2副驾、3后排。"""
        udp_class = self.udp_class_from_label(
            (result or {}).get("predicted_label")
        )
        if udp_class != "alcohol":
            return int(UDP_POSITION_NONE)

        fusion = self.evaluate_driver_drinking_decision()
        if not fusion.get("final") or fusion.get("position_ambiguous"):
            return int(UDP_POSITION_NONE)
        return int(
            UDP_POSITION_BY_KEY.get(
                str(fusion.get("first_key") or ""),
                UDP_POSITION_NONE,
            )
        )

    def build_udp_recognition_payload(self, result):
        result = result or {}
        try:
            timestamp_ms = float(result.get("timestamp_ms"))
        except (TypeError, ValueError):
            timestamp_ms = float("nan")
        if not np.isfinite(timestamp_ms):
            timestamp_ms = time.time() * 1000.0

        self.udp_result_serial_no += 1
        return {
            "type": UDP_DATA_TYPE,
            "timestamp": int(round(timestamp_ms)),
            "serialNo": int(self.udp_result_serial_no),
            "class": self.udp_class_from_label(result.get("predicted_label")),
            "level": self.udp_level_from_result(result),
            "position": self.udp_position_from_result(result),
        }

    def send_udp_air_result_if_needed(self, result):
        if self.udp_class_from_label((result or {}).get("predicted_label")) != "air":
            return
        self.send_udp_frame_result(result)

    def send_udp_level_update_if_needed(self, result):
        udp_class = self.udp_class_from_label((result or {}).get("predicted_label"))
        if udp_class not in {"alcohol", "smoke", "abnormalOdor"}:
            return
        self.send_udp_frame_result(result)

    def send_udp_frame_result(self, result):
        if not result:
            return
        label = str((result or {}).get("predicted_label", "")).strip()
        if not label or label == "识别中":
            return
        if not self.udp_class_from_label(label):
            return
        self.send_udp_recognition_result(result)

    def send_udp_recognition_result(self, result):
        if not self.udp_result_enabled:
            return
        if not self._ensure_udp_registration_server_running():
            return
        if not self.udp_registered_client_addr:
            self.udp_result_status = "UDP: 等待车机注册"
            self.update_model_center_udp_status()
            return

        payload = self.build_udp_recognition_payload(result or {})
        try:
            thread = self.udp_server_thread
            if thread is None or not thread.isRunning():
                raise RuntimeError("UDP服务未开启")
            ok, message = thread.send_json(payload, self.udp_registered_client_addr)
            if not ok:
                raise RuntimeError(message)
        except Exception as exc:
            message = str(exc)
            if "UDP服务未开启" in message:
                self._restart_udp_registration_after_interruption(message)
                return
            self.udp_result_status = f"UDP: 发送失败 {exc}"
            self.update_model_center_udp_status()
            if hasattr(self, "statusbar"):
                self.statusbar.showMessage(self.udp_result_status, 3000)
            return

        self.last_udp_result_payload = payload
        host, port = self.udp_registered_client_addr
        self.udp_result_status = (
            f"UDP: 已发送 #{payload['serialNo']} -> "
            f"{host}:{port}"
        )
        self.update_model_center_udp_status()

    def update_model_center_inference_display(self):
        """同步主界面与模型训练/推理窗口中的推理显示。"""
        self.maybe_commit_recognition_snapshot(self.last_inference_result)
        self.update_main_inference_display()
        if self.model_center_dialog is not None and self.model_center_dialog.isVisible():
            self.model_center_dialog.update_inference_display(self.last_inference_result)
        live_result = self.last_inference_result
        if live_result:
            realtime_result = self.recognition_result_for_display(live_result)
            self.send_udp_frame_result(realtime_result)

    def update_main_inference_display(self):
        """刷新主界面温湿度曲线右侧的实时推理结果。"""
        if not hasattr(self, "main_inference_prediction_label"):
            return

        live_result = getattr(self, "last_inference_result", None)
        disp = self.recognition_result_for_display(live_result) if live_result else None

        prediction = "--"
        odor_level = "--"
        score_text = "--"
        if disp:
            prediction = str(disp.get("predicted_label", "--")).strip() or "--"
            odor_level = odor_level_for_display(disp.get("odor_level", "--"))
        latest_resp_values = getattr(self, "latest_resp_values", None)
        if latest_resp_values is not None:
            dev_raw = self.get_device_id_text()
            score_value = response_strength_score_from_resp_values(
                latest_resp_values, dev_raw
            )
            if np.isfinite(score_value):
                score_text = f"{float(score_value):.2f}"

        self.main_inference_prediction_label.setText(f"当前识别：{prediction}")
        self.main_inference_odor_level_label.setText(f"气味等级：{odor_level}")
        if hasattr(self, "main_inference_score_label"):
            self.main_inference_score_label.setText(f"score：{score_text}")

        self.update_driver_drinking_display()

    def is_inference_recognition_wait_frame(self, result):
        """寻峰等待中：尚未产生新的分类结果，对外沿用上一条确认结果。"""
        if not result:
            return True
        if bool(result.get("recognition_pending")):
            return True
        return str(result.get("predicted_label", "")).strip() == "识别中"

    def maybe_commit_recognition_snapshot(self, result):
        """有新的完整识别帧时更新快照；寻峰等待帧只沿用旧快照。"""
        if not result:
            self.last_committed_recognition_result = None
            return
        if self.is_inference_recognition_wait_frame(result):
            return
        self.last_committed_recognition_result = copy.deepcopy(result)

    def calculate_alcohol_odor_level(self, resp_values):
        """按当前设备标定表和 top3 归一化响应分数计算酒精气味等级。"""
        dev_raw = self.get_device_id_text()
        th = load_conc_calibration_resp_thresholds(dev_raw) if dev_raw else None
        score = response_strength_score_from_resp_values(resp_values, dev_raw)
        if th is not None and np.isfinite(score):
            return odor_level_from_resp_and_cal_thresholds(score, th)
        return "--"

    def calculate_ethanol_equiv_odor_level(self, predicted_label, resp_values):
        """第二类别实时等级：乙醇等效响应强度，不代表真实 ppm 浓度。"""
        dev_raw = self.get_device_id_text()
        strength_score = response_strength_score_from_resp_values(resp_values, dev_raw)
        th = load_conc_calibration_resp_thresholds(dev_raw) if dev_raw else None
        level = response_strength_level_from_ethanol_equiv_thresholds(
            predicted_label,
            strength_score,
            th,
        )
        return strength_score, level

    @staticmethod
    def _copy_live_timestamp_fields(target, live_result):
        if not isinstance(target, dict) or not isinstance(live_result, dict):
            return target
        for key in ("timestamp", "timestamp_ms"):
            if key in live_result:
                target[key] = live_result[key]
        return target

    def recognition_result_for_display(self, live_result):
        """推理区展示用：寻峰等待时保留上一次识别结果；气味等级按当前帧实时刷新。"""
        if not live_result:
            return None
        if (
            self.is_inference_recognition_wait_frame(live_result)
            and self.last_committed_recognition_result
        ):
            display_result = copy.deepcopy(self.last_committed_recognition_result)
        else:
            display_result = live_result
        latest_resp_values = getattr(self, "latest_resp_values", None)
        predicted_label = str(display_result.get("predicted_label", "")).strip()
        if predicted_label == "酒精" and latest_resp_values is not None:
            display_result = copy.deepcopy(display_result)
            display_result["odor_level"] = self.calculate_alcohol_odor_level(
                latest_resp_values
            )
            display_result["odor_level_realtime"] = True
            self._copy_live_timestamp_fields(display_result, live_result)
        elif (
            predicted_label == self.task_config["class_labels"][1]
            and latest_resp_values is not None
        ):
            strength_score, level = self.calculate_ethanol_equiv_odor_level(
                predicted_label,
                latest_resp_values,
            )
            display_result = copy.deepcopy(display_result)
            display_result["odor_strength_score"] = strength_score
            display_result["odor_strength_metric"] = (
                f"top3_resp_over_{RESP_X_CAP_CAL_CONC:g}ppm_gate"
            )
            display_result["odor_strength_reference"] = "ethanol_equivalent_10_50_100ppm"
            display_result["odor_level"] = level
            display_result["odor_level_realtime"] = True
            self._copy_live_timestamp_fields(display_result, live_result)
        return display_result
