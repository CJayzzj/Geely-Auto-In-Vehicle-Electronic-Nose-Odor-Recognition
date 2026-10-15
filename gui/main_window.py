# -*- coding: utf-8 -*-
"""Top-level window composed from focused V4.6 behavior mixins."""

from runtime_support import *
from acquisition.peak_detector import PeakDetector, VocEventDetector
from acquisition.serial_threads import SerialThread
from gui.main_window_acquisition import MainWindowAcquisitionMixin
from gui.main_window_network import MainWindowNetworkMixin
from gui.main_window_sensor import MainWindowSensorMixin
from gui.main_window_setup import MainWindowSetupMixin
from gui.main_window_voc import MainWindowVocMixin
from gui.model_center_dialog import ModelCenterDialog
from gui.plot_widgets import _apply_standard_plot_axes


class MainWindow(
    MainWindowSetupMixin,
    MainWindowVocMixin,
    MainWindowNetworkMixin,
    MainWindowAcquisitionMixin,
    MainWindowSensorMixin,
    QtWidgets.QMainWindow,
    Ui_MainWindow,
):
    def __init__(self, task_config):
        super(MainWindow, self).__init__()
        self.task_config = validate_task_config(task_config)
        self.setupUi(self)
        ensure_runtime_dirs()
        # 程序运行目录与原始数据目录（按设备号分子文件夹，见 get_raw_root_dir）。
        self.base_dir = get_app_dir()
        self.data_dir = DATA_DIR
        self.matched_resistance_value = float(MATCHED_RESISTANCE)
        self._init_device_id_field()
        self._init_matched_resistance_field()
        os.makedirs(self.get_raw_root_dir(), exist_ok=True)
        # 绘图、状态显示与时间轴相关变量。
        self.serial_thread = None
        self.curves = []
        self.baseline_curves = []
        self.setWindowTitle(self.task_config["app_title"])
        self.value_labels = []
        self.xdata = []
        self.sample_index = 0
        self._last_error_text = ""
        self._last_error_time_ms = 0
        # 运行时间从「确认连接」串口成功启动后开始计时；未连接前为 None。
        self.system_start_time = None
        self.system_info_label = None
        self.data_collection_btn = None
        self.data_collection_dialog = None
        self.model_center_btn = None
        self.model_center_dialog = None
        self.feature_btn = None
        self.feature_window = None
        self.th_comp_btn = None
        self.th_comp_dialog = None
        self.th_comp_coeffs = default_th_comp_coeffs()
        self.comp_curves = []
        self.compensated_ydata = [[] for _ in range(16)]
        self.loaded_model_bundle = None
        self.loaded_model_path = ""
        self.inference_enabled = False
        self.last_inference_result = None
        self.last_committed_recognition_result = None
        self.inference_history = []
        self.udp_result_enabled = False
        self.udp_result_port = int(UDP_RESULT_DEFAULT_PORT)
        self.udp_result_serial_no = 0
        self.udp_result_status = "UDP: 未开启"
        self.last_udp_result_payload = None
        self.udp_server_thread = None
        self.udp_registered_client_addr = None
        self.udp_registered_payload = None
        self.main_udp_endpoint_input = None
        self.main_udp_status_label = None
        self.peak_detector = PeakDetector()
        self.reset_peak_detector()
        self.peak_record_detector = PeakDetector()
        self.next_peak_record_armed = False
        self.next_peak_record_label = ""
        self.next_peak_record_status = "未启动"
        self.next_peak_record_count = 0
        self.pre_peak_window_record_armed = False
        self.pre_peak_window_record_label = ""
        self.pre_peak_window_record_status = "未启动"
        self.pre_peak_window_record_count = 0
        self.pre_peak_window_sample_count = 0
        self.latest_dataset_record = None
        self.latest_resp_values = None
        self.voc_threads = {}
        self.voc_error_keys = set()
        self.voc_port_combos = {}
        self.voc_status_labels = {}
        self.voc_r_labels = {}
        self.voc_dr_labels = {}
        self.voc_event_labels = {}
        self.voc_plot_event_labels = {}
        self.voc_time_labels = {}
        self.voc_event_id_labels = {}
        self.voc_fusion_state_labels = {}
        self.voc_plot_widgets = {}
        self.voc_curves = {}
        self.voc_curve_event_marker_items = {
            key: [] for key, _ in VOC_POSITIONS
        }
        self.voc_curve_event_marker_signatures = {
            key: None for key, _ in VOC_POSITIONS
        }
        self.voc_xdata = {key: [] for key, _ in VOC_POSITIONS}
        self.voc_rdata = {key: [] for key, _ in VOC_POSITIONS}
        self.voc_sample_index = {key: 0 for key, _ in VOC_POSITIONS}
        self.voc_event_detectors = {
            key: VocEventDetector(
                VOC_EVENT_DEFAULT_PARAMS[key]["window"],
                VOC_EVENT_DEFAULT_PARAMS[key]["dr_threshold"],
                VOC_EVENT_DEFAULT_PARAMS[key]["duration"],
            )
            for key, _ in VOC_POSITIONS
        }
        self.driver_drinking_decision = "--"
        self.voc_fusion_results = {}
        self.voc_fusion_first_seen_at = {}
        self.voc_active_recognition_id = None
        self.voc_visual_recognition_id = None
        self.voc_visual_anchor_timestamp = None
        self.voc_visual_anchor_source = None
        self.voc_visual_fusion_state = None
        self.voc_cluster_sequence = 0
        self.voc_cluster_records = {}
        self.voc_timeline_dynamic_items = []
        self.voc_timeline_signature = None
        # 扩展到 18 个通道（前 16 为 ADC，17 温度，18 湿度）
        self.ydata = [[] for _ in range(18)]
        self.baseline_ydata = [[] for _ in range(16)]
        self.resp_ydata = [[] for _ in range(16)]
        self.pattern_ydata = [[] for _ in range(16)]
        self.group_zscore_resp_ydata = [[] for _ in range(16)]
        self.baseline_values = [None for _ in range(16)]
        self.baseline_start_times = [None for _ in range(16)]
        # 初始化 UI 交互、绘图区域和数值标签。
        self.init_ui_logic()
        self.init_plot()
        self.init_value_labels()
        self._init_main_inference_result_panel()
        self._configure_visible_e_nose_plots()
        self._init_voc_event_panel()

        # ===== 保存功能变量 =====
        self.is_saving = False
        self.current_raw_save_label = ""
        self.save_file = None
        self.csv_writer = None
        self.raw_save_rows_written = 0
        self.raw_save_status_text = "未保存"
        self.current_raw_save_paths = {}
        self.voc_save_files = {}
        self.voc_csv_writers = {}
        self.voc_save_buffers = {key: [] for key, _ in VOC_POSITIONS}
        self.voc_save_rows_written = {key: 0 for key, _ in VOC_POSITIONS}

        # 缓冲区（新加入）
        self.data_buffer = []
        self.save_error_stopped = False

        # 每5秒写一次定时器（新加入）
        self.save_timer = QtCore.QTimer()
        self.save_timer.timeout.connect(self.flush_buffer_to_file)

        # ===== 绑定保存按钮 =====
        self.save_begin.clicked.connect(self.start_saving)
        self.save_stop.clicked.connect(self.stop_saving)
        try:
            self.save_begin.hide()
            self.save_stop.hide()
        except Exception:
            pass
        self._remove_legacy_main_label_controls()

        self.latest_sensor_values = None  # 最近一次收到的18通道数据
        self._init_model_center_button()
        self._init_feature_button()
        self._init_th_comp_button()
        self._init_system_info_label()
        self._init_main_udp_endpoint_display()
        self._populate_dashboard_control_panel()
        self.reload_th_comp_coeffs_from_disk()
        self._load_default_realtime_model()
        self._initial_window_geometry_done = False
        self._screen_change_window_handle = None
        self._pending_target_screen = None
        self._screen_change_timer = QtCore.QTimer(self)
        self._screen_change_timer.setSingleShot(True)
        self._screen_change_timer.setInterval(120)
        self._screen_change_timer.timeout.connect(self._apply_pending_screen_change)
        try:
            self.device_id_input.textChanged.connect(self.reload_th_comp_coeffs_from_disk)
        except Exception:
            pass
