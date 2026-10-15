# -*- coding: utf-8 -*-
"""Shared runtime constants and sensor-domain helpers.

This module deliberately contains no widget classes. It gives the mechanically
split acquisition and GUI modules one stable import surface while preserving the
V4.6 timing, peak, VOC, concentration, and display behavior.
"""

import copy
import csv
import json
import os
import socket
import struct
import time
from collections import deque
from datetime import datetime
from enum import Enum, auto

import joblib
import numpy as np
import pandas as pd
import pyqtgraph as pg
import serial
import serial.tools.list_ports
from PyQt5 import QtCore, QtGui, QtWidgets

from algorithm.air_gate import (
    INFERENCE_RESP_GATE_SCORE_THRESHOLD,
    INFERENCE_RESP_GATE_TOP_K,
    RESP_X_CAP_CAL_CONC,
)
from algorithm.classifier import predict_feature_vector
from algorithm.features import *
from algorithm.model_io import load_model_bundle, save_model_bundle
from task_config import validate_task_config
from ui.ui_main_window import Ui_MainWindow
from utils.dataset_io import *
from utils.paths import *
from utils.th_compensation import (
    TH_COMP_COEFF_FILENAME,
    ThCompensationDialog,
    apply_th_comp_resistances,
    default_th_comp_coeffs,
    load_th_comp_coeffs_xlsx,
    tighten_plot_layout,
)
#绘图与提示相关配置。
MAX_POINTS = 1000
SERIAL_READ_TIMEOUT_SEC = 0.1
SERIAL_WRITE_TIMEOUT_SEC = 0.2
SERIAL_STOP_TIMEOUT_MS = 1500
SAVE_BUFFER_MAX_ROWS = 30000
VOC_QUERY_COMMAND = bytes.fromhex("FE 00 87 00 00 00 00 00 87")
VOC_FRAME_HEAD = bytes.fromhex("FE FE 68")
VOC_FRAME_TAIL = 0x16
VOC_FRAME_LEN = 15
VOC_DEFAULT_BAUDRATE = 115200
VOC_DEFAULT_INTERVAL_SEC = 1.0
VOC_FRAME_TIMEOUT_SEC = 1.0
VOC_RECONNECT_TIMEOUT_LIMIT = 3
VOC_RECONNECT_INITIAL_DELAY_SEC = 0.5
VOC_RECONNECT_MAX_DELAY_SEC = 5.0
VOC_RECONNECT_ERROR_THROTTLE_SEC = 5.0
VOC_POSITIONS = (("driver", "主驾"), ("passenger", "副驾"), ("rear", "后排"))
VOC_EVENT_DEFAULT_PARAMS = {
    "driver": {"window": 20, "dr_threshold": 10.0, "duration": 10},
    "passenger": {"window": 20, "dr_threshold": 10.0, "duration": 10},
    "rear": {"window": 20, "dr_threshold": 10.0, "duration": 10},
}
VOC_MEDIAN_FILTER_SIZE = 3
VOC_EVENT_RETENTION_SEC = 180.0
VOC_EVENT_HISTORY_LIMIT = 30
VOC_EVENT_INDICATOR_HOLD_SEC = 5.0
VOC_FUSION_LOOKBACK_SEC = 60.0
VOC_FUSION_LOOKAHEAD_SEC = 10.0
VOC_ONSET_LOOKBACK_SEC = 20.0
VOC_ONSET_LOOKAHEAD_SEC = 60.0
VOC_UNMATCHED_EVENT_EXPIRE_SEC = VOC_EVENT_RETENTION_SEC
VOC_EVENT_CLUSTER_GAP_SEC = 20.0
VOC_EVENT_GROUP_MAX_SPAN_SEC = 30.0
VOC_POSITION_AMBIGUITY_SEC = 3.0
E_NOSE_VISIBLE_CHANNEL_INDICES = (0, 4, 8, 12)
MAIN_SIGNAL_PLOT_HEIGHT = 92
MAIN_INFERENCE_PANEL_HEIGHT = 96
DEFAULT_REALTIME_MODEL_RELATIVE_PATH = os.path.join(
    "V2", "20260629_173431_酒精_香烟.joblib"
)
ERROR_DIALOG_THROTTLE_MS = 3000
#ABNORMAL_ODOR_LABEL = "异味"
AIR_LABEL = "空气"
UDP_RESULT_DEFAULT_PORT = 5005
UDP_DATA_TYPE = "data"
UDP_POSITION_NONE = 0
UDP_POSITION_BY_KEY = {"driver": 1, "passenger": 2, "rear": 3}
UDP_REGISTER_TYPE = "register"
UDP_REGISTER_ACK_TYPE = "register_ack"
ALCOHOL_LEVEL_TRACE = "traceAlcohol_0_10ppm"
ALCOHOL_LEVEL_LOW = "lowAlcohol_10_50ppm"
ALCOHOL_LEVEL_POSSIBLE_DUI = "possibleDUI_50_100ppm"
ALCOHOL_LEVEL_POSSIBLE_DWI = "possibleDWI_over100ppm"
ODOR_LEVEL_ENGLISH_MAP = {
    "微量酒精(0-10ppm)": ALCOHOL_LEVEL_TRACE,
    "少量酒精(10-50ppm)": ALCOHOL_LEVEL_LOW,
    "疑似酒驾(50-100ppm)": ALCOHOL_LEVEL_POSSIBLE_DUI,
    "疑似醉驾(>100ppm)": ALCOHOL_LEVEL_POSSIBLE_DWI,
    "微弱烟味": "weakSmoke",
    "中等烟味": "moderateSmoke",
    "强烈烟味": "strongSmoke",
    "微弱异味": "weakAbnormalOdor",
    "中等异味": "moderateAbnormalOdor",
    "强烈异味": "strongAbnormalOdor",
}
ODOR_LEVEL_CHINESE_MAP = {value: key for key, value in ODOR_LEVEL_ENGLISH_MAP.items()}
INFERENCE_RESP_GATE_LABEL = AIR_LABEL
INFERENCE_RESP_GATE_CONFIRM_FRAMES = 2
PEAK_DETECT_MIN_PROMINENCE = 0.03
PEAK_VALLEY_HISTORY_FRAMES = 30
PEAK_PROMINENCE_XCAP_MULTIPLIER = 0.30
PEAK_PROMINENCE_MIN_ABS = PEAK_DETECT_MIN_PROMINENCE
PEAK_PROMINENCE_MAX_ABS = 0.10
PEAK_DETECT_DECLINE_FRAMES = 2
PEAK_GATE_LOW_RELEASE_FRAMES = 3
PEAK_DETECT_MIN_PULLBACK = 0.006
PEAK_DETECT_FLAT_EPS = 0.003
PEAK_DETECT_SMALL_REBOUND_TOL = 0.012
PEAK_TRACK_SIGNAL_SMOOTH_ALPHA = 0.5
PEAK_PRETRIGGER_HISTORY_FRAMES = max(8, PEAK_VALLEY_HISTORY_FRAMES)
PEAK_PRE_PEAK_SAVE_FRAMES = PEAK_DETECT_DECLINE_FRAMES
PEAK_SENSOR_PEAK_SMOOTH_ALPHA = 0.45
PEAK_ACTIVE_SENSOR_GATE_RATIO = 0.5
PEAK_ALL_SENSOR_STABLE_FRAMES = 3
PEAK_ACTIVE_SENSOR_MIN_COUNT = 1
PEAK_POST_CONFIRM_COLLECT_DURATION_S = 1.0
PEAK_EVENT_MAX_DURATION_S = 180.0
PEAK_REARM_DROP_RATIO = 0.25
RESPONSE_STRENGTH_TOP_K = INFERENCE_RESP_GATE_TOP_K
BASELINE_SIGMOID_X_CAP_CAL_CONC = 2.0
BASELINE_SIGMOID_K_DEFAULT = 55.0
BASELINE_SIGMOID_ALPHA_HI_DEFAULT = 0.02
BASELINE_SIGMOID_ALPHA_LO_DEFAULT = 0.00005
BASELINE_SIGMOID_X_C = 0.1
BASELINE_SIGMOID_X_CAP = 0.4
CONC2_RESP_GATE_X_CAP_DEFAULT = 0.1
CONC2_RESP_GATE_X_CAP = [CONC2_RESP_GATE_X_CAP_DEFAULT for _ in range(16)]
BASELINE_INITIAL_ALPHA_DURATION_S = 60.0
BASELINE_INITIAL_ALPHA = 0.5
BASELINE_RISE_FAST_ALPHA = 0.5
TH_COMP_DLOG_BETA_DEFAULT = 0.8
MATCHED_RESISTANCE = 10.0
SENSOR_CURVE_BL_RGB = (220, 80, 80)
SENSOR_CURVE_COMP_RGB = (40, 160, 60)
SAMPLE_INDEX_COLUMN = "sample_index"
DATASET_EXTRA_COLUMNS = (SAMPLE_INDEX_COLUMN,)
_CONC_CAL_RESP_THRESH_CACHE = {}
_BASELINE_SIGMOID_RESP_PARAMS_CACHE = {}


def udp_class_label_map(task_config):
    config = validate_task_config(task_config)
    mapping = {"酒精": "alcohol", "空气": "air"}
    second = config["class_labels"][1]
    mapping[second] = "smokeOrOdor" if second == "异味" else "smoke"
    return mapping


def QtGuiColor(r, g, b):
    return QtGui.QColor(r, g, b)


def conc_calibration_xlsx_path(device_segment_sanitized):
    """
    返回该设备号下的浓度标定表路径；文件不存在则返回 None。

    源码运行：路径为 data/calibration/Conc_Cal/<设备>/Conc_Cal.xlsx。
    PyInstaller -F：__file__ 常在 _MEIPASS；标定表需二选一部署——
    1）把 Conc_Cal 文件夹放在 exe 同目录（与 data/datasets 一致）；或
    2）打包时用 --add-data 把 Conc_Cal 打进包体（解压到 _MEIPASS/Conc_Cal/）。
    下面按顺序探测上述位置。
    """
    if not device_segment_sanitized:
        return None
    rel = os.path.join("data", "calibration", "Conc_Cal", device_segment_sanitized, "Conc_Cal.xlsx")
    candidates = []
    if getattr(sys, "frozen", False):
        candidates.append(os.path.join(get_app_dir(), rel))
        mei = getattr(sys, "_MEIPASS", None)
        if mei:
            candidates.append(os.path.join(mei, rel))
    else:
        candidates.append(os.path.join(CONC_CAL_DIR, device_segment_sanitized, "Conc_Cal.xlsx"))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

def load_conc_calibration_resp_thresholds(device_id_text):
    """
    读取 Conc_Cal.xlsx：用 resp_Sensor1..16 与 Conc==1 每通道阈值计算 top3 归一化响应分数。
    酒精等级阈值由标气表在 10/50/100 ppm 处的 top3 分数插值得到。
    返回 dict: score10,score50,score100,x_cap,path；失败返回 None。
    """
    raw = (device_id_text or "").strip()
    if not raw:
        return None
    safe = sanitize_path_segment(raw, "default")
    path = conc_calibration_xlsx_path(safe)
    if not path:
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    cache_key = (os.path.abspath(path), mtime, float(RESP_X_CAP_CAL_CONC))
    if cache_key in _CONC_CAL_RESP_THRESH_CACHE:
        return _CONC_CAL_RESP_THRESH_CACHE[cache_key]

    try:
        df = pd.read_excel(path)
    except Exception:
        _CONC_CAL_RESP_THRESH_CACHE[cache_key] = None
        return None

    if "Conc" not in df.columns:
        _CONC_CAL_RESP_THRESH_CACHE[cache_key] = None
        return None

    available_columns = [column for column in RESP_COLUMNS if column in df.columns]
    if not available_columns:
        _CONC_CAL_RESP_THRESH_CACHE[cache_key] = None
        return None

    conc = pd.to_numeric(df["Conc"], errors="coerce").values.astype(float)
    resp_values = df[available_columns].apply(pd.to_numeric, errors="coerce").values.astype(float)
    finite_conc = np.isfinite(conc)
    if not np.any(finite_conc):
        _CONC_CAL_RESP_THRESH_CACHE[cache_key] = None
        return None

    x_cap_mask = finite_conc & np.isclose(
        conc, float(RESP_X_CAP_CAL_CONC), rtol=0.0, atol=1e-9
    )
    x_caps = []
    for column_idx, column in enumerate(available_columns):
        sensor_idx = RESP_COLUMNS.index(column)
        values = resp_values[x_cap_mask, column_idx]
        values = values[np.isfinite(values)]
        if values.size:
            x_cap = float(np.mean(np.abs(values)))
        else:
            x_cap = float(CONC2_RESP_GATE_X_CAP[sensor_idx])
        if not np.isfinite(x_cap) or x_cap <= 0.0:
            x_cap = float(CONC2_RESP_GATE_X_CAP[sensor_idx])
        if not np.isfinite(x_cap) or x_cap <= 0.0:
            _CONC_CAL_RESP_THRESH_CACHE[cache_key] = None
            return None
        x_caps.append(x_cap)
    x_caps = np.asarray(x_caps, dtype=float)

    scores = []
    for row in resp_values:
        valid = np.isfinite(row) & np.isfinite(x_caps) & (x_caps > 0.0)
        if not np.any(valid):
            scores.append(float("nan"))
            continue
        ratios = np.abs(row[valid]) / x_caps[valid]
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size == 0:
            scores.append(float("nan"))
            continue
        k = min(max(1, int(RESPONSE_STRENGTH_TOP_K)), ratios.size)
        scores.append(float(np.mean(np.sort(ratios)[-k:])))
    scores = np.asarray(scores, dtype=float)

    valid = finite_conc & np.isfinite(scores)
    if int(np.sum(valid)) < 2:
        _CONC_CAL_RESP_THRESH_CACHE[cache_key] = None
        return None

    agg = (
        pd.DataFrame({"Conc": conc[valid], "score": scores[valid]})
        .groupby("Conc", as_index=False)["score"]
        .mean()
    )
    c = agg["Conc"].values.astype(float)
    score_by_conc = agg["score"].values.astype(float)
    order = np.argsort(c)
    c, score_by_conc = c[order], score_by_conc[order]
    if c.size < 2:
        _CONC_CAL_RESP_THRESH_CACHE[cache_key] = None
        return None

    targets = np.array([10.0, 50.0, 100.0], dtype=float)
    score_interp = np.interp(targets, c, score_by_conc)
    score10, score50, score100 = (
        float(score_interp[0]),
        float(score_interp[1]),
        float(score_interp[2]),
    )
    out = {
        "score10": score10,
        "score50": score50,
        "score100": score100,
        "r10": score10,
        "r50": score50,
        "r100": score100,
        "x_cap": x_caps.tolist(),
        "feature_mode": f"top3_resp_over_{RESP_X_CAP_CAL_CONC:g}ppm_gate",
        "x_cap_conc": float(RESP_X_CAP_CAL_CONC),
        "path": path,
    }
    _CONC_CAL_RESP_THRESH_CACHE[cache_key] = out
    return out

def odor_level_from_resp_and_cal_thresholds(score, th):
    """
    用标定表在 10/50/100 ppm 插值得到的 top3 分数作为分界，
    与实时 top3(abs(resp_i / 1ppm阈值_i)) 分数比较划分酒精等级。
    """
    if th is None:
        return "--"
    score10 = float(th.get("score10", th.get("r10", float("nan"))))
    score50 = float(th.get("score50", th.get("r50", float("nan"))))
    score100 = float(th.get("score100", th.get("r100", float("nan"))))
    if not (
        np.isfinite(score)
        and np.isfinite(score10)
        and np.isfinite(score50)
        and np.isfinite(score100)
    ):
        return "--"
    decreasing = (score10 > score50) and (score50 > score100)
    increasing = (score10 < score50) and (score50 < score100)
    if decreasing:
        if score > score10:
            return "微量酒精(0-10ppm)"
        if score > score50:
            return "少量酒精(10-50ppm)"
        if score > score100:
            return "疑似酒驾(50-100ppm)"
        return "疑似醉驾(>100ppm)"
    if increasing:
        if score < score10:
            return "微量酒精(0-10ppm)"
        if score < score50:
            return "少量酒精(10-50ppm)"
        if score < score100:
            return "疑似酒驾(50-100ppm)"
        return "疑似醉驾(>100ppm)"
    return "--"

def normalize_odor_level_output(level):
    level_text = str(level or "").strip()
    if not level_text or level_text in ("--", "None"):
        return ""
    return ODOR_LEVEL_ENGLISH_MAP.get(level_text, level_text)

def odor_level_for_display(level):
    level_text = str(level or "").strip()
    if not level_text or level_text in ("--", "None"):
        return "--"
    return ODOR_LEVEL_CHINESE_MAP.get(level_text, level_text)

def load_baseline_sigmoid_resp_params(device_id_text, x_cap_conc=None):
    """
    从 Conc_Cal.xlsx 中读取指定浓度的 resp_Sensor1..16，默认 Conc==1。
    每路空气门控阈值 x_cap = mean(|resp@Conc|)；缺失时回退 CONC2_RESP_GATE_X_CAP。
    """
    x_caps = list(CONC2_RESP_GATE_X_CAP)
    target_conc = (
        float(RESP_X_CAP_CAL_CONC) if x_cap_conc is None else float(x_cap_conc)
    )
    raw = (device_id_text or "").strip()
    if not raw:
        return {
            "x_cap": x_caps,
            "x_cap_conc": target_conc,
            "path": None,
            "loaded": False,
        }

    safe = sanitize_path_segment(raw, "default")
    path = conc_calibration_xlsx_path(safe)
    if not path:
        return {
            "x_cap": x_caps,
            "x_cap_conc": target_conc,
            "path": None,
            "loaded": False,
        }

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {
            "x_cap": x_caps,
            "x_cap_conc": target_conc,
            "path": path,
            "loaded": False,
        }

    cache_key = (os.path.abspath(path), mtime, target_conc)
    if cache_key in _BASELINE_SIGMOID_RESP_PARAMS_CACHE:
        return _BASELINE_SIGMOID_RESP_PARAMS_CACHE[cache_key]

    try:
        df = pd.read_excel(path)
    except Exception:
        out = {
            "x_cap": x_caps,
            "x_cap_conc": target_conc,
            "path": path,
            "loaded": False,
        }
        _BASELINE_SIGMOID_RESP_PARAMS_CACHE[cache_key] = out
        return out

    if "Conc" not in df.columns:
        out = {
            "x_cap": x_caps,
            "x_cap_conc": target_conc,
            "path": path,
            "loaded": False,
        }
        _BASELINE_SIGMOID_RESP_PARAMS_CACHE[cache_key] = out
        return out

    conc = np.asarray(pd.to_numeric(df["Conc"], errors="coerce"), dtype=float)
    rows = df[
        np.isclose(conc, target_conc, rtol=0.0, atol=1e-9)
    ]
    if rows.empty:
        out = {
            "x_cap": x_caps,
            "x_cap_conc": target_conc,
            "path": path,
            "loaded": False,
        }
        _BASELINE_SIGMOID_RESP_PARAMS_CACHE[cache_key] = out
        return out

    loaded_any = False
    for idx, column in enumerate(RESP_COLUMNS):
        if column not in rows.columns:
            continue
        values = np.asarray(pd.to_numeric(rows[column], errors="coerce"), dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        x_cap = float(np.mean(np.abs(values)))
        if not np.isfinite(x_cap) or x_cap <= 0.0:
            continue
        x_caps[idx] = x_cap
        loaded_any = True

    out = {
        "x_cap": x_caps,
        "x_cap_conc": target_conc,
        "path": path,
        "loaded": loaded_any,
    }
    _BASELINE_SIGMOID_RESP_PARAMS_CACHE[cache_key] = out
    return out

def conc2_resp_x_caps(device_id_text):
    """每路 Conc=1 门控阈值 x_cap 列表。"""
    params = load_baseline_sigmoid_resp_params(device_id_text or "")
    return params.get("x_cap") or CONC2_RESP_GATE_X_CAP

def response_strength_score_from_resp_values(resp_values, device_id_text, top_k=RESPONSE_STRENGTH_TOP_K):
    """按 1ppm 空气门控阈值归一化，取 top-k 通道平均值作为非酒精气味响应强度。"""
    if resp_values is None:
        return float("nan")
    try:
        values = np.asarray(list(resp_values), dtype=float)
    except (TypeError, ValueError):
        return float("nan")
    if values.size == 0:
        return float("nan")

    thresholds = np.asarray(conc2_resp_x_caps(device_id_text), dtype=float)
    usable = min(values.size, thresholds.size)
    if usable <= 0:
        return float("nan")

    values = values[:usable]
    thresholds = thresholds[:usable]
    valid = np.isfinite(values) & np.isfinite(thresholds) & (thresholds > 0.0)
    if not np.any(valid):
        return float("nan")

    ratios = np.abs(values[valid]) / thresholds[valid]
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return float("nan")

    k = min(max(1, int(top_k)), ratios.size)
    return float(np.mean(np.sort(ratios)[-k:]))

def resp_gate_score_from_resp_values(resp_values, device_id_text):
    """Air gate score: mean(top3(abs(resp_i) / resp_1ppm_i))."""
    return response_strength_score_from_resp_values(
        resp_values, device_id_text, top_k=INFERENCE_RESP_GATE_TOP_K
    )

def should_run_recognition_inference(resp_values, device_id_text):
    """Run recognition only when the top-k normalized response score reaches the gate."""
    score = resp_gate_score_from_resp_values(resp_values, device_id_text)
    try:
        threshold = float(INFERENCE_RESP_GATE_SCORE_THRESHOLD)
    except (TypeError, ValueError):
        threshold = 1.0
    return bool(np.isfinite(score) and np.isfinite(threshold) and score >= threshold)

def build_top3_resp_gate_status_fields(resp_values, device_id_text):
    """Air gate status fields based on mean(top3(abs(resp_i) / resp_1ppm_i))."""
    score = resp_gate_score_from_resp_values(resp_values, device_id_text)
    return {
        "resp_gate_score": float(score) if np.isfinite(score) else float("nan"),
        "resp_gate_score_threshold": float(INFERENCE_RESP_GATE_SCORE_THRESHOLD),
        "resp_gate_top_k": int(INFERENCE_RESP_GATE_TOP_K),
    }

def response_strength_level_from_ethanol_equiv_thresholds(predicted_label, score, th):
    """
    第二类别强度等级：使用本设备乙醇标气 10/50/100 ppm 对应的响应强度作为参考。
    该等级表示乙醇等效响应强度，不代表这些气味的真实 ppm 浓度。
    """
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(score):
        return "--"
    if th is None:
        return "--"
    try:
        score10 = float(th.get("score10", th.get("r10")))
        score50 = float(th.get("score50", th.get("r50")))
        score100 = float(th.get("score100", th.get("r100")))
    except (AttributeError, TypeError, ValueError):
        return "--"
    if not (
        np.isfinite(score10)
        and np.isfinite(score50)
        and np.isfinite(score100)
    ):
        return "--"

    label = str(predicted_label or "").strip()
    if label == "香烟":
        kind = "烟味"
    #elif label == ABNORMAL_ODOR_LABEL:
        #kind = "异味"
    elif label == "香烟/异味":
        kind = "烟味/异味"
    else:
        return "--"

    increasing = (score10 < score50) and (score50 < score100)
    decreasing = (score10 > score50) and (score50 > score100)
    if increasing:
        if score < score10:
            return f"微弱{kind}"
        if score < score50:
            return f"中等{kind}"
        return f"强烈{kind}"
    if decreasing:
        if score > score10:
            return f"微弱{kind}"
        if score > score50:
            return f"中等{kind}"
        return f"强烈{kind}"
    return "--"

def response_strength_threshold_entries(device_id_text, class_labels=None):
    """按当前二分类任务生成 score_thresholds 列的阈值说明。"""
    th = load_conc_calibration_resp_thresholds(device_id_text) if device_id_text else None
    if th is None:
        return []
    try:
        score10 = float(th.get("score10", th.get("r10")))
        score50 = float(th.get("score50", th.get("r50")))
        score100 = float(th.get("score100", th.get("r100")))
    except (AttributeError, TypeError, ValueError):
        return []
    if not (
        np.isfinite(score10)
        and np.isfinite(score50)
        and np.isfinite(score100)
    ):
        return []
    if score10 < score50 < score100:
        direction = "increasing"
    elif score10 > score50 > score100:
        direction = "decreasing"
    else:
        direction = "non_monotonic"

    suffix = f"metric=top3_resp_over_{RESP_X_CAP_CAL_CONC:g}ppm_gate; curve={direction}"
    entries = [
        "酒精",
        f"微量酒精(0-10ppm): score10={score10:.6g}; {suffix}",
        f"少量酒精(10-50ppm): score50={score50:.6g}; {suffix}",
        f"疑似酒驾(50-100ppm): score100={score100:.6g}; {suffix}",
        f"疑似醉驾(>100ppm): score100={score100:.6g}; {suffix}",
    ]
    labels = tuple(class_labels or ("酒精", "香烟"))
    second_label = next((label for label in labels if label != "酒精"), None)
    if second_label:
        level_name = "烟味" if second_label == "香烟" else second_label
        entries.extend(
            [
                second_label,
                f"微弱{level_name}: score10={score10:.6g}; 乙醇等效响应强度; {suffix}",
                f"中等{level_name}: score50={score50:.6g}; 乙醇等效响应强度; {suffix}",
                f"强烈{level_name}: score100={score100:.6g}; 乙醇等效响应强度; {suffix}",
            ]
        )
    return entries

def _sensor_abs_resp_above_conc2(resp, threshold):
    return (
        np.isfinite(resp)
        and np.isfinite(threshold)
        and float(threshold) > 0.0
        and abs(float(resp)) > float(threshold)
    )

def select_peak_track_sensor_index(resp_values, device_id_text):
    """
    在已超过 Conc=1 阈值的通道中，选取 |resp| / x_cap 最大者作为本轮寻峰锁定通道。
    若无超阈通道（不应与 should_run_recognition 同时为真），回退为全局归一化响应最大通道。
    """
    x_caps = conc2_resp_x_caps(device_id_text)
    best_idx = None
    best_ratio = -1.0
    best_abs = -1.0
    for idx, resp in enumerate(resp_values):
        if idx >= len(x_caps):
            break
        threshold = float(x_caps[idx])
        if not _sensor_abs_resp_above_conc2(resp, threshold):
            continue
        abs_resp = abs(float(resp))
        ratio = abs_resp / threshold
        if ratio > best_ratio or (np.isclose(ratio, best_ratio) and abs_resp > best_abs):
            best_ratio = ratio
            best_abs = abs_resp
            best_idx = idx
    if best_idx is not None:
        return best_idx
    for idx, resp in enumerate(resp_values):
        if not np.isfinite(resp):
            continue
        abs_resp = abs(float(resp))
        try:
            threshold = float(x_caps[idx])
        except (TypeError, ValueError, IndexError):
            threshold = float("nan")
        ratio = abs_resp / threshold if np.isfinite(threshold) and threshold > 0.0 else abs_resp
        if ratio > best_ratio or (np.isclose(ratio, best_ratio) and abs_resp > best_abs):
            best_ratio = ratio
            best_abs = abs_resp
            best_idx = idx
    return best_idx

def peak_track_signal_from_resp(resp_values, track_idx):
    """锁定通道上的 |resp|，用于寻峰波形（避免 max 跨通道跳变）。"""
    if track_idx is None or track_idx < 0 or track_idx >= len(resp_values):
        return 0.0
    resp = float(resp_values[track_idx])
    return abs(resp) if np.isfinite(resp) else 0.0

def peak_track_sensor_display_name(track_idx):
    if track_idx is None or track_idx < 0:
        return "--"
    if track_idx < len(RESP_COLUMNS):
        name = str(RESP_COLUMNS[track_idx])
        if name.startswith("resp_"):
            return name.replace("resp_", "", 1)
        return name
    return f"Sensor{track_idx + 1}"

def compute_baseline_alpha_sigmoid(x, alpha_hi, alpha_lo, k, x_c, x_cap):
    """
    α(x) = α_lo + (α_hi - α_lo) / (1 + exp(k*(x - x_c))) ，0 <= x < x_cap；
    x >= x_cap 时返回 α_lo。x 为相对残差 |(R-baseline)/baseline|。
    """
    x = float(x)
    if not np.isfinite(x) or x < 0.0:
        return float(alpha_lo)
    if x >= float(x_cap):
        return float(alpha_lo)
    z = float(k) * (x - float(x_c))
    if z > 50.0:
        return float(alpha_lo)
    if z < -50.0:
        return float(alpha_hi)
    return float(alpha_lo + (alpha_hi - alpha_lo) / (1.0 + np.exp(z)))
