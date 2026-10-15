# -*- coding: utf-8 -*-
"""Runtime paths shared by both tools.

Source mode writes below the project directory. A packaged executable writes
beside that executable, so the two final tools naturally own independent data.
"""

import os
import socket
import sys
from pathlib import Path


def get_app_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return str(Path(__file__).resolve().parents[1])


def get_lan_ips():
    ips = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            ip = probe.getsockname()[0]
            if ip and not ip.startswith("127.") and ip != "0.0.0.0":
                ips.append(ip)
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip != "0.0.0.0" and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    return ips


def sanitize_path_segment(text, fallback="default"):
    value = str(text or "").strip()
    if not value:
        return fallback
    invalid = '<>:"/\\|?*\x00'
    safe = "".join("_" if char in invalid or char == " " else char for char in value)
    safe = safe.strip("._")
    return (safe or fallback)[:80]


APP_ROOT = get_app_dir()
DATA_DIR = os.path.join(APP_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
MODELS_DIR = os.path.join(APP_ROOT, "models")
CONFIG_DIR = os.path.join(DATA_DIR, "config")

# Calibration is shipped once and shared by both source entry points. In the
# final independent builds each executable receives its own calibration folder.
CALIBRATION_DIR = os.path.join(APP_ROOT, "calibration")
CONC_CAL_DIR = os.path.join(CALIBRATION_DIR, "Conc_Cal")
TH_COMP_DIR = os.path.join(CALIBRATION_DIR, "th_compensation")


def ensure_runtime_dirs():
    for directory in (
        RAW_DATA_DIR,
        DATASETS_DIR,
        MODELS_DIR,
        CONFIG_DIR,
        CONC_CAL_DIR,
        TH_COMP_DIR,
    ):
        os.makedirs(directory, exist_ok=True)


def raw_data_file_path(device_id, label, timestamp, suffix=""):
    """Return data/raw/<device>/<YYYYMMDD>/<label>_<timestamp><suffix>.csv."""
    device = sanitize_path_segment(device_id)
    label_part = sanitize_path_segment(label, "unlabeled")
    day = timestamp.strftime("%Y%m%d")
    stamp = timestamp.strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(RAW_DATA_DIR, device, day)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "{}_{}{}.csv".format(label_part, stamp, suffix))


def device_dataset_dir(device_id):
    path = os.path.join(DATASETS_DIR, sanitize_path_segment(device_id))
    os.makedirs(path, exist_ok=True)
    return path


def device_model_dir(device_id):
    path = os.path.join(MODELS_DIR, sanitize_path_segment(device_id))
    os.makedirs(path, exist_ok=True)
    return path
