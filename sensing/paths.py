# -*- coding: utf-8 -*-
"""
sensing/paths.py

标定/补偿文件的路径解析，calibration.py 和 thermal_compensation.py 共用。
不依赖任何 GUI 框架，源码运行和打包后运行（PyInstaller onefile）都要能用。

统一目录约定（跟 v3_6 保持一致）：
    data/calibration/<category>/<device_segment_sanitized>/<filename>
例如：
    data/calibration/Conc_Cal/回流1/Conc_Cal.xlsx
    data/calibration/th_compensation/回流1/comp_params.xlsx
"""

import os
import sys


def get_app_dir():
    """可写根目录：源码运行为 enose_app 项目根目录，打包后为可执行文件所在目录。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    # 本文件位于 enose_app/sensing/paths.py，上跳两级即项目根目录。
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_device_calibration_file(category, device_segment_sanitized, filename):
    """
    按设备号解析标定/补偿文件的实际路径，找不到返回 None。

    源码运行：从 enose_app 项目根目录下的 data/... 找。
    PyInstaller onefile 打包后：优先找可执行文件同目录下的 data/...，
    找不到再找 _MEIPASS（如果用 --add-data 把标定文件打进了包体）下的
    data/...，探测顺序跟 v3_6 的 conc_calibration_xlsx_path 保持一致。
    """
    if not device_segment_sanitized:
        return None
    rel = os.path.join("data", "calibration", category, device_segment_sanitized, filename)
    candidates = [os.path.join(get_app_dir(), rel)]
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(os.path.join(meipass, rel))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def ensure_device_calibration_dir(category, device_segment_sanitized):
    """确保标定文件所在目录存在，返回目录路径。

    enose_app 运行时通常只读标定文件，这个函数主要是为以后如果需要在
    enose_app 里写入/更新标定文件时用；纯推理场景一般用不到，先留着。
    """
    base = get_app_dir()
    seg = device_segment_sanitized or "default"
    directory = os.path.join(base, "data", "calibration", category, seg)
    os.makedirs(directory, exist_ok=True)
    return directory
