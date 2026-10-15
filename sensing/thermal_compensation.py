# -*- coding: utf-8 -*-
"""
sensing/thermal_compensation.py

温湿度补偿：只保留 enose_app 运行时需要的纯函数部分，从项目原有的
th_compensation.py（内部温湿度补偿系数标定工具）里精简出来。

【原文件里以下内容不在这里，也不需要迁移】：
    - 所有 PyQt5 / pyqtgraph 相关的界面代码（标定点采集界面、曲线绘制、
      文件对话框等，原文件里 CAL_SHEET_TEMP 常量往下的绝大部分内容）
    - fit_poly2()：用标定点拟合出补偿系数的多项式拟合函数
    - save_th_comp_coeffs_xlsx()：保存系数用的（制作标定文件时用）
    - save_calibration_samples_xlsx() / load_calibration_samples_xlsx()：
      保存/加载"标定过程中间数据"用的
    以上都是"怎么做出补偿系数表"的职责，属于标定制作工具，跟 v3_6 一样是
    内部工具，不是 enose_app 交付物的一部分。如果以后需要单独维护一个
    标定制作工具，直接用原 th_compensation.py 即可，不需要跟这个精简版
    保持同步——两者服务于不同阶段（制作标定文件 vs. 运行时应用标定文件）。

【enose_app 只需要】：
    1. 启动时用 load_th_comp_coeffs_for_device() 读一次系数（找不到标定
       文件或读取失败都会自动回退到 default_th_comp_coeffs()，不能因为
       标定文件缺失/损坏就崩溃退出——enose_app 是无人值守的终端服务）；
    2. 每帧数据用 apply_th_comp_resistances() 对 16 路电阻做补偿。

    补偿后的电阻值才是后续 baseline 更新 / resp 特征计算 / 门控判定 /
    模型推理全部计算的基准，这个顺序在 CALIBRATION_BRIEF_AND_MAIN_REVIEW.md
    第四节有详细说明，接入主循环时不能打乱这个顺序。
"""

import numpy as np
import pandas as pd

from sensing.paths import resolve_device_calibration_file

# 标定基准（℃ / %RH），跟原始 th_compensation.py 保持一致。
TH_COMP_REF_TEMP_C = 20.0
TH_COMP_REF_HUMI_PCT = 50.0

# 设备目录 data/calibration/th_compensation/<device>/ 下的系数文件名。
TH_COMP_COEFF_FILENAME = "comp_params.xlsx"

# resolve_device_calibration_file 的 category 参数，跟 calibration.py 里
# Conc_Cal 那个类别是同一套目录规则，只是类别名不同。
TH_COMP_CALIBRATION_CATEGORY = "th_compensation"


def default_th_comp_coeffs():
    """未标定 / 标定文件缺失时的兜底系数：全零，应用后与不补偿完全等价。"""
    zeros = [0.0] * 16
    return {
        "ref_temp_c": TH_COMP_REF_TEMP_C,
        "ref_humi_pct": TH_COMP_REF_HUMI_PCT,
        "At": zeros[:],
        "Bt": zeros[:],
        "Ct": zeros[:],
        "Ah": zeros[:],
        "Bh": zeros[:],
        "Ch": zeros[:],
    }


def load_th_comp_coeffs_xlsx(path):
    """从 xlsx 读取补偿系数（原样保留 th_compensation.py 里的解析逻辑）。

    读取失败（文件不存在 / 缺列 / 行数不对）会抛异常，调用方必须自己
    catch 住并回退到 default_th_comp_coeffs()——本函数本身不做兜底，
    需要兜底行为请用下面的 load_th_comp_coeffs_for_device()。

    Returns:
        (coeffs, apply_plot)：apply_plot 是原工具里"是否显示补偿曲线"的
        标志位，enose_app 没有界面用不上，保留只是为了函数签名跟原版
        对得上，方便以后对照排查问题。
    """
    coeffs = default_th_comp_coeffs()
    df_c = pd.read_excel(path, sheet_name="coeffs")
    need = ["SensorID", "At", "Bt", "Ct", "Ah", "Bh", "Ch"]
    for col in need:
        if col not in df_c.columns:
            raise ValueError(f"工作表 coeffs 缺少列: {col}")
    df_c = df_c.sort_values("SensorID").reset_index(drop=True)
    if len(df_c) != 16:
        raise ValueError(f"coeffs 应为 16 行，实际 {len(df_c)}")
    for key in ("At", "Bt", "Ct", "Ah", "Bh", "Ch"):
        coeffs[key] = [float(x) for x in df_c[key].tolist()]

    apply_plot = False
    try:
        df_m = pd.read_excel(path, sheet_name="meta")
        if len(df_m) >= 1:
            row = df_m.iloc[0]
            if "ref_temp_c" in df_m.columns and pd.notna(row["ref_temp_c"]):
                coeffs["ref_temp_c"] = float(row["ref_temp_c"])
            if "ref_humi_pct" in df_m.columns and pd.notna(row["ref_humi_pct"]):
                coeffs["ref_humi_pct"] = float(row["ref_humi_pct"])
            if "show_comp_curve" in df_m.columns and pd.notna(row["show_comp_curve"]):
                apply_plot = bool(row["show_comp_curve"])
    except Exception:
        pass
    return coeffs, apply_plot


def load_th_comp_coeffs_for_device(device_segment_sanitized):
    """路径解析 + 读取 + 兜底，一步到位，main.py 启动时直接调用这个即可。

    找不到标定文件、或者文件存在但读取失败，都会自动回退到
    default_th_comp_coeffs()，同时返回 used_default=True，方便调用方在
    启动日志里提示"当前设备没有可用的温湿度补偿标定，正在使用默认零系数"，
    而不是让程序直接崩溃退出。

    Returns:
        (coeffs, used_default)
    """
    path = resolve_device_calibration_file(
        TH_COMP_CALIBRATION_CATEGORY, device_segment_sanitized, TH_COMP_COEFF_FILENAME
    )
    if not path:
        return default_th_comp_coeffs(), True
    try:
        coeffs, _apply_plot = load_th_comp_coeffs_xlsx(path)
        return coeffs, False
    except Exception:
        return default_th_comp_coeffs(), True


def apply_th_comp_resistances(resistances_16, temp_c, humi_pct, coeffs):
    """
    对 16 路电阻做温湿度补偿（对数域加性修正）：
        log10(R') = log10(R) + f_T(T) + f_H(H)
    系数全零时 R' 与 R 完全相同（未标定/读取失败兜底后就是这种情况）。

    这是 enose_app 每帧数据都会调用的核心函数，调用顺序见模块顶部文档：
    必须排在 adc_to_resistance() 之后、baseline 更新和 resp 特征计算
    之前，不能打乱。
    """
    if coeffs is None:
        return list(resistances_16)
    r = np.asarray(resistances_16, dtype=float)
    t = float(temp_c)
    h = float(humi_pct)
    at = np.asarray(coeffs["At"], dtype=float)
    bt = np.asarray(coeffs["Bt"], dtype=float)
    ct = np.asarray(coeffs["Ct"], dtype=float)
    ah = np.asarray(coeffs["Ah"], dtype=float)
    bh = np.asarray(coeffs["Bh"], dtype=float)
    ch = np.asarray(coeffs["Ch"], dtype=float)
    dlog = ((at * t + bt) * t + ct) + ((ah * h + bh) * h + ch)
    out = r.copy()
    valid = np.isfinite(r) & (r > 0) & (r < np.inf)
    if valid.any():
        safe = np.clip(r[valid], 1e-30, None)
        out[valid] = 10.0 ** (np.log10(safe) + dlog[valid])
    return out.tolist()
