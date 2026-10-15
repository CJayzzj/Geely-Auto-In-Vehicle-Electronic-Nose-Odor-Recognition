# -*- coding: utf-8 -*-
"""
sensing/resistance.py

ADC 转电阻，是整条数据处理链路的第一步（排在温湿度补偿之前）：
    串口帧 {adc, temp, humid} → adc_to_resistance() → apply_th_comp_resistances() → ...

公式和边界处理原样保留自 v3_6 的 adc_to_resistance 方法，只是从类方法
改成了独立函数（不依赖任何 UI 状态）。
"""

import math
from config.constants import MATCHED_RESISTANCE


def adc_to_resistance(adc_value, matched_resistance=MATCHED_RESISTANCE):
    """按公式 R = matched_resistance * ADC / (4096 - ADC) 将 ADC 转为电阻值。

    ADC 无效（非数字/NaN/inf）、ADC >= 4096、或分母 <= 0 时返回
    float("inf")，跟 v3_6 原实现（np.isfinite 判断）的边界处理完全一致。
    """
    try:
        adc_value = float(adc_value)
    except (TypeError, ValueError):
        return float("inf")
    if not math.isfinite(adc_value) or adc_value >= 4096:
        return float("inf")
    denominator = 4096.0 - adc_value
    if denominator <= 0.0:
        return float("inf")
    return float(matched_resistance) * adc_value / denominator


def adc_values_to_resistances(adc_values, matched_resistance=MATCHED_RESISTANCE):
    """对 16 路 ADC 值批量转换，方便主循环一行调用。"""
    return [adc_to_resistance(v, matched_resistance) for v in adc_values]

