# -*- coding: utf-8 -*-
"""Plot axes and curve-color helpers moved out of the main window."""

from runtime_support import *


def sensor_raw_curve_rgb(sensor_idx):
    """R_raw 主曲线 pen 的 RGB，与 sensor_idx 0..15 对应。"""
    return (sensor_idx * 15 % 255, 100, 200)

class _AxisTickTwoDecimals(pg.AxisItem):
    """Y 轴刻度保留两位小数。"""

    def tickStrings(self, values, scale, spacing):
        return [f"{(v * scale):.2f}" for v in values]

class _AxisTickIntegerX(pg.AxisItem):
    """X 轴刻度显示为整数。"""

    def tickStrings(self, values, scale, spacing):
        return [str(int(round(v * scale))) for v in values]

def _apply_standard_plot_axes(plot_widget):
    plot_widget.plotItem.setAxisItems(
        {
            "left": _AxisTickTwoDecimals(orientation="left"),
            "bottom": _AxisTickIntegerX(orientation="bottom"),
        }
    )
    # 替换轴后必须把网格绑到当前轴上（若先 showGrid 再替换轴，或复选框已为 True，则可能不触发 updateGrid）。
    plot_widget.showGrid(x=True, y=True)
    plot_widget.plotItem.updateGrid()
