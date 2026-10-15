# -*- coding: utf-8 -*-
"""Independent realtime feature-curve inspection window."""

from runtime_support import *
from gui.plot_widgets import _apply_standard_plot_axes


class FeatureWindow(QtWidgets.QDialog):
    """特征显示窗口：可切换查看 resp 或 group_zscore_resp 曲线。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("特征曲线窗口")
        self.resize(1000, 600)

        self.xdata = []
        self.resp_ydata = [[] for _ in range(16)]
        self.group_zscore_resp_ydata = [[] for _ in range(16)]
        self.feature_curves = []
        self.group_colors = [
            (220, 180, 40),   # Sensor1-4 黄色
            (60, 170, 80),    # Sensor5-8 绿色
            (70, 130, 220),   # Sensor9-12 蓝色
            (200, 90, 170),   # Sensor13-16 紫色
        ]

        layout = QtWidgets.QVBoxLayout(self)

        top_layout = QtWidgets.QHBoxLayout()
        mode_label = QtWidgets.QLabel("显示特征:")
        mode_label.setStyleSheet("font: 11pt \"楷体\";")
        self.mode_combo = QtWidgets.QComboBox(self)
        self.mode_combo.addItems(["resp", "group_zscore_resp_Sensor"])
        self.mode_combo.setStyleSheet("font: 11pt \"楷体\";")
        top_layout.addWidget(mode_label)
        top_layout.addWidget(self.mode_combo)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        self.plot_widget = pg.PlotWidget(self)
        self.plot_widget.setBackground("w")
        _apply_standard_plot_axes(self.plot_widget)
        self.plot_widget.setLabel("left", "Feature Value", color="black")
        self.plot_widget.setLabel("bottom", "Time", color="black")
        layout.addWidget(self.plot_widget)

        for i in range(16):
            group_color = self.group_colors[i // 4]
            curve = self.plot_widget.plot(pen=pg.mkPen(color=group_color, width=2))
            self.feature_curves.append(curve)

        tighten_plot_layout(self.plot_widget, left_axis_width=78, bottom_axis_height=40)
        self.mode_combo.currentTextChanged.connect(self.refresh_plot)

    def set_feature_data(self, xdata, resp_ydata, group_zscore_resp_ydata):
        """接收主窗口传来的特征时间序列，并刷新绘图。"""
        self.xdata = list(xdata)[-MAX_POINTS:]
        self.resp_ydata = [list(series)[-MAX_POINTS:] for series in resp_ydata]
        self.group_zscore_resp_ydata = [
            list(series)[-MAX_POINTS:] for series in group_zscore_resp_ydata
        ]
        self.refresh_plot()

    def refresh_plot(self):
        """根据当前下拉框选择，显示 resp 或 group_zscore_resp 曲线。"""
        selected_mode = self.mode_combo.currentText()
        selected_data = (
            self.resp_ydata
            if selected_mode == "resp"
            else self.group_zscore_resp_ydata
        )

        for i, curve in enumerate(self.feature_curves):
            curve.setData(self.xdata, selected_data[i])

        self.plot_widget.setTitle(f"{selected_mode} 曲线", color="black")

    def clear_data(self):
        """清空窗口中的所有特征曲线。"""
        self.xdata = []
        self.resp_ydata = [[] for _ in range(16)]
        self.group_zscore_resp_ydata = [[] for _ in range(16)]
        self.refresh_plot()
