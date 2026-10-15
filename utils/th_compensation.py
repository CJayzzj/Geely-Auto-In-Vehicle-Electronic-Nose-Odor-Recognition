# -*- coding: utf-8 -*-
"""温湿度补偿：标定点采集、可视化、参数计算与应用（电阻直接取对数，无一致化映射）。"""
import os
import time

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtWidgets

from utils.paths import TH_COMP_DIR

# 标定点与基准（℃ / %RH）
TH_COMP_TEMP_POINTS = (20.0, 35.0, 50.0)
TH_COMP_HUMI_POINTS = (20.0, 50.0, 80.0)
TH_COMP_REF_TEMP_C = 20.0
TH_COMP_REF_HUMI_PCT = 50.0
# 标定数据表仅展示代表性通道（0-based 索引对应 R1,R5,R9,R13）
TH_COMP_TABLE_R_LABELS = ("R1", "R5", "R9", "R13")
TH_COMP_TABLE_R_IDX = (0, 4, 8, 12)
# 设备目录 th_compensation/<device>/ 下默认文件名（系数与标定原始数据同目录）
TH_COMP_COEFF_FILENAME = "comp_params.xlsx"


class _AxisTickTwoDecimals(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [f"{(v * scale):.2f}" for v in values]


class _AxisTickIntegerX(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [str(int(round(v * scale))) for v in values]


def _apply_standard_plot_axes(plot_widget):
    plot_widget.plotItem.setAxisItems(
        {
            "left": _AxisTickTwoDecimals(orientation="left"),
            "bottom": _AxisTickIntegerX(orientation="bottom"),
        }
    )
    plot_widget.showGrid(x=True, y=True)
    plot_widget.plotItem.updateGrid()


def tighten_plot_layout(plot_widget, left_axis_width=56, bottom_axis_height=40):
    """
    压缩 PlotItem 内部 layout 与坐标轴占位，让 ViewBox 曲线区尽量占满控件。
    须在 setLabel / plot 等配置好轴之后再调用。
    """
    plot_widget.setContentsMargins(0, 0, 0, 0)
    pi = plot_widget.plotItem
    lay = getattr(pi, "layout", None)
    if lay is not None:
        lay.setContentsMargins(0, 0, 0, 0)
        try:
            lay.setSpacing(0)
        except Exception:
            pass
    left = plot_widget.getAxis("left")
    bottom = plot_widget.getAxis("bottom")
    if left is not None:
        left.setWidth(left_axis_width)
        try:
            left.setStyle(autoExpandTextSpace=False)
        except Exception:
            pass
    if bottom is not None:
        bottom.setHeight(bottom_axis_height)
        try:
            bottom.setStyle(autoExpandTextSpace=False)
        except Exception:
            pass
    try:
        pi.hideButtons()
    except Exception:
        pass


def fit_poly2(x, y):
    """二次多项式 y ≈ a x^2 + b x + c（与 JF_T_H_Comp 一致）。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    xx = x[mask]
    yy = y[mask]
    X = np.column_stack([xx ** 2, xx, np.ones(len(xx))])
    try:
        params, *_ = np.linalg.lstsq(X, yy, rcond=None)
        return float(params[0]), float(params[1]), float(params[2])
    except Exception:
        return np.nan, np.nan, np.nan


def default_th_comp_coeffs():
    """未标定 / 全零：应用后与无补偿等价。"""
    z = [0.0] * 16
    return {
        "ref_temp_c": TH_COMP_REF_TEMP_C,
        "ref_humi_pct": TH_COMP_REF_HUMI_PCT,
        "At": z[:],
        "Bt": z[:],
        "Ct": z[:],
        "Ah": z[:],
        "Bh": z[:],
        "Ch": z[:],
    }


def apply_th_comp_resistances(resistances_16, temp_c, humi_pct, coeffs, dlog_beta=1.0):
    """
    对 16 路电阻做温湿度补偿（对数域加性修正）：
    log10(R') = log10(R) + f_T(T) + f_H(H)，无映射归一化。
    系数全零时 R' 与 R 相同（用于特征/数据集/推理始终走此路径）。
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
    # Scale the fitted correction: log10(R_comp) = log10(R_raw) + beta * dlog.
    try:
        beta = float(dlog_beta)
    except Exception:
        beta = 1.0
    if not np.isfinite(beta):
        beta = 1.0
    dlog = beta * (((at * t + bt) * t + ct) + ((ah * h + bh) * h + ch))
    out = r.copy()
    valid = np.isfinite(r) & (r > 0) & (r < np.inf)
    if valid.any():
        safe = np.clip(r[valid], 1e-30, None)
        out[valid] = 10.0 ** (np.log10(safe) + dlog[valid])
    return out.tolist()


def save_th_comp_coeffs_xlsx(path, coeffs, apply_plot: bool):
    """保存补偿系数与元数据到 xlsx（coeffs / meta 两个工作表）。"""
    rows = []
    for j in range(16):
        rows.append(
            {
                "SensorID": j + 1,
                "At": coeffs["At"][j],
                "Bt": coeffs["Bt"][j],
                "Ct": coeffs["Ct"][j],
                "Ah": coeffs["Ah"][j],
                "Bh": coeffs["Bh"][j],
                "Ch": coeffs["Ch"][j],
            }
        )
    df_c = pd.DataFrame(rows)
    df_m = pd.DataFrame(
        [
            {
                "ref_temp_c": coeffs["ref_temp_c"],
                "ref_humi_pct": coeffs["ref_humi_pct"],
                "show_comp_curve": bool(apply_plot),
            }
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df_c.to_excel(w, sheet_name="coeffs", index=False)
        df_m.to_excel(w, sheet_name="meta", index=False)


def load_th_comp_coeffs_xlsx(path):
    """从 xlsx 读取系数与「是否显示补偿曲线」标志；缺失字段用默认。"""
    c = default_th_comp_coeffs()
    df_c = pd.read_excel(path, sheet_name="coeffs")
    need = ["SensorID", "At", "Bt", "Ct", "Ah", "Bh", "Ch"]
    for col in need:
        if col not in df_c.columns:
            raise ValueError(f"工作表 coeffs 缺少列: {col}")
    df_c = df_c.sort_values("SensorID").reset_index(drop=True)
    if len(df_c) != 16:
        raise ValueError(f"coeffs 应为 16 行，实际 {len(df_c)}")
    for key in ("At", "Bt", "Ct", "Ah", "Bh", "Ch"):
        c[key] = [float(x) for x in df_c[key].tolist()]
    apply_plot = False
    try:
        df_m = pd.read_excel(path, sheet_name="meta")
        if len(df_m) >= 1:
            row = df_m.iloc[0]
            if "ref_temp_c" in df_m.columns and pd.notna(row["ref_temp_c"]):
                c["ref_temp_c"] = float(row["ref_temp_c"])
            if "ref_humi_pct" in df_m.columns and pd.notna(row["ref_humi_pct"]):
                c["ref_humi_pct"] = float(row["ref_humi_pct"])
            if "show_comp_curve" in df_m.columns and pd.notna(row["show_comp_curve"]):
                apply_plot = bool(row["show_comp_curve"])
    except Exception:
        pass
    return c, apply_plot


CAL_SHEET_TEMP = "temp_calibration"
CAL_SHEET_HUMI = "humi_calibration"


def _match_calibration_setpoint(value, targets, tol=0.05):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    for x in targets:
        if abs(v - float(x)) <= tol:
            return float(x)
    return None


def save_calibration_samples_xlsx(path, temp_samples, humi_samples):
    """
    将温度/湿度标定点内存数据写入 xlsx。
    工作表 temp_calibration / humi_calibration：每行一个标定点，含 16 路 R 与记录温湿度。
    """
    rows_t = []
    for t in TH_COMP_TEMP_POINTS:
        s = temp_samples.get(t)
        d = {"setpoint_C": float(t)}
        if s is not None:
            d["T_recorded"] = float(s["T"])
            d["H_recorded"] = float(s["H"])
            for j in range(16):
                d[f"R{j + 1}"] = float(s["R"][j])
        else:
            d["T_recorded"] = np.nan
            d["H_recorded"] = np.nan
            for j in range(16):
                d[f"R{j + 1}"] = np.nan
        rows_t.append(d)
    df_t = pd.DataFrame(rows_t)

    rows_h = []
    for h in TH_COMP_HUMI_POINTS:
        s = humi_samples.get(h)
        d = {"setpoint_RH_pct": float(h)}
        if s is not None:
            d["T_recorded"] = float(s["T"])
            d["H_recorded"] = float(s["H"])
            for j in range(16):
                d[f"R{j + 1}"] = float(s["R"][j])
        else:
            d["T_recorded"] = np.nan
            d["H_recorded"] = np.nan
            for j in range(16):
                d[f"R{j + 1}"] = np.nan
        rows_h.append(d)
    df_h = pd.DataFrame(rows_h)

    df_meta = pd.DataFrame(
        [
            {
                "format": "JL_th_calibration_v1",
                "ref_temp_c": TH_COMP_REF_TEMP_C,
                "ref_humi_pct": TH_COMP_REF_HUMI_PCT,
            }
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df_t.to_excel(w, sheet_name=CAL_SHEET_TEMP, index=False)
        df_h.to_excel(w, sheet_name=CAL_SHEET_HUMI, index=False)
        df_meta.to_excel(w, sheet_name="meta", index=False)


def load_calibration_samples_xlsx(path):
    """
    从 xlsx 恢复 temp_samples / humi_samples（与内存结构一致）。
    未采集或电阻无效的行在对应标定点上为 None。
    """
    temp_samples = {t: None for t in TH_COMP_TEMP_POINTS}
    humi_samples = {h: None for h in TH_COMP_HUMI_POINTS}

    df_t = pd.read_excel(path, sheet_name=CAL_SHEET_TEMP)
    need_t = ["setpoint_C", "T_recorded", "H_recorded"] + [f"R{k}" for k in range(1, 17)]
    for col in need_t:
        if col not in df_t.columns:
            raise ValueError(f"工作表「{CAL_SHEET_TEMP}」缺少列: {col}")
    for _, row in df_t.iterrows():
        t_key = _match_calibration_setpoint(row["setpoint_C"], TH_COMP_TEMP_POINTS)
        if t_key is None:
            continue
        r = np.array(
            [pd.to_numeric(row[f"R{k}"], errors="coerce") for k in range(1, 17)],
            dtype=float,
        )
        if not np.all(np.isfinite(r)) or not np.all(r > 0):
            temp_samples[t_key] = None
            continue
        tr = pd.to_numeric(row["T_recorded"], errors="coerce")
        hr = pd.to_numeric(row["H_recorded"], errors="coerce")
        temp_samples[t_key] = {
            "R": r.copy(),
            "T": float(tr) if np.isfinite(tr) else 0.0,
            "H": float(hr) if np.isfinite(hr) else 0.0,
        }

    df_h = pd.read_excel(path, sheet_name=CAL_SHEET_HUMI)
    need_h = ["setpoint_RH_pct", "T_recorded", "H_recorded"] + [f"R{k}" for k in range(1, 17)]
    for col in need_h:
        if col not in df_h.columns:
            raise ValueError(f"工作表「{CAL_SHEET_HUMI}」缺少列: {col}")
    for _, row in df_h.iterrows():
        h_key = _match_calibration_setpoint(row["setpoint_RH_pct"], TH_COMP_HUMI_POINTS)
        if h_key is None:
            continue
        r = np.array(
            [pd.to_numeric(row[f"R{k}"], errors="coerce") for k in range(1, 17)],
            dtype=float,
        )
        if not np.all(np.isfinite(r)) or not np.all(r > 0):
            humi_samples[h_key] = None
            continue
        tr = pd.to_numeric(row["T_recorded"], errors="coerce")
        hr = pd.to_numeric(row["H_recorded"], errors="coerce")
        humi_samples[h_key] = {
            "R": r.copy(),
            "T": float(tr) if np.isfinite(tr) else 0.0,
            "H": float(hr) if np.isfinite(hr) else 0.0,
        }

    return temp_samples, humi_samples


class ThCompensationDialog(QtWidgets.QDialog):
    """温湿度补偿：采集标定点、曲线展示、计算系数、应用与保存。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.setWindowTitle("温湿度补偿")
        self.resize(1280, 760)

        self._temp_samples = {t: None for t in TH_COMP_TEMP_POINTS}
        self._humi_samples = {h: None for h in TH_COMP_HUMI_POINTS}
        self._coeffs = default_th_comp_coeffs()

        self._table_headers_temp = ["设定℃", "T记录", "H记录"] + list(TH_COMP_TABLE_R_LABELS) + ["状态"]
        self._table_headers_humi = ["设定%RH", "T记录", "H记录"] + list(TH_COMP_TABLE_R_LABELS) + ["状态"]
        self._r_col0 = 3
        self._status_col = self._r_col0 + len(TH_COMP_TABLE_R_IDX)

        root = QtWidgets.QVBoxLayout(self)
        tip = QtWidgets.QLabel(
            "基准点：20℃、50%RH。温度标定建议在约 50%RH 下采集 20/35/50℃；"
            "湿度标定建议在约 20℃ 下采集 20/50/80%RH。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet('font: 10pt "楷体"; color: rgb(70, 70, 70);')
        root.addWidget(tip)

        split = QtWidgets.QHBoxLayout()
        root.addLayout(split, 1)

        left = QtWidgets.QVBoxLayout()
        split.addLayout(left, 2)

        g_temp = QtWidgets.QGroupBox("温度标定点（℃）")
        g_temp.setStyleSheet('font: 11pt "楷体";')
        g_temp.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        lt = QtWidgets.QVBoxLayout(g_temp)
        self._tbl_temp = QtWidgets.QTableWidget(len(TH_COMP_TEMP_POINTS), len(self._table_headers_temp))
        self._tbl_temp.setHorizontalHeaderLabels(self._table_headers_temp)
        self._tbl_temp.verticalHeader().setVisible(False)
        self._tbl_temp.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        for i, t in enumerate(TH_COMP_TEMP_POINTS):
            self._tbl_temp.setItem(i, 0, QtWidgets.QTableWidgetItem(str(int(t))))
        self._stretch_fill_calibration_table(self._tbl_temp)
        lt.addWidget(self._tbl_temp, 1)
        left.addWidget(g_temp, 1)

        g_humi = QtWidgets.QGroupBox("湿度标定点（%RH）")
        g_humi.setStyleSheet('font: 11pt "楷体";')
        g_humi.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        lh = QtWidgets.QVBoxLayout(g_humi)
        self._tbl_humi = QtWidgets.QTableWidget(len(TH_COMP_HUMI_POINTS), len(self._table_headers_humi))
        self._tbl_humi.setHorizontalHeaderLabels(self._table_headers_humi)
        self._tbl_humi.verticalHeader().setVisible(False)
        self._tbl_humi.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        for i, h in enumerate(TH_COMP_HUMI_POINTS):
            self._tbl_humi.setItem(i, 0, QtWidgets.QTableWidgetItem(str(int(h))))
        self._stretch_fill_calibration_table(self._tbl_humi)
        lh.addWidget(self._tbl_humi, 1)
        left.addWidget(g_humi, 1)

        cap_row = QtWidgets.QHBoxLayout()
        self._mode_combo = QtWidgets.QComboBox()
        self._mode_combo.addItems(["湿度标定", "温度标定"])
        self._point_combo = QtWidgets.QComboBox()
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        cap_row.addWidget(QtWidgets.QLabel("模式:"))
        cap_row.addWidget(self._mode_combo)
        cap_row.addWidget(QtWidgets.QLabel("标定点:"))
        cap_row.addWidget(self._point_combo, 1)
        self._btn_capture = QtWidgets.QPushButton("采集当前点")
        self._btn_capture.clicked.connect(self._on_capture)
        cap_row.addWidget(self._btn_capture)
        left.addLayout(cap_row)

        cal_file_row = QtWidgets.QHBoxLayout()
        self._btn_save_cal = QtWidgets.QPushButton("保存标定数据")
        self._btn_save_cal.setToolTip("将当前温度/湿度标定点表格中的数据保存为 xlsx（含 16 路 R）")
        self._btn_save_cal.clicked.connect(self._on_save_calibration_data)
        self._btn_load_cal = QtWidgets.QPushButton("加载标定数据")
        self._btn_load_cal.setToolTip("从 xlsx 恢复标定点数据，可继续计算或核对")
        self._btn_load_cal.clicked.connect(self._on_load_calibration_data)
        cal_file_row.addWidget(self._btn_save_cal)
        cal_file_row.addWidget(self._btn_load_cal)
        cal_file_row.addStretch()
        left.addLayout(cal_file_row)

        btn_row = QtWidgets.QHBoxLayout()
        self._btn_calc = QtWidgets.QPushButton("计算补偿参数")
        self._btn_calc.clicked.connect(self._on_calculate)
        self._btn_save = QtWidgets.QPushButton("保存参数")
        self._btn_save.clicked.connect(self._on_save_params)
        self._btn_load = QtWidgets.QPushButton("加载参数")
        self._btn_load.clicked.connect(self._on_load_params)
        btn_row.addWidget(self._btn_calc)
        btn_row.addWidget(self._btn_save)
        btn_row.addWidget(self._btn_load)
        left.addLayout(btn_row)

        right = QtWidgets.QVBoxLayout()
        split.addLayout(right, 1)

        sen_row = QtWidgets.QHBoxLayout()
        sen_row.addWidget(QtWidgets.QLabel("展示传感器通道:"))
        self._sensor_spin = QtWidgets.QSpinBox()
        self._sensor_spin.setRange(1, 16)
        self._sensor_spin.valueChanged.connect(self._refresh_plots)
        sen_row.addWidget(self._sensor_spin)
        sen_row.addStretch()
        right.addLayout(sen_row)

        tabs = QtWidgets.QTabWidget()
        self._plot_temp = pg.PlotWidget()
        self._plot_temp.setBackground("w")
        _apply_standard_plot_axes(self._plot_temp)
        self._plot_temp.setLabel("left", "log10(R_ref/R)", color="black")
        self._plot_temp.setLabel("bottom", "温度 ℃", color="black")
        tighten_plot_layout(self._plot_temp, left_axis_width=52, bottom_axis_height=38)
        self._plot_humi = pg.PlotWidget()
        self._plot_humi.setBackground("w")
        _apply_standard_plot_axes(self._plot_humi)
        self._plot_humi.setLabel("left", "log10(R_ref/R)", color="black")
        self._plot_humi.setLabel("bottom", "湿度 %RH", color="black")
        tighten_plot_layout(self._plot_humi, left_axis_width=52, bottom_axis_height=38)
        tabs.addTab(self._plot_temp, "温度补偿拟合")
        tabs.addTab(self._plot_humi, "湿度补偿拟合")
        right.addWidget(tabs, 1)

        self._log = QtWidgets.QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(140)
        self._log.setPlaceholderText("计算与操作日志…")
        right.addWidget(self._log)

        close_row = QtWidgets.QHBoxLayout()
        close_row.addStretch()
        btn_close = QtWidgets.QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        close_row.addWidget(btn_close)
        root.addLayout(close_row)

        self._scatter_temp = self._plot_temp.plot([], [], pen=None, symbol="o", symbolBrush="b", symbolSize=8)
        self._line_temp = self._plot_temp.plot([], [], pen=pg.mkPen("r", width=2))
        self._scatter_humi = self._plot_humi.plot([], [], pen=None, symbol="o", symbolBrush="b", symbolSize=8)
        self._line_humi = self._plot_humi.plot([], [], pen=pg.mkPen("r", width=2))

        self._on_mode_changed()
        self._sync_from_main_window()
        self._refresh_tables()
        self._refresh_plots()

    def _log_line(self, text):
        self._log.append(text)

    @staticmethod
    def _stretch_fill_calibration_table(table):
        """标定表格横向铺满父区域：列均分剩余宽度。"""
        table.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        table.setMinimumHeight(90)
        hdr = table.horizontalHeader()
        hdr.setMinimumSectionSize(56)
        for c in range(table.columnCount()):
            hdr.setSectionResizeMode(c, QtWidgets.QHeaderView.Stretch)

    def _th_comp_storage_dir(self):
        """温湿度补偿存储目录：th_compensation/<设备号>/（系数与标定原始数据共用）。"""
        seg = self.main_window.get_device_segment() if self.main_window else "default"
        d = os.path.join(TH_COMP_DIR, seg)
        os.makedirs(d, exist_ok=True)
        return d

    def _params_path(self):
        return os.path.join(self._th_comp_storage_dir(), TH_COMP_COEFF_FILENAME)

    def _sync_from_main_window(self):
        if not self.main_window:
            return
        self._coeffs = getattr(self.main_window, "th_comp_coeffs", default_th_comp_coeffs())

    def _on_mode_changed(self):
        self._point_combo.clear()
        if self._mode_combo.currentIndex() == 0:
            for h in TH_COMP_HUMI_POINTS:
                self._point_combo.addItem(f"{int(h)} %RH", h)
        else:
            for t in TH_COMP_TEMP_POINTS:
                self._point_combo.addItem(f"{int(t)} ℃", t)

    def _read_r16_from_main(self):
        if not self.main_window or not self.main_window.latest_sensor_values:
            QtWidgets.QMessageBox.warning(self, "提示", "主窗口尚无串口数据，请先连接并采集。")
            return None
        lv = self.main_window.latest_sensor_values
        if len(lv) < 18:
            return None
        r16 = np.asarray(lv[:16], dtype=float)
        t, h = float(lv[16]), float(lv[17])
        return r16, t, h

    def _on_capture(self):
        pack = self._read_r16_from_main()
        if pack is None:
            return
        r16, t, h = pack
        if self._mode_combo.currentIndex() == 0:
            key = float(self._point_combo.currentData())
            self._humi_samples[key] = {"R": r16.copy(), "T": t, "H": h}
            self._log_line(f"湿度标定点 {key:g}%RH 已采集（T={t:.2f}, H={h:.2f}）。")
        else:
            key = float(self._point_combo.currentData())
            self._temp_samples[key] = {"R": r16.copy(), "T": t, "H": h}
            self._log_line(f"温度标定点 {key:g}℃ 已采集（T={t:.2f}, H={h:.2f}）。")
        self._refresh_tables()
        self._refresh_plots()

    @staticmethod
    def _fmt_r_cell(rv):
        if not np.isfinite(rv) or rv == float("inf"):
            return ""
        return f"{rv:.6g}"

    def _refresh_tables(self):
        for i, t in enumerate(TH_COMP_TEMP_POINTS):
            s = self._temp_samples[t]
            self._tbl_temp.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{s['T']:.4f}" if s else ""))
            self._tbl_temp.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{s['H']:.4f}" if s else ""))
            for col_k, j in enumerate(TH_COMP_TABLE_R_IDX):
                txt = self._fmt_r_cell(s["R"][j]) if s else ""
                self._tbl_temp.setItem(i, self._r_col0 + col_k, QtWidgets.QTableWidgetItem(txt))
            self._tbl_temp.setItem(
                i, self._status_col, QtWidgets.QTableWidgetItem("已采集" if s else "未采集")
            )
        for i, h in enumerate(TH_COMP_HUMI_POINTS):
            s = self._humi_samples[h]
            self._tbl_humi.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{s['T']:.4f}" if s else ""))
            self._tbl_humi.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{s['H']:.4f}" if s else ""))
            for col_k, j in enumerate(TH_COMP_TABLE_R_IDX):
                txt = self._fmt_r_cell(s["R"][j]) if s else ""
                self._tbl_humi.setItem(i, self._r_col0 + col_k, QtWidgets.QTableWidgetItem(txt))
            self._tbl_humi.setItem(
                i, self._status_col, QtWidgets.QTableWidgetItem("已采集" if s else "未采集")
            )

    def _refresh_plots(self):
        """散点：各标定点的 (实测 T 或 H, log10(R_ref/R))；红线：当前通道 f_T、f_H 二次拟合曲线。"""
        j = self._sensor_spin.value() - 1
        c = self._coeffs
        at, bt, ct = float(c["At"][j]), float(c["Bt"][j]), float(c["Ct"][j])
        ah, bh, ch = float(c["Ah"][j]), float(c["Bh"][j]), float(c["Ch"][j])

        ref_t_key = TH_COMP_REF_TEMP_C
        ref_h_key = TH_COMP_REF_HUMI_PCT
        ref_temp = self._temp_samples.get(ref_t_key)
        ref_humi = self._humi_samples.get(ref_h_key)

        xs_t, ys_t = [], []
        if ref_temp is not None:
            rref_t = ref_temp["R"][j]
            if np.isfinite(rref_t) and rref_t > 0:
                for t in TH_COMP_TEMP_POINTS:
                    s = self._temp_samples[t]
                    if s is None:
                        continue
                    rv = s["R"][j]
                    tv = float(s["T"])
                    if not np.isfinite(rv) or rv <= 0 or not np.isfinite(tv):
                        continue
                    xs_t.append(tv)
                    ys_t.append(float(np.log10(rref_t) - np.log10(rv)))
        if xs_t:
            order = np.argsort(xs_t)
            xs_t = [xs_t[i] for i in order]
            ys_t = [ys_t[i] for i in order]

        if len(xs_t) >= 1:
            t_lo, t_hi = min(xs_t), max(xs_t)
            if t_lo == t_hi:
                t_lo -= 0.5
                t_hi += 0.5
            tf = np.linspace(t_lo, t_hi, 96)
            yf = at * tf**2 + bt * tf + ct
        else:
            tf = np.array([])
            yf = np.array([])
        self._scatter_temp.setData(xs_t, ys_t)
        self._line_temp.setData(tf, yf)

        xs_h, ys_h = [], []
        if ref_humi is not None:
            rref_h = ref_humi["R"][j]
            if np.isfinite(rref_h) and rref_h > 0:
                for h in TH_COMP_HUMI_POINTS:
                    s = self._humi_samples[h]
                    if s is None:
                        continue
                    rv = s["R"][j]
                    hv = float(s["H"])
                    if not np.isfinite(rv) or rv <= 0 or not np.isfinite(hv):
                        continue
                    xs_h.append(hv)
                    ys_h.append(float(np.log10(rref_h) - np.log10(rv)))
        if xs_h:
            order = np.argsort(xs_h)
            xs_h = [xs_h[i] for i in order]
            ys_h = [ys_h[i] for i in order]

        if len(xs_h) >= 1:
            h_lo, h_hi = min(xs_h), max(xs_h)
            if h_lo == h_hi:
                h_lo -= 1.0
                h_hi += 1.0
            hf = np.linspace(h_lo, h_hi, 96)
            yfh = ah * hf**2 + bh * hf + ch
        else:
            hf = np.array([])
            yfh = np.array([])
        self._scatter_humi.setData(xs_h, ys_h)
        self._line_humi.setData(hf, yfh)

    def _on_save_calibration_data(self):
        default_name = time.strftime("TH_Cal_data_%Y%m%d_%H%M%S.xlsx")
        default_path = os.path.join(self._th_comp_storage_dir(), default_name)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存标定数据",
            default_path,
            "Excel (*.xlsx)",
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path = path + ".xlsx"
        try:
            save_calibration_samples_xlsx(path, self._temp_samples, self._humi_samples)
            self._log_line(f"标定数据已保存: {path}")
            QtWidgets.QMessageBox.information(
                self,
                "完成",
                f"已保存标定数据（工作表 {CAL_SHEET_TEMP}、{CAL_SHEET_HUMI}）：\n{path}",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"保存失败（需安装 openpyxl）：{e}")

    def _on_load_calibration_data(self):
        start_dir = self._th_comp_storage_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "加载标定数据",
            start_dir,
            "Excel (*.xlsx)",
        )
        if not path:
            return
        try:
            t_samp, h_samp = load_calibration_samples_xlsx(path)
            self._temp_samples = t_samp
            self._humi_samples = h_samp
            self._refresh_tables()
            self._refresh_plots()
            self._log_line(f"已从文件加载标定数据: {path}")
            QtWidgets.QMessageBox.information(
                self,
                "完成",
                "标定数据已载入表格与曲线，可直接「计算补偿参数」或继续补采。",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"加载失败：{e}")

    def _on_calculate(self):
        for h in TH_COMP_HUMI_POINTS:
            if self._humi_samples[h] is None:
                QtWidgets.QMessageBox.warning(self, "提示", f"湿度标定点尚未全部采集。")
                return
        for t in TH_COMP_TEMP_POINTS:
            if self._temp_samples[t] is None:
                QtWidgets.QMessageBox.warning(self, "提示", f"温度标定点尚未全部采集。")
                return

        ref_h_key = TH_COMP_REF_HUMI_PCT
        ref_t_key = TH_COMP_REF_TEMP_C
        humi_r_ref = self._humi_samples[ref_h_key]["R"]
        ref_s_h = self._humi_samples[ref_h_key]
        ref_s_t = self._temp_samples[ref_t_key]
        ref_h_meta = float(ref_s_h["H"]) if np.isfinite(ref_s_h["H"]) else float(ref_h_key)
        ref_t_meta = float(ref_s_t["T"]) if np.isfinite(ref_s_t["T"]) else float(ref_t_key)
        ah, bh, ch = [], [], []
        for j in range(16):
            h_arr = []
            y_arr = []
            for h in TH_COMP_HUMI_POINTS:
                s = self._humi_samples[h]
                r = s["R"][j]
                hv = float(s["H"])
                if not np.isfinite(r) or r <= 0:
                    continue
                if not np.isfinite(hv):
                    continue
                h_arr.append(hv)
                y_arr.append(np.log10(humi_r_ref[j]) - np.log10(r))
            a, b, c = fit_poly2(np.array(h_arr), np.array(y_arr))
            ah.append(0.0 if not np.isfinite(a) else a)
            bh.append(0.0 if not np.isfinite(b) else b)
            ch.append(0.0 if not np.isfinite(c) else c)

        temp_r_ref = self._temp_samples[ref_t_key]["R"]
        at, bt, ct = [], [], []
        for j in range(16):
            t_arr = []
            y_arr = []
            for t in TH_COMP_TEMP_POINTS:
                s = self._temp_samples[t]
                r = s["R"][j]
                tv = float(s["T"])
                if not np.isfinite(r) or r <= 0:
                    continue
                if not np.isfinite(tv):
                    continue
                t_arr.append(tv)
                y_arr.append(np.log10(temp_r_ref[j]) - np.log10(r))
            a, b, c = fit_poly2(np.array(t_arr), np.array(y_arr))
            at.append(0.0 if not np.isfinite(a) else a)
            bt.append(0.0 if not np.isfinite(b) else b)
            ct.append(0.0 if not np.isfinite(c) else c)

        self._coeffs = {
            "ref_temp_c": ref_t_meta,
            "ref_humi_pct": ref_h_meta,
            "At": at,
            "Bt": bt,
            "Ct": ct,
            "Ah": ah,
            "Bh": bh,
            "Ch": ch,
        }
        if self.main_window:
            self.main_window.th_comp_coeffs = dict(self._coeffs)
        self._log_line(
            "计算完成：已对 16 路分别拟合 f_H(H)、f_T(T)（log10 域加性项）。"
        )
        self._refresh_plots()
        QtWidgets.QMessageBox.information(self, "完成", "温湿度补偿参数已计算并写入主窗口，可保存为 xlsx。")

    def _on_save_params(self):
        path = self._params_path()
        try:
            save_th_comp_coeffs_xlsx(path, self._coeffs, True)
            self._log_line(f"已保存: {path}")
            QtWidgets.QMessageBox.information(self, "完成", f"已保存到:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"保存失败（需安装 openpyxl）: {e}")

    def _on_load_params(self):
        path = self._params_path()
        if not os.path.isfile(path):
            p, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "选择补偿参数 xlsx", os.path.dirname(path), "Excel (*.xlsx)"
            )
            if not p:
                return
            path = p
        try:
            c, _ = load_th_comp_coeffs_xlsx(path)
            self._coeffs = c
            if self.main_window:
                self.main_window.th_comp_coeffs = dict(c)
            self._log_line(f"已加载: {path}")
            self._refresh_plots()
            QtWidgets.QMessageBox.information(self, "完成", "参数已从 xlsx 加载。")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"加载失败: {e}")

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_from_main_window()
        self._refresh_tables()
        self._refresh_plots()

    def closeEvent(self, event):
        super().closeEvent(event)
