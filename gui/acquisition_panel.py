# -*- coding: utf-8 -*-
"""Data-acquisition panel shared unchanged by both binary products.

It contains raw continuous saving controls and peak/pulse dataset-row management.
Classification task differences are intentionally absent from this module.
"""

from pathlib import Path

from runtime_support import *
from algorithm.classifier import apply_task_labels
from gui.panel_base import OwnerPanel


class DataAcquisitionPanel(OwnerPanel):
    def _task_label(self, label):
        value = str(label or "").strip()
        return (self.task_config.get("label_merge_rule") or {}).get(value, value)

    def _build_raw_save_group(self, parent_layout):
        group_box = QtWidgets.QGroupBox("原始数据保存", self)
        group_box.setStyleSheet("font: 11pt \"楷体\";")
        layout = QtWidgets.QVBoxLayout(group_box)

        control_layout = QtWidgets.QHBoxLayout()
        self.raw_save_label_input = QtWidgets.QLineEdit(self)
        self.raw_save_label_input.setPlaceholderText("原始数据标签")
        if self.main_window is not None:
            self.raw_save_label_input.setText(
                getattr(self.main_window, "current_raw_save_label", "")
            )
        self.raw_save_start_btn = QtWidgets.QPushButton("开始保存")
        self.raw_save_stop_btn = QtWidgets.QPushButton("结束保存")
        control_layout.addWidget(QtWidgets.QLabel("标签:"))
        control_layout.addWidget(self.raw_save_label_input, 1)
        control_layout.addWidget(self.raw_save_start_btn)
        control_layout.addWidget(self.raw_save_stop_btn)
        layout.addLayout(control_layout)

        self.raw_save_status_label = QtWidgets.QLabel("")
        self.raw_save_status_label.setStyleSheet(
            "font: 10pt \"楷体\"; color: rgb(90, 90, 90);"
        )
        layout.addWidget(self.raw_save_status_label)

        parent_layout.addWidget(group_box)

        self.raw_save_label_input.textChanged.connect(self.remember_raw_save_label)
        self.raw_save_start_btn.clicked.connect(self.start_raw_saving)
        self.raw_save_stop_btn.clicked.connect(self.stop_raw_saving)
        self.refresh_raw_save_status()

    def remember_raw_save_label(self, text):
        if self.main_window is not None:
            self.main_window.current_raw_save_label = str(text or "").strip()

    def refresh_raw_save_status(self):
        if not hasattr(self, "raw_save_status_label"):
            return
        saving = bool(getattr(self.main_window, "is_saving", False))
        buffer_len = len(getattr(self.main_window, "data_buffer", []) or [])
        if saving:
            raw_written = int(
                getattr(self.main_window, "raw_save_rows_written", 0) or 0
            )
            voc_written = getattr(
                self.main_window,
                "voc_save_rows_written",
                {},
            )
            voc_buffers = getattr(self.main_window, "voc_save_buffers", {})
            voc_parts = []
            for key, label in VOC_POSITIONS:
                written = int(voc_written.get(key, 0) or 0)
                pending = len(voc_buffers.get(key, []) or [])
                voc_parts.append(f"{label} {written + pending}")
            text = (
                f"保存状态: 正在保存 | 电子鼻 {raw_written + buffer_len} 条"
                f" | VOC {' / '.join(voc_parts)}"
            )
        else:
            status_text = str(
                getattr(self.main_window, "raw_save_status_text", "未保存")
                or "未保存"
            )
            text = f"保存状态: {status_text}"
        self.raw_save_status_label.setText(text)
        if hasattr(self, "raw_save_start_btn"):
            self.raw_save_start_btn.setEnabled(not saving)
        if hasattr(self, "raw_save_stop_btn"):
            self.raw_save_stop_btn.setEnabled(saving)

    def start_raw_saving(self):
        self.remember_raw_save_label(self.raw_save_label_input.text())
        self.main_window.start_saving(self.raw_save_label_input.text())
        self.refresh_raw_save_status()

    def stop_raw_saving(self):
        self.main_window.stop_saving()
        self.refresh_raw_save_status()

    def sync_data_collection_state(self):
        if hasattr(self, "raw_save_label_input"):
            self.raw_save_label_input.blockSignals(True)
            self.raw_save_label_input.setText(
                getattr(self.main_window, "current_raw_save_label", "")
            )
            self.raw_save_label_input.blockSignals(False)
        self.refresh_raw_save_status()
        self.sync_peak_record_state()

    def _build_dataset_group(self, parent_layout):
        group_box = QtWidgets.QGroupBox("数据集管理", self)
        group_box.setStyleSheet("font: 11pt \"楷体\";")
        layout = QtWidgets.QVBoxLayout(group_box)

        file_layout = QtWidgets.QHBoxLayout()
        self.dataset_path_input = QtWidgets.QLineEdit(self)
        self.dataset_path_input.setPlaceholderText("路径：选择已有文件，或点「新建」按设备号创建 xlsx")
        self.dataset_select_btn = QtWidgets.QPushButton("选择")
        self.dataset_new_btn = QtWidgets.QPushButton("新建")
        self.dataset_load_btn = QtWidgets.QPushButton("加载")
        self.dataset_save_btn = QtWidgets.QPushButton("保存修改")
        self.dataset_save_as_btn = QtWidgets.QPushButton("另存为")
        file_layout.addWidget(self.dataset_path_input, 1)
        file_layout.addWidget(self.dataset_select_btn)
        file_layout.addWidget(self.dataset_new_btn)
        file_layout.addWidget(self.dataset_load_btn)
        file_layout.addWidget(self.dataset_save_btn)
        file_layout.addWidget(self.dataset_save_as_btn)
        layout.addLayout(file_layout)

        action_layout = QtWidgets.QHBoxLayout()
        self.dataset_label_filter = QtWidgets.QComboBox(self)
        self.dataset_label_filter.addItem("全部标签")
        self.append_label_input = QtWidgets.QLineEdit(self)
        self.append_label_input.setPlaceholderText("填写标签：可追加当前样本或记录峰值样本")
        self.append_current_btn = QtWidgets.QPushButton("追加当前样本")
        self.record_peak_btn = QtWidgets.QPushButton("开始记录峰值")
        self.record_pre_peak_window_btn = QtWidgets.QPushButton("开始记录峰前连续数据")
        self.delete_selected_btn = QtWidgets.QPushButton("删除选中样本")
        action_layout.addWidget(QtWidgets.QLabel("标签筛选:"))
        action_layout.addWidget(self.dataset_label_filter)
        action_layout.addWidget(self.append_label_input, 1)
        action_layout.addWidget(self.append_current_btn)
        action_layout.addWidget(self.record_peak_btn)
        action_layout.addWidget(self.record_pre_peak_window_btn)
        action_layout.addWidget(self.delete_selected_btn)
        layout.addLayout(action_layout)

        table_layout = QtWidgets.QHBoxLayout()
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(0)

        self.dataset_fixed_table = QtWidgets.QTableWidget(self)
        self.dataset_fixed_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.dataset_fixed_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.dataset_fixed_table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.SelectedClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
        )
        self.dataset_fixed_table.verticalHeader().setVisible(False)
        self.dataset_fixed_table.setAlternatingRowColors(True)
        self.dataset_fixed_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.dataset_fixed_table.setMinimumWidth(10)
        self.dataset_fixed_table.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        self.dataset_fixed_table.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.dataset_table = QtWidgets.QTableWidget(self)
        self.dataset_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.dataset_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.dataset_table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.SelectedClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
        )
        self.dataset_table.verticalHeader().setVisible(False)
        self.dataset_table.setAlternatingRowColors(True)
        self.dataset_table.setFrameShape(QtWidgets.QFrame.NoFrame)
        table_layout.addWidget(self.dataset_fixed_table)
        table_layout.addWidget(self.dataset_table, 1)
        layout.addLayout(table_layout, 1)

        self.dataset_status_label = QtWidgets.QLabel("当前未加载数据集。")
        self.dataset_status_label.setStyleSheet("font: 10pt \"楷体\"; color: rgb(90, 90, 90);")
        layout.addWidget(self.dataset_status_label)
        self.peak_record_status_label = QtWidgets.QLabel("峰值记录: 未启动")
        self.peak_record_status_label.setStyleSheet(
            "font: 10pt \"楷体\"; color: rgb(70, 90, 150);"
        )
        layout.addWidget(self.peak_record_status_label)

        parent_layout.addWidget(group_box, 1)

        self.dataset_select_btn.clicked.connect(self.select_dataset_file)
        self.dataset_new_btn.clicked.connect(self.new_dataset_xlsx)
        self.dataset_load_btn.clicked.connect(self.load_dataset_from_input)
        self.dataset_save_btn.clicked.connect(self.save_current_dataset)
        self.dataset_save_as_btn.clicked.connect(self.save_current_dataset_as)
        self.dataset_label_filter.currentTextChanged.connect(self.refresh_dataset_table)
        self.append_current_btn.clicked.connect(self.append_current_sample)
        self.record_peak_btn.clicked.connect(self.record_next_peak_sample)
        self.record_pre_peak_window_btn.clicked.connect(
            self.record_next_pre_peak_window_sample
        )
        self.delete_selected_btn.clicked.connect(self.delete_selected_samples)
        self.dataset_fixed_table.verticalScrollBar().valueChanged.connect(
            self.dataset_table.verticalScrollBar().setValue
        )
        self.dataset_table.verticalScrollBar().valueChanged.connect(
            self.dataset_fixed_table.verticalScrollBar().setValue
        )
        self.dataset_fixed_table.itemSelectionChanged.connect(self.sync_selection_from_fixed_table)
        self.dataset_table.itemSelectionChanged.connect(self.sync_selection_from_main_table)
        self.dataset_fixed_table.itemChanged.connect(self.mark_dataset_table_dirty)
        self.dataset_table.itemChanged.connect(self.mark_dataset_table_dirty)
        self.dataset_recovery_timer = QtCore.QTimer(self)
        self.dataset_recovery_timer.setSingleShot(True)
        self.dataset_recovery_timer.setInterval(800)
        self.dataset_recovery_timer.timeout.connect(
            self.write_dataset_recovery_snapshot
        )

    def mark_dataset_table_dirty(self, _item=None):
        self.dataset_table_dirty = True
        if hasattr(self, "dataset_recovery_timer"):
            self.dataset_recovery_timer.start()

    def dataset_recovery_path(self):
        """Return a sidecar workbook used to protect unsaved table changes."""
        target_path = self.dataset_path_input.text().strip()
        if target_path:
            target = Path(normalize_dataset_save_path(target_path))
            return str(
                target.with_name(f"{target.stem}_recovery{target.suffix}")
            )
        return os.path.join(
            self._device_datasets_dir(),
            "unsaved_dataset_recovery.xlsx",
        )

    def write_dataset_recovery_snapshot(self):
        """Persist the current table without changing the user's main file."""
        try:
            snapshot = (
                self.read_table_back_to_dataset()
                if self.dataset_table_dirty
                else self.dataset_df.copy()
            )
            snapshot = self.main_window._dataset_with_score_threshold_entries(
                snapshot
            )
            recovery_path = self.dataset_recovery_path()
            save_dataset_with_extra_columns(
                snapshot,
                recovery_path,
                self.dataset_format_info,
            )
            self.dataset_df = snapshot
            self.dataset_table_dirty = False
            self.dataset_status_label.setText(
                f"未保存修改已自动备份: {recovery_path} | 样本数: {len(snapshot)}"
            )
            return recovery_path
        except Exception as exc:
            self.dataset_status_label.setText(f"自动备份失败: {exc}")
            return ""

    def _device_datasets_dir_path(self):
        """data/datasets/<设备号> 路径（不创建目录）。"""
        return os.path.join(DATASETS_DIR, self.main_window.get_device_segment())

    def _device_datasets_dir(self):
        d = self._device_datasets_dir_path()
        os.makedirs(d, exist_ok=True)
        return d

    def _device_models_dir(self):
        d = os.path.join(MODELS_DIR, self.main_window.get_device_segment())
        os.makedirs(d, exist_ok=True)
        return d

    def select_dataset_file(self):
        start = self.dataset_path_input.text().strip()
        if not start:
            candidate = self._device_datasets_dir_path()
            start = candidate if os.path.isdir(candidate) else DATASETS_DIR
        selected_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择数据集文件",
            start,
            "Dataset Files (*.csv *.xlsx *.xls)",
        )
        if selected_path:
            self.dataset_path_input.setText(selected_path)

    def new_dataset_xlsx(self):
        """在 data/datasets/<设备号>/ 下按「设备号_时间戳」创建空 xlsx，并自动加载。"""
        base = self._device_datasets_dir()
        seg = self.main_window.get_device_segment()
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{seg}_{ts}.xlsx"
        path = os.path.join(base, filename)
        n = 0
        while os.path.exists(path):
            n += 1
            filename = f"{seg}_{ts}_{n}.xlsx"
            path = os.path.join(base, filename)

        try:
            empty_df = dataframe_from_records_with_extra([])
            save_dataset_with_extra_columns(
                empty_df,
                path,
                DatasetFormatInfo(path=path, extension=".xlsx"),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "错误",
                f"创建数据集文件失败（需安装 openpyxl）：\n{exc}",
            )
            return

        self.dataset_path_input.setText(path)
        self.load_dataset_from_input()
        QtWidgets.QMessageBox.information(
            self,
            "提示",
            f"已新建空数据集：\n{path}",
        )

    def load_dataset_from_input(self):
        dataset_path = self.dataset_path_input.text().strip()
        if not dataset_path:
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择数据集文件。")
            return

        try:
            if os.path.exists(dataset_path):
                self.dataset_df, self.dataset_format_info = load_dataset_with_extra_columns(dataset_path)
            else:
                self.dataset_df = dataframe_from_records_with_extra([])
                self.dataset_format_info = DatasetFormatInfo(path=dataset_path, extension=os.path.splitext(dataset_path)[1].lower())
            self.dataset_df, _ = apply_task_labels(
                self.dataset_df, self.task_config
            )
            self.dataset_df = ensure_dataset_extra_columns(self.dataset_df)
            self.dataset_path = dataset_path
            self.dataset_table_dirty = False
            self.refresh_label_filter()
            self.refresh_dataset_table()
            if hasattr(self, "train_path_input"):
                self.train_path_input.setText(dataset_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "错误", f"加载数据集失败：{exc}")

    def refresh_label_filter(self):
        current_label = self.dataset_label_filter.currentText()
        labels = sorted(
            {
                str(label).strip()
                for label in self.dataset_df["Label"].tolist()
                if str(label).strip()
            }
        )
        self.dataset_label_filter.blockSignals(True)
        self.dataset_label_filter.clear()
        self.dataset_label_filter.addItem("全部标签")
        self.dataset_label_filter.addItems(labels)
        if current_label in ["全部标签"] + labels:
            self.dataset_label_filter.setCurrentText(current_label)
        self.dataset_label_filter.blockSignals(False)

    def refresh_dataset_table(self, focus_source_index=None, _filter_retry=False):
        if self.dataset_table_dirty:
            self.dataset_df = self.read_table_back_to_dataset()
            self.dataset_table_dirty = False
        self.dataset_df = ensure_dataset_extra_columns(self.dataset_df)
        selected_label = self.dataset_label_filter.currentText()
        if selected_label and selected_label != "全部标签":
            view_df = self.dataset_df[self.dataset_df["Label"].astype(str) == selected_label]
        else:
            view_df = self.dataset_df

        self.current_view_indices = list(view_df.index)
        display_df = view_df.reset_index(drop=True)
        display_df.insert(0, "序号", [idx + 1 for idx in self.current_view_indices])
        fixed_columns = list(self.DATASET_FIXED_FIELDS)
        fixed_df = display_df[fixed_columns]
        main_df = display_df[[col for col in display_df.columns if col not in fixed_columns]]

        self.dataset_fixed_table.blockSignals(True)
        self.dataset_table.blockSignals(True)
        self.dataset_fixed_table.clear()
        self.dataset_fixed_table.setRowCount(len(fixed_df))
        self.dataset_fixed_table.setColumnCount(len(fixed_df.columns))
        self.dataset_fixed_table.setHorizontalHeaderLabels(list(self.DATASET_FIXED_HEADERS))

        self.dataset_table.clear()
        self.dataset_table.setRowCount(len(main_df))
        self.dataset_table.setColumnCount(len(main_df.columns))
        self.dataset_table.setHorizontalHeaderLabels(main_df.columns.tolist())

        header_colors = {
            "meta": QtGuiColor(245, 245, 245),
            "adc": QtGuiColor(255, 245, 210),
            "r": QtGuiColor(220, 245, 220),
            "baseline": QtGuiColor(220, 235, 255),
            "resp": QtGuiColor(245, 220, 255),
            "pattern": QtGuiColor(255, 225, 235),
        }

        for col_idx, column_name in enumerate(fixed_df.columns):
            header_item = self.dataset_fixed_table.horizontalHeaderItem(col_idx)
            if header_item is not None:
                header_item.setBackground(header_colors[self.get_column_group(column_name)])

        for col_idx, column_name in enumerate(main_df.columns):
            header_item = self.dataset_table.horizontalHeaderItem(col_idx)
            if header_item is not None:
                header_item.setBackground(header_colors[self.get_column_group(column_name)])

        for row_idx in range(len(fixed_df)):
            for col_idx, column_name in enumerate(fixed_df.columns):
                value = fixed_df.iloc[row_idx, col_idx]
                item = QtWidgets.QTableWidgetItem("" if pd_isna(value) else str(value))
                item.setBackground(header_colors[self.get_column_group(column_name)])
                if column_name == "序号":
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.dataset_fixed_table.setItem(row_idx, col_idx, item)

        for row_idx in range(len(main_df)):
            for col_idx, column_name in enumerate(main_df.columns):
                value = main_df.iloc[row_idx, col_idx]
                item = QtWidgets.QTableWidgetItem("" if pd_isna(value) else str(value))
                item.setBackground(header_colors[self.get_column_group(column_name)])
                self.dataset_table.setItem(row_idx, col_idx, item)

        self.dataset_fixed_table.resizeColumnsToContents()
        # 左侧固定三列：序号 | 时间戳 | 标签；控制最小宽度避免挤在一起或截断。
        if self.dataset_fixed_table.columnCount() >= 3:
            idx_col, ts_col, label_col = 0, 1, 2
            w_idx = self.dataset_fixed_table.columnWidth(idx_col)
            self.dataset_fixed_table.setColumnWidth(idx_col, max(w_idx, 52))
            w_ts = self.dataset_fixed_table.columnWidth(ts_col)
            self.dataset_fixed_table.setColumnWidth(ts_col, max(w_ts, 168))
            w_lb = self.dataset_fixed_table.columnWidth(label_col)
            self.dataset_fixed_table.setColumnWidth(label_col, max(w_lb, 100))

        fixed_width = self.dataset_fixed_table.verticalHeader().width() + 2
        for col in range(self.dataset_fixed_table.columnCount()):
            fixed_width += self.dataset_fixed_table.columnWidth(col)
        fixed_width += self.dataset_fixed_table.frameWidth() * 2
        self.dataset_fixed_table.setFixedWidth(max(360, fixed_width))
        self.dataset_table.resizeColumnsToContents()
        self.dataset_fixed_table.blockSignals(False)
        self.dataset_table.blockSignals(False)
        self.dataset_status_label.setText(
            f"当前数据集样本数: {len(self.dataset_df)} | 当前视图样本数: {len(display_df)}"
        )

        # 默认定位到首行；若指定 focus_source_index 则定位到该样本在视图中的行（必要时切换标签筛选）
        try:
            resp_first_col = RESP_COLUMNS[0]
            if resp_first_col not in main_df.columns or len(main_df) == 0:
                return

            target_col_idx = list(main_df.columns).index(resp_first_col)
            view_row = 0

            if focus_source_index is not None:
                try:
                    view_row = self.current_view_indices.index(focus_source_index)
                except ValueError:
                    if not _filter_retry and focus_source_index in self.dataset_df.index:
                        label_val = str(self.dataset_df.at[focus_source_index, "Label"]).strip()
                        self.dataset_label_filter.blockSignals(True)
                        if label_val and self.dataset_label_filter.findText(label_val) >= 0:
                            self.dataset_label_filter.setCurrentText(label_val)
                        else:
                            self.dataset_label_filter.setCurrentText("全部标签")
                        self.dataset_label_filter.blockSignals(False)
                        self.refresh_dataset_table(
                            focus_source_index=focus_source_index, _filter_retry=True
                        )
                        return
                    view_row = 0

            self.dataset_table.clearSelection()
            self.dataset_fixed_table.clearSelection()
            self.dataset_table.selectRow(view_row)
            self.dataset_fixed_table.selectRow(view_row)
            self.dataset_table.setCurrentCell(view_row, target_col_idx)
            target_item = self.dataset_table.item(view_row, target_col_idx)
            if target_item is not None:
                self.dataset_table.scrollToItem(
                    target_item, QtWidgets.QAbstractItemView.PositionAtCenter
                )
            fixed_item = self.dataset_fixed_table.item(view_row, 0)
            if fixed_item is not None:
                self.dataset_fixed_table.scrollToItem(
                    fixed_item, QtWidgets.QAbstractItemView.PositionAtCenter
                )
        except Exception:
            pass

    @staticmethod
    def get_column_group(column_name):
        if column_name == "序号":
            return "meta"
        if column_name == "Timestamp":
            return "meta"
        if column_name == SAMPLE_INDEX_COLUMN:
            return "meta"
        if column_name.startswith("ADC_"):
            return "adc"
        if column_name.startswith("R_"):
            return "r"
        if column_name.startswith("baseline_"):
            return "baseline"
        if column_name.startswith("centered_resp_"):
            return "resp"
        if column_name.startswith("group_zscore_resp_"):
            return "resp"
        if column_name.startswith("resp_"):
            return "resp"
        if column_name.startswith("pattern_"):
            return "pattern"
        if column_name == RESPONSE_STRENGTH_SCORE_COLUMN:
            return "resp"
        return "meta"

    def read_table_back_to_dataset(self):
        if (
            self.dataset_table.columnCount() == 0
            or self.dataset_fixed_table.columnCount() == 0
            or not self.current_view_indices
        ):
            return self.dataset_df.copy()

        updated_df = self.dataset_df.copy()
        main_columns = [
            self.dataset_table.horizontalHeaderItem(col).text()
            for col in range(self.dataset_table.columnCount())
        ]
        fixed_columns = list(self.DATASET_FIXED_FIELDS)

        text_columns = {
            "Time",
            "Label",
            "Timestamp",
            SAMPLE_INDEX_COLUMN,
            RESPONSE_STRENGTH_THRESHOLDS_COLUMN,
        }

        for view_row, source_index in enumerate(self.current_view_indices):
            for col_idx, column_name in enumerate(fixed_columns):
                if column_name == "序号":
                    continue
                item = self.dataset_fixed_table.item(view_row, col_idx)
                cell_text = item.text().strip() if item else ""
                updated_df.at[source_index, column_name] = cell_text

            for col_idx, column_name in enumerate(main_columns):
                item = self.dataset_table.item(view_row, col_idx)
                cell_text = item.text().strip() if item else ""

                if column_name in text_columns:
                    updated_df.at[source_index, column_name] = cell_text
                else:
                    try:
                        updated_df.at[source_index, column_name] = float(cell_text) if cell_text else np.nan
                    except ValueError:
                        updated_df.at[source_index, column_name] = cell_text

        return updated_df

    def sync_selection_from_fixed_table(self):
        self._sync_table_selection(self.dataset_fixed_table, self.dataset_table)

    def sync_selection_from_main_table(self):
        self._sync_table_selection(self.dataset_table, self.dataset_fixed_table)

    @staticmethod
    def _sync_table_selection(source_table, target_table):
        if target_table.rowCount() != source_table.rowCount():
            return
        selected_rows = {index.row() for index in source_table.selectionModel().selectedRows()}
        blocker = QtCore.QSignalBlocker(target_table)
        target_table.clearSelection()
        for row in selected_rows:
            target_table.selectRow(row)
        del blocker

    def get_selected_view_rows(self):
        rows = {index.row() for index in self.dataset_table.selectionModel().selectedRows()}
        rows.update(index.row() for index in self.dataset_fixed_table.selectionModel().selectedRows())
        return sorted(rows)

    def append_current_sample(self):
        label = self._task_label(self.append_label_input.text())
        record = self.main_window.get_current_dataset_record(label)
        if not record:
            QtWidgets.QMessageBox.warning(self, "提示", "当前没有可追加的实时样本。")
            return

        self.dataset_df = dataframe_append(self.dataset_df, [record])
        self.dataset_table_dirty = False
        self.write_dataset_recovery_snapshot()
        self.refresh_label_filter()
        new_source_index = len(self.dataset_df) - 1
        self.refresh_dataset_table(focus_source_index=new_source_index)

    def record_next_peak_sample(self):
        if self.main_window.is_next_peak_record_armed():
            self.main_window.cancel_next_peak_record()
            return

        label = self._task_label(self.append_label_input.text())
        ok, message = self.main_window.arm_next_peak_record(label)
        if not ok:
            QtWidgets.QMessageBox.warning(self, "提示", message)

    def record_next_pre_peak_window_sample(self):
        if self.main_window.is_pre_peak_window_record_armed():
            self.main_window.cancel_pre_peak_window_record()
            return

        label = self._task_label(self.append_label_input.text())
        ok, message = self.main_window.arm_pre_peak_window_record(label)
        if not ok:
            QtWidgets.QMessageBox.warning(self, "提示", message)

    def append_peak_record(self, record):
        if not record:
            QtWidgets.QMessageBox.warning(self, "提示", "未获取到峰值样本，无法追加。")
            return

        if self.dataset_table_dirty:
            self.dataset_df = self.read_table_back_to_dataset()
            self.dataset_table_dirty = False
        self.dataset_df = dataframe_append(self.dataset_df, [record])
        self.dataset_table_dirty = False
        self.write_dataset_recovery_snapshot()
        self.refresh_label_filter()
        new_source_index = len(self.dataset_df) - 1
        self.refresh_dataset_table(focus_source_index=new_source_index)

    def sync_peak_record_state(self):
        if not hasattr(self, "record_peak_btn") or not hasattr(self, "peak_record_status_label"):
            return
        armed = self.main_window.is_next_peak_record_armed()
        self.record_peak_btn.setText("结束记录峰值" if armed else "开始记录峰值")
        if hasattr(self, "record_pre_peak_window_btn"):
            window_armed = self.main_window.is_pre_peak_window_record_armed()
            self.record_pre_peak_window_btn.setText(
                "结束记录峰前连续数据" if window_armed else "开始记录峰前连续数据"
            )
        self.peak_record_status_label.setText(
            f"峰值记录: {self.main_window.next_peak_record_status}\n"
            f"峰前连续数据: {self.main_window.pre_peak_window_record_status}"
        )

    def delete_selected_samples(self):
        selected_rows = self.get_selected_view_rows()
        if not selected_rows:
            QtWidgets.QMessageBox.information(self, "提示", "请先在表格中选择要删除的样本行。")
            return

        source_indices = [self.current_view_indices[row] for row in selected_rows]
        self.dataset_df = self.read_table_back_to_dataset().drop(index=source_indices).reset_index(drop=True)
        self.dataset_table_dirty = False
        self.write_dataset_recovery_snapshot()
        self.refresh_label_filter()
        self.refresh_dataset_table()

    def save_current_dataset(self):
        self.dataset_df = self.read_table_back_to_dataset()
        self.dataset_df = self.main_window._dataset_with_score_threshold_entries(
            self.dataset_df
        )
        target_path = self.dataset_path_input.text().strip()
        if not target_path:
            self.save_current_dataset_as()
            return

        try:
            target_path = normalize_dataset_save_path(target_path)
            saved_path = save_dataset_with_extra_columns(
                self.dataset_df,
                target_path,
                self.dataset_format_info,
            )
            self.dataset_path = saved_path
            self.dataset_path_input.setText(saved_path)
            self.dataset_table_dirty = False
            self.dataset_status_label.setText(
                f"数据集已保存到: {saved_path} | 样本数: {len(self.dataset_df)}"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "错误", f"保存数据集失败：{exc}")

    def save_current_dataset_as(self):
        default_path = self.dataset_path_input.text().strip() or os.path.join(
            self._device_datasets_dir(), "dataset.xlsx"
        )
        selected_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "另存为数据集",
            default_path,
            "Excel Workbook (*.xlsx);;CSV File (*.csv)",
            "Excel Workbook (*.xlsx)",
        )
        if not selected_path:
            return

        try:
            selected_path = normalize_dataset_save_path(
                selected_path,
                selected_filter,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "错误", f"保存路径无效：{exc}")
            return
        self.dataset_path_input.setText(selected_path)
        self.dataset_path = selected_path
        self.save_current_dataset()
