# -*- coding: utf-8 -*-
"""Background worker for the shared closed-set binary training pipeline.

The thread owns file I/O and CPU work only. Widget state is updated through Qt
signals in ModelCenterDialog, keeping the original non-blocking interaction.
"""

from PyQt5 import QtCore
from sklearn.metrics import accuracy_score

from algorithm.classifier import train_group_zscore_resp_linear_model
from algorithm.features import GROUP_ZSCORE_RESP_COLUMNS
from algorithm.model_io import save_model_bundle
from task_config import validate_task_config
from utils.dataset_io import load_dataset


class TrainingThread(QtCore.QThread):
    training_finished = QtCore.pyqtSignal(dict)
    training_failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        train_path,
        test_path=None,
        model_output_path=None,
        validation_path=None,
        task_config=None,
    ):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        self.validation_path = validation_path
        self.task_config = validate_task_config(task_config or {})

    def run(self):
        try:
            train_df, _ = load_dataset(self.train_path)
            test_df = None
            if self.test_path:
                test_df, _ = load_dataset(self.test_path)
            artifacts = train_group_zscore_resp_linear_model(
                train_df,
                self.task_config,
                test_df=test_df,
            )
            model_path = save_model_bundle(self.model_output_path, artifacts)
            feature_df = artifacts["train_feature_df"]
            encoder = artifacts["label_encoder"]
            truth = encoder.transform(feature_df["Label"].astype(str))
            predicted = artifacts["svm_model"].predict(
                feature_df[GROUP_ZSCORE_RESP_COLUMNS].astype(float).values
            )
            summary = artifacts["train_summary"]
            mapping = summary.get("mapping_stats", {})
            summary_lines = [
                "训练完成",
                "模型保存路径: {}".format(model_path),
                "任务: {}".format(self.task_config.get("app_title", "")),
                "训练样本数: {}".format(summary["train_samples"]),
                "标签: {}".format(", ".join(self.task_config["class_labels"])),
                "特征模式: {}".format(artifacts["feature_spec"]["feature_mode"]),
                "训练集准确率: {:.4f}".format(
                    float(accuracy_score(truth, predicted))
                ),
                "标签合并样本: {}".format(mapping.get("merged_rows", 0)),
                "已排除空气样本: {}".format(mapping.get("air_rows", 0)),
                "已排除非本任务标签: {}".format(mapping.get("ignored_rows", 0)),
            ]
            if self.validation_path:
                summary_lines.append("验证集已忽略：闭集二分类不标定拒识阈值")
            evaluation = artifacts.get("evaluation") or {}
            if evaluation:
                summary_lines.extend(
                    [
                        "测试样本数: {}".format(evaluation.get("test_samples", 0)),
                        "测试集准确率: {:.4f}".format(
                            evaluation.get("accuracy", 0.0)
                        ),
                        "",
                        evaluation.get("classification_report_text", ""),
                    ]
                )
            self.training_finished.emit(
                {
                    "model_path": model_path,
                    "summary_text": "\n".join(summary_lines),
                }
            )
        except Exception as exc:
            self.training_failed.emit(str(exc))
