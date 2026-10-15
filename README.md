# 吉利汽车车载电子鼻两套二分类上位机

本项目由 `JL_sys_v4_6_demo.py` 拆分而来。两套工具共享采集、峰值汇总、特征工程、训练和界面代码，只在启动时读取一次任务配置，不提供运行时任务切换。

## 两个入口

- `python main_alcohol_vs_smoke.py`
  - 窗口标题：酒精/香烟二分类上位机
  - 分类标签：酒精、香烟
- `python main_alcohol_vs_smoke_odor.py`
  - 窗口标题：酒精/香烟-异味二分类上位机
  - 分类标签：酒精、香烟/异味
  - 加载数据时一次性将“香烟”和“异味”合并为“香烟/异味”
  - 历史数据中的“臭豆腐”等明确异味别名会先规范成“异味”

两套任务都使用 `group_zscore_resp_Sensor1~16 + StandardScaler + LinearSVC(C=1.0)`。空气继续在峰值识别前由设备标定门控，不进入二分类器训练。

## 目录职责

- `algorithm/`：特征、空气门控、共享二分类器、模型持久化和二分类评估。
- `acquisition/`：电子鼻/VOC串口线程、UDP注册线程和峰值事件检测。
- `gui/`：主窗口功能 mixin、模型中心容器、采集/训练/推理组合面板。
- `ui/`：Qt Designer 生成的主窗口代码与 `.ui` 原文件。
- `utils/`：运行路径、数据集读写和温湿度补偿。
- `task_config.py`：两个产品唯一的任务差异来源。

## 模型中心

两个入口的“模型中心”都使用 `ModelCenterDialog(mode="all")`，同一窗口保留：

1. 原始数据保存
2. 数据条管理与峰值样本记录
3. 模型训练
4. 模型加载和实时推理

容器本身仍支持 `mode="data"`、`mode="model"` 和 `mode="all"`。

## 数据路径

- 连续原始数据：
  `data/raw/<设备号>/<YYYYMMDD>/<标签>_<时间戳>.csv`
- VOC伴随文件：
  `data/raw/<设备号>/<YYYYMMDD>/<标签>_<时间戳>_VOC_<位置>.csv`
- 峰值/脉冲数据集：
  `data/datasets/<设备号>/*.xlsx`
- 训练模型：
  `models/<设备号>/*.joblib`

源码运行时两个入口位于同一项目目录，因此共享上述开发数据目录。将来分别打包并放入不同安装目录后，每个 EXE 都只使用自己所在目录下的 `data/` 和 `models/`，互不共享采集数据。

## 暂不打包

本轮先执行两个 Python 入口检查 GUI。后续确认后可运行：

```powershell
python build_all.py
```

两份 PyInstaller 配置位于 `build/`，会生成文件名和图标不同的独立程序。
