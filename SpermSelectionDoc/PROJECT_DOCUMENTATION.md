# 实时单精子优选分级系统 - 项目技术文档

> **版本**: 2.0  
> **更新日期**: 2026-05-23  
> **作者**: Simon  
> **适用对象**: 新入职程序员（无需软件开发经验）

---

## 目录

1. [项目简介](#1-项目简介)
2. [系统架构总览](#2-系统架构总览)
3. [环境安装与依赖配置](#3-环境安装与依赖配置)
4. [项目目录结构](#4-项目目录结构)
5. [核心算法详解](#5-核心算法详解)
   - 5.1 [YOLOv8 目标检测](#51-yolov8-目标检测)
   - 5.2 [JPDAF 多目标跟踪](#52-jpdaf-多目标跟踪)
   - 5.3 [ADSCNet 语义分割](#53-adscnet-语义分割)
   - 5.4 [运动学参数计算](#54-运动学参数计算)
   - 5.5 [形态学参数计算](#55-形态学参数计算)
   - 5.6 [精子分级算法](#56-精子分级算法)
6. [平滑与稳定机制](#6-平滑与稳定机制)
   - 6.1 [EMA 指数移动平均](#61-ema-指数移动平均)
   - 6.2 [等级投票机制](#62-等级投票机制)
   - 6.3 [滞后切换机制](#63-滞后切换机制)
   - 6.4 [边缘检测与强制切换](#64-边缘检测与强制切换)
7. [最优精子选择逻辑](#7-最优精子选择逻辑)
   - 7.1 [综合评分公式](#71-综合评分公式)
   - 7.2 [加权轮转调度](#72-加权轮转调度)
   - 7.3 [候选池管理](#73-候选池管理)
8. [数据库设计](#8-数据库设计)
   - 8.1 [数据库表结构](#81-数据库表结构)
   - 8.2 [数据持久化策略](#82-数据持久化策略)
9. [代码文件详解](#9-代码文件详解)
   - 9.1 [main.py — 程序入口](#91-mainpy--程序入口)
   - 9.2 [app.py — 主窗口与 UI 控制](#92-apppy--主窗口与-ui-控制)
   - 9.3 [video_thread.py — 视频采集线程](#93-video_threadpy--视频采集线程)
   - 9.4 [camera_thread.py — 相机采集线程](#94-camera_threadpy--相机采集线程)
   - 9.5 [detection_thread.py — 检测线程](#95-detection_threadpy--检测线程)
   - 9.6 [segment_thread.py — 分割线程](#96-segment_threadpy--分割线程)
   - 9.7 [jpdaf_tracker.py — JPDAF 跟踪器](#97-jpdaf_trackerpy--jpdaf-跟踪器)
   - 9.8 [sperm_registry.py — 精子注册表](#98-sperm_registrypy--精子注册表)
   - 9.9 [modules.py — 神经网络基础模块](#99-modulespy--神经网络基础模块)
   - 9.10 [model_zoo/ — 模型定义](#910-model_zoo--模型定义)
10. [线程架构与信号机制](#10-线程架构与信号机制)
11. [模型文件说明](#11-模型文件说明)
12. [使用指南](#12-使用指南)
13. [验证脚本说明](#13-验证脚本说明)
14. [常见问题与排查](#14-常见问题与排查)

---

## 1. 项目简介

### 这个项目是做什么的？

这是一个**实时单精子优选分级系统**。简单来说，它的工作是：

1. **看见精子** — 用显微镜摄像头拍摄精液样本的视频
2. **找到精子** — 用 AI（深度学习）自动识别视频中的每一个精子
3. **跟踪精子** — 持续追踪每个精子的运动轨迹
4. **分析精子** — 测量每个精子的运动速度、头部形状、颈部形态等参数
5. **给精子打分** — 根据 WHO 标准给精子分级（1级最好，5级最差）
6. **选出最优** — 从几百个精子中自动选出形态和运动能力综合最优的那个
7. **实时展示** — 在屏幕上用绿色方框标出"最佳精子"，供医生观察

### 通俗理解

想象一个操场上有很多人在跑步。这个系统就像一个"智能裁判"：
- 它能同时看到操场上所有人（检测）
- 它能记住每个人是谁、跑得多快（跟踪）
- 它能测量每个人的身高、体型（形态分析）
- 它能根据速度和体型综合打分（分级）
- 它能从中选出最优秀的运动员，并一直关注他（最优选择）

### 技术栈

| 组件 | 技术 | 作用 |
|------|------|------|
| 检测模型 | YOLOv8 + TensorRT | 实时检测视频中的精子 |
| 跟踪算法 | JPDAF (联合概率数据关联滤波器) | 多目标跟踪，处理遮挡和交叉 |
| 分割模型 | ADSCNet + TensorRT | 精确分割精子头部和颈部 |
| GUI 框架 | PyQt6 | 桌面图形界面 |
| 数据库 | SQLite (WAL 模式) | 持久化存储精子数据 |
| GPU 加速 | CUDA + TensorRT | 模型推理加速 |
| 相机 SDK | pypylon (Basler) | 工业相机控制 |

---

## 2. 系统架构总览

### 整体流程

```
视频/相机 → 帧图像 → YOLOv8检测 → JPDAF跟踪 → 精子注册表 → 最优精子选择
                                                        ↓
                              TensorRT分割 ← 调度器分配 ← 候选池
                                   ↓
                              形态学分析 → 分级 → 综合评分
```

### 多线程架构

系统采用**四线程流水线**设计，各线程通过 Qt 信号/槽机制通信：

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  VideoThread │────→│  DetectionThread  │────→│  SegmentThread   │
│  (视频采集)   │     │  (目标检测+跟踪)   │     │  (分割+形态分析)   │
└─────────────┘     └──────────────────┘     └──────────────────┘
       │                    │                        │
       │                    │                        │
       ▼                    ▼                        ▼
   帧图像信号         检测结果信号             分割结果信号
   new_frame         detection_result        segmented_frame
                          │                        │
                          ▼                        ▼
                   ┌──────────────┐         ┌──────────────┐
                   │ SpermRegistry │←────────│  最优精子更新   │
                   │ (精子注册表)   │         └──────────────┘
                   └──────────────┘
                          │
                          ▼
                   SQLite 数据库
```

**各线程职责：**

| 线程 | 文件 | 职责 |
|------|------|------|
| VideoThread | `video_thread.py` | 读取视频文件或相机，输出帧图像 |
| CameraThread | `camera_thread.py` | Basler 工业相机采集 |
| DetectionThread | `detection_thread.py` | YOLOv8 检测 + JPDAF 跟踪 + 最优精子选择 |
| SegmentThread | `segment_thread.py` | ADSCNet 分割 + 形态学测量 + 精子分级 |

---

## 3. 环境安装与依赖配置

### 3.1 硬件要求

| 硬件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | NVIDIA GPU（支持 CUDA 12.1） | RTX 3060 或更高 |
| 显存 | 4 GB | 8 GB+ |
| 内存 | 8 GB | 16 GB+ |
| 存储 | 20 GB 可用空间 | SSD 50 GB+ |

### 3.2 软件依赖

#### Python 版本
- **Python 3.9**（必须）

#### 核心依赖包及版本

| 包名 | 版本 | 用途 |
|------|------|------|
| `torch` | 2.5.0 | PyTorch 深度学习框架 |
| `torchvision` | 0.20.0 | 视觉模型工具 |
| `torchaudio` | 2.5.0 | 音频处理（PyTorch 依赖） |
| `tensorrt` | 10.9.0.34 | NVIDIA 推理加速引擎 |
| `pycuda` | - | CUDA Python 绑定 |
| `onnxruntime-gpu` | - | ONNX 模型推理 |
| `opencv-python` (cv2) | - | 图像处理 |
| `PyQt6` | - | GUI 框架 |
| `numpy` | - | 数值计算 |
| `albumentations` | - | 图像预处理增强 |
| `pypylon` | - | Basler 相机 SDK |
| `pyyaml` | - | YAML 配置文件解析 |

#### NVIDIA 驱动与工具链

| 组件 | 版本 | 说明 |
|------|------|------|
| CUDA | 12.1 | GPU 计算平台 |
| cuDNN | 8.7.29 | 深度学习加速库 |
| TensorRT | 10.9.0.34 | 推理优化引擎（项目内置） |

### 3.3 安装步骤

#### 第一步：安装 CUDA 12.1

1. 下载 CUDA 12.1：`https://developer.nvidia.com/cuda-12-1-1-download-archive`
2. 安装时选择"自定义安装"，确保安装 CUDA Toolkit
3. 添加环境变量：
   - `YOUR_PATH\cuda12.1-install\libnvvp`
   - `YOUR_PATH\cuda12.1-install\bin`

#### 第二步：安装 cuDNN 8.7.29

1. 下载 cuDNN：`https://developer.nvidia.com/rdp/cudnn-archive`
2. 解压后将 `bin/`、`include/`、`lib/` 文件夹复制到 CUDA 安装目录

#### 第三步：创建 Conda 虚拟环境

```bash
# 创建环境
conda create -n spermselectionv2 python=3.9
conda activate spermselectionv2

# 安装 PyTorch（CUDA 12.1 版本）
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

# 安装 ONNX Runtime GPU 版
pip install onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装其他依赖
pip install PyQt6 opencv-python numpy albumentations pypylon pyyaml pycuda
```

#### 第四步：安装 TensorRT

项目已内置 TensorRT 10.9.0.34（在 `TensorRT-10.9.0.34/` 目录），无需额外下载。

```bash
# 安装 TensorRT Python wheel
pip install TensorRT-10.9.0.34/python/tensorrt-10.9.0.34-cp39-none-win_amd64.whl
```

#### 第五步：验证安装

```python
import onnxruntime as ort
import tensorrt
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"ONNX Runtime device: {ort.get_device()}")
print(f"ONNX providers: {ort.get_available_providers()}")
print(f"TensorRT: {tensorrt.__version__}")
```

---

## 4. 项目目录结构

```
SpermSelectionV2_ForDemo_20250815/
│
├── main.py                          # 程序入口
├── app.py                           # 主窗口（41KB，最大文件）
├── video_thread.py                  # 视频采集线程
├── camera_thread.py                 # 相机采集线程
├── detection_thread.py              # 检测线程（42KB）
├── segment_thread.py                # 分割线程（60KB）
├── jpdaf_tracker.py                 # JPDAF 跟踪器（24KB）
├── sperm_registry.py                # 精子注册表（25KB）
├── modules.py                       # 神经网络基础模块
│
├── model_zoo/                       # 模型定义目录
│   ├── adscnet.py                   # ADSCNet 分割模型
│   └── edanet.py                    # EDANet 分割模型
│
├── yolo_weights/                    # YOLO 检测模型权重
│   ├── best.pt                      # PyTorch 原始权重（6MB）
│   ├── best.onnx                    # ONNX 导出权重（6MB）
│   └── best.engine                  # TensorRT 引擎（16MB）
│
├── seg_weights/                     # 分割模型权重
│   ├── ADSCNet_sperm_ROINAHead_250707/
│   │   ├── config.yml               # 训练配置
│   │   ├── model.onnx               # ONNX 权重
│   │   ├── model.pth                # PyTorch 权重
│   │   └── model_fp32.engine        # TensorRT FP32 引擎
│   ├── EDANet_sperm_NAHeadLatest250610_0610/
│   └── EDANet_sperm_ROINAHead_250707/
│
├── interfacedesign.ui               # Qt Designer UI 文件（52KB）
├── SpermSelection.qrc               # Qt 资源文件
├── resource-files/                  # UI 资源图片
│
├── SpermDatabase/                   # SQLite 数据库目录
│   └── sperm_experiment_*.db        # 实验数据库文件
│
├── SpermDatabaseVisualization/      # 数据库可视化工具
│   └── index.html                   # HTML 可视化页面
│
├── test-videoes/                    # 测试视频目录（~11GB）
│   └── *.mp4                        # 17 个测试视频文件
│
├── TensorRT-10.9.0.34/             # TensorRT SDK（项目内置）
│
├── yolov8_onnx2engine.py           # YOLOv8 ONNX→TensorRT 转换工具
├── convert_segModel_to_tensorrt.py  # 分割模型→TensorRT 转换工具
├── val_ROI_video_tensorrt.py        # 验证脚本：ROI 检测
├── val_ROI_video_JPDFAF_tensorrt.py # 验证脚本：JPDAF 跟踪
├── val_ROI_video_sperm_VSL_selection.py      # 验证脚本：VSL 选择
├── val_ROI_video_sperm_VSLMorph_selection.py # 验证脚本：VSL+形态选择
├── val_folder_seg_tensorrt.py       # 验证脚本：批量分割
├── val_video_seg_tensorrt.py        # 验证脚本：视频分割
└── val_video_tracking_tensorrt.py   # 验证脚本：视频跟踪
```

### 关键目录说明

| 目录 | 说明 |
|------|------|
| `yolo_weights/` | YOLOv8 检测模型，三种格式：.pt（训练用）、.onnx（中间格式）、.engine（推理用） |
| `seg_weights/` | 分割模型，包含 ADSCNet 和 EDANet 两种架构，每种都有多个训练版本 |
| `SpermDatabase/` | 每次运行自动创建新的 SQLite 数据库，按时间戳命名 |
| `test-videoes/` | 测试用的精液样本视频 |
| `TensorRT-10.9.0.34/` | TensorRT SDK 完整包，包含头文件、库文件、示例 |
| `resource-files/` | UI 界面使用的图标、Logo、产品说明书图片 |

---

## 5. 核心算法详解

### 5.1 YOLOv8 目标检测

#### 什么是 YOLOv8？

YOLO（You Only Look Once）是一种**实时目标检测**算法。你给它一张图片，它能告诉你：
- 图片里有哪些物体（精子）
- 每个物体在哪里（用方框标出位置）
- 它有多确信（置信度分数）

YOLOv8 是 YOLO 系列的最新版本，速度快、精度高。

#### 在本项目中的应用

```
输入：一帧显微镜图像（任意尺寸）
    ↓
预处理：BGR→RGB，缩放到 640×640，归一化到 [0,1]
    ↓
YOLOv8 推理：TensorRT 加速，FP16 精度
    ↓
后处理：置信度过滤 (≥0.25)，NMS 去重 (IoU≥0.45)
    ↓
输出：每个精子的边界框 [x1, y1, x2, y2] + 置信度
```

#### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 输入尺寸 | 640×640 | 模型输入分辨率 |
| 置信度阈值 | 0.25 | 低于此值的检测结果被丢弃 |
| NMS IoU 阈值 | 0.45 | 重叠度超过此值的框被合并 |
| 最大检测数 | 150 | 每帧最多检测 150 个目标 |
| 数据类型 | FP16 | 半精度浮点，加速推理 |

#### 代码位置

- **推理逻辑**：`detection_thread.py` → `run()` 方法中的 `_preprocess()` → `_infer()` → `_postprocess()`
- **NMS 实现**：`detection_thread.py` → `_nms()` 方法

---

### 5.2 JPDAF 多目标跟踪

#### 为什么需要跟踪？

检测只能告诉我们"这一帧里有精子 A、B、C"，但不知道"A 在上一帧是哪个"。跟踪算法解决这个问题：它给每个精子分配一个**唯一 ID**，跨帧持续追踪。

#### 什么是 JPDAF？

JPDAF（Joint Probabilistic Data Association Filter，联合概率数据关联滤波器）是一种多目标跟踪算法。它擅长处理：
- **多个目标靠近**：精子们经常游到一起
- **目标交叉**：两个精子的轨迹交叉时，不会搞混
- **漏检**：某一帧没检测到某个精子，跟踪不会断
- **新目标出现**：新的精子进入视野时自动创建新轨迹

#### 算法流程

```
当前帧检测结果（N 个边界框）
         ↓
    ┌────────────┐
    │  预测步骤    │  对每个已有轨迹，用卡尔曼滤波器预测下一帧位置
    │ (Predict)   │  预测位置 + 不确定性椭圆（门控区域）
    └────────────┘
         ↓
    ┌────────────┐
    │  验证步骤    │  检查哪些检测结果落在哪个轨迹的门控区域内
    │ (Validate)  │  构建验证矩阵（tracks × measures）
    └────────────┘
         ↓
    ┌────────────┐
    │  聚类步骤    │  将有重叠关联的轨迹和检测分组
    │ (Cluster)   │  同一组内的目标可能互相干扰
    └────────────┘
         ↓
    ┌────────────┐
    │  概率计算    │  对每个聚类，枚举所有可能的关联事件
    │ (JPDAF Prob)│  计算每个事件的概率，得到边缘关联概率
    └────────────┘
         ↓
    ┌────────────┐
    │  更新步骤    │  用加权创新（innovation）更新每个轨迹的状态
    │ (Correct)   │  卡尔曼滤波器的 JPDAF 变体
    └────────────┘
         ↓
    ┌────────────┐
    │  管理步骤    │  删除消失的轨迹，创建新轨迹
    │ (Manage)    │  连续 10 帧未匹配 → 删除
    └────────────┘
```

#### 卡尔曼滤波器

每个轨迹内部使用**4 维状态卡尔曼滤波器**：

**状态向量**：`[x, y, vx, vy]`（位置 + 速度）

**状态转移矩阵** F（假设 30fps）：
```
F = [1  0  1/30  0  ]
    [0  1  0     1/30]
    [0  0  1     0   ]
    [0  0  0     1   ]
```

**自适应过程噪声**：`Q = 0.9 × Q_prev + 0.1 × Q0`

#### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 过程噪声 | 20.0 | 卡尔曼滤波器的过程噪声 |
| 测量噪声 | 10.0 | 卡尔曼滤波器的测量噪声 |
| 检测概率 | 0.80 | 期望的检测率 |
| 门控概率 | 0.95 | 门控区域覆盖概率 |
| 删除阈值 | 5.0 | 轨迹删除的门控椭圆放大倍数 |
| 最大连续丢失 | 10 帧 | 超过则删除轨迹 |
| 最小轨迹年龄 | 5 帧 | 超过此年龄才报告为活跃轨迹 |

#### 代码位置

- **完整实现**：`jpdaf_tracker.py`
- **JPDATrack 类**：单个轨迹，包含卡尔曼滤波器（line 12-323）
- **JPDAFilter 类**：多轨迹管理器（line 325-617）

---

### 5.3 ADSCNet 语义分割

#### 什么是语义分割？

语义分割是给图片中的**每个像素**分类。在本项目中，它把精子的 64×64 ROI 图像分成 4 类：

| 类别索引 | 名称 | 颜色 | 说明 |
|---------|------|------|------|
| 0 | 顶体 (Acrosome) | 红色 | 精子头部前端 |
| 1 | 细胞核 (Nucleus) | 绿色 | 精子头部主体 |
| 2 | 不可测量头部 | 蓝色 | 形态异常的头部 |
| 3 | 颈部 (Neck) | 黄色 | 连接头部和尾部的部分 |

#### ADSCNet 架构

ADSCNet（Asymmetric Depthwise Separable Convolution Network）是一种轻量级实时分割网络：

```
输入: 64×64×3 (RGB 图像)
    ↓
编码器 (Encoder)
    ├─ Conv0: 3→32 通道, stride=2  → 32×32×32
    ├─ Conv1: ADSCModule(32)      → 32×32×32
    ├─ Conv2-4: 32→32→64          → 16×16×64
    └─ Conv5: ADSCModule(64, s=2) → 8×8×128
    ↓
上下文模块 (DDCC)
    ├─ 4 个不同膨胀率的 ADSCModule: [3, 5, 9, 13]
    ├─ 密集连接：每个模块接收前面所有模块的输出
    └─ 输出: 8×8×128
    ↓
解码器 (Decoder)
    ├─ Up1: 反卷积 128→64 + 跳跃连接  → 16×16×64
    ├─ Up2: 反卷积 64→32             → 32×32×32
    ├─ Up3: 反卷积 32→16             → 64×64×16
    └─ Conv: 1×1 卷积 16→4           → 64×64×4
    ↓
输出: 64×64×4 (4 类概率图)
```

**ADSCModule 核心**：使用非对称卷积 `(3,1)` 和 `(1,3)` 代替标准 `3×3` 卷积，减少计算量。

#### 训练配置

| 参数 | 值 |
|------|-----|
| 输入尺寸 | 64×64 |
| 类别数 | 4 |
| 损失函数 | BCEDiceLoss |
| 优化器 | Adam (lr=0.001) |
| 学习率调度 | CosineAnnealingLR |
| 批大小 | 8 |
| 训练轮数 | 100 |

#### 代码位置

- **模型定义**：`model_zoo/adscnet.py` → `ADSCNet` 类
- **推理流程**：`segment_thread.py` → `_preprocess()` → `_infer()` → `_postprocess()`
- **训练配置**：`seg_weights/ADSCNet_sperm_ROINAHead_250707/config.yml`

---

### 5.4 运动学参数计算

运动学参数描述精子的**运动特征**，主要在 `jpdaf_tracker.py` 的 `JPDATrack.calculate_motion_parameters()` 中计算。

#### VSL（直线运动速度）

**定义**：精子在一定时间内走过的直线距离除以时间。

```
VSL = distance(起点, 终点) / 时间差

起点 = 轨迹中最近 30 个点的第一个
终点 = 轨迹中最近 30 个点的最后一个
时间差 = 终点时间 - 起点时间
单位: μm/s（微米/秒）
```

**通俗理解**：如果精子从 A 点直线游到 B 点，VSL 就是 A→B 的速度。游得越直越快，VSL 越高。

#### ALH（头部侧向振幅）

**定义**：精子头部偏离直线运动方向的平均距离。

```
1. 对最近 N 个轨迹点拟合一条直线（最小二乘法）
2. 计算每个点到这条直线的垂直距离
3. ALH = 2 × 平均垂直距离
单位: μm（微米）
```

**通俗理解**：精子游泳时头部会左右摆动，ALH 就是摆动的幅度。摆动太大说明运动不协调。

#### 像素到微米转换

```python
pixel_to_micron = 0.7  # 1 像素 = 0.7 微米
```

---

### 5.5 形态学参数计算

形态学参数描述精子的**物理形状**，主要在 `segment_thread.py` 中根据分割结果计算。

#### 头部参数

| 参数 | 计算方法 | 说明 |
|------|---------|------|
| head_length | 椭圆拟合长轴 | 头部长度（μm） |
| head_width | 椭圆拟合短轴 | 头部宽度（μm） |
| head_ratio | length / width | 头部长宽比 |
| head_area | 轮廓面积 × pixel²_to_μm² | 头部面积（μm²） |

**计算流程**：
1. 合并顶体（class 0）和细胞核（class 1）的 mask
2. 提取最大轮廓
3. 用 `cv2.fitEllipse()` 拟合椭圆
4. 从椭圆的长轴、短轴计算各项参数

#### 颈部参数

| 参数 | 计算方法 | 说明 |
|------|---------|------|
| neck_width | 距离变换 95 分位数 × 2 | 颈部宽度（μm） |
| neck_length | 骨架点链长度之和 | 颈部长度（μm） |
| neck_bent_angle | 骨架方向变化最大角度 | 颈部弯曲角度（°） |
| neck_head_angle | 头部椭圆长轴 vs 颈部起始方向 | 头颈角度（°） |

**计算流程**：
1. 提取颈部 mask（class 3）
2. `cv2.ximgproc.thinning()` 骨架化（Guo-Hall 算法）
3. 最近邻排序骨架点，形成有序链
4. 沿骨架采样 `cv2.distanceTransform` 值，取 95 分位数作为宽度
5. 计算骨架方向变化，取最大角度作为弯曲角
6. 拟合头部椭圆，计算长轴与颈部起始方向的夹角

---

### 5.6 精子分级算法

#### 分级标准

系统根据运动学和形态学参数，将精子分为 5 个等级：

| 等级 | 含义 | VSL (μm/s) | ALH (μm) | 头部长宽比 | 头部面积 | 颈部弯曲角 | 头颈角 |
|------|------|-----------|----------|-----------|---------|-----------|--------|
| Grade 1 | 优秀 | ≥11.0 | ≤0.6 | 1.2~2.0 | 正常范围 | <10° | <10° |
| Grade 2 | 良好 | ≥10.0 | ≤0.8 | - | - | - | - |
| Grade 3 | 一般 | ≥7.5 | ≤1.0 | - | - | - | - |
| Grade 4 | 较差 | ≥5.0 | - | ≤2.5 | (10,90] | [10,20) | [10,20) |
| Grade 5 | 差 | ≥2.0 | - | - | (10,110] | (0,34) | (0,34) |
| -1 | 未分级 | <2.0 | - | - | - | - | - |

**Grade 1 最好，Grade 5 最差，-1 表示数据不足无法分级。**

#### 分级流程

```
形态学测量结果
    ↓
检查各项参数是否满足等级阈值
    ↓
从 Grade 1 开始逐级检查
    ↓
满足所有条件 → 返回该等级
不满足任何等级 → 返回 -1（未分级）
```

#### 代码位置

- **分级逻辑**：`segment_thread.py` → `_grade_sperm()` 方法
- **等级常量**：`detection_thread.py` → `GradeParameters` 类

---

## 6. 平滑与稳定机制

### 6.1 EMA 指数移动平均

#### 什么是 EMA？

EMA（Exponential Moving Average）是一种数据平滑方法。它给新数据和旧数据不同的权重，让结果不会因为某一次测量的波动而剧烈变化。

**公式**：
```
smooth_new = α × raw_new + (1 - α) × smooth_old

其中 α = 0.15（平滑系数）
```

**通俗理解**：想象你在开车看导航。导航不会因为 GPS 跳了一下就突然改变路线，它会"综合考虑"之前的位置。α 就是"多大程度上相信最新的 GPS 信号"。

#### 应用范围

EMA 被应用到以下参数：

| 参数类别 | 具体参数 | 平滑字段名 |
|---------|---------|-----------|
| 运动学 | VSL | vsl_smooth |
| 运动学 | ALH | alh_smooth |
| 形态学 | 头部长度 | head_length_smooth |
| 形态学 | 头部宽度 | head_width_smooth |
| 形态学 | 头部长宽比 | head_ratio_smooth |
| 形态学 | 头部面积 | head_area_smooth |
| 形态学 | 颈部宽度 | neck_width_smooth |
| 形态学 | 颈部弯曲角 | neck_bent_angle_smooth |
| 形态学 | 头颈角 | neck_head_angle_smooth |

#### 代码位置

- **EMA 函数**：`sperm_registry.py` → `_ema()` 方法
- **调用位置**：`update_kinematics()` 和 `update_morphology()` 方法中

---

### 6.2 等级投票机制

#### 为什么要投票？

精子的形态学分级可能因为测量噪声而在帧与帧之间波动（比如这一帧是 Grade 4，下一帧变成 Grade 5，再下一帧又回到 Grade 4）。投票机制确保只有**稳定一致**的等级才会被采纳。

#### 投票规则

```
1. 维护一个滑动窗口，记录最近 20 次形态学测量的等级
2. 过滤掉 -1（未分级）的记录
3. 统计每个等级出现的次数
4. 如果某个等级的占比 ≥ 60%，采纳该等级为"稳定等级"
5. 否则，稳定等级保持为 -1（待定）
```

**示例**：
```
最近 20 次测量: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4]
Grade 4 出现 18 次 = 90% ≥ 60% → 稳定等级 = 4
```

#### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 窗口大小 | 20 | 滑动窗口记录的测量次数 |
| 投票阈值 | 0.6 (60%) | 等级被采纳的最低占比 |

#### 代码位置

- **投票函数**：`sperm_registry.py` → `_vote_grade()` 方法
- **调用位置**：`update_morphology()` 方法中

---

### 6.3 滞后切换机制

#### 为什么要滞后？

假设当前最优精子是 A（得分 0.85）。突然 B 的得分变成 0.86，如果立即切换，下一帧 A 可能又变成 0.87。这样来回切换会导致屏幕上的绿框不断闪烁。

滞后机制要求：**挑战者必须连续 30 帧都比当前最优高出 0.05 分，才会真正切换。**

#### 切换规则

```
当前最优精子: A (score = 0.80)
挑战者: B (score = 0.86)

帧 1:  B - A = 0.06 > 0.05 → hold_count = 1
帧 2:  B - A = 0.07 > 0.05 → hold_count = 2
...
帧 30: B - A = 0.05 > 0.05 → hold_count = 30
→ 切换！最优精子变为 B
```

如果中间某一帧 B 的优势不足 0.05，`hold_count` 重置为 0，重新开始计数。

#### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 持续帧数 | 30 | 挑战者需要持续领先的帧数 |
| 分数差距 | 0.05 | 挑战者必须超过当前最优的分数 |

#### 代码位置

- **滞后逻辑**：`sperm_registry.py` → `get_best_sperm()` 方法

---

### 6.4 边缘检测与强制切换

#### 为什么要检测边缘？

当最优精子游到画面边缘时，它的检测框可能不完整（部分超出画面），导致形态学测量不准确。此时应该**立即切换**到其他精子，而不是等滞后机制慢慢切换。

#### 边缘判断

使用**检测框（bounding box）**是否触碰画面边界来判断：

```python
def _is_bbox_at_edge(rec, frame_w, frame_h):
    return (rec.box_x1 <= 0 or      # 左边触碰
            rec.box_y1 <= 0 or       # 上边触碰
            rec.box_x2 >= frame_w or # 右边触碰
            rec.box_y2 >= frame_h)   # 下边触碰
```

**只要检测框的任意一边碰到画面边界，就触发强制切换。**

#### 切换逻辑

```
如果当前最优精子的 bbox 触碰边缘:
    从候选池中选择排名最高的非边缘精子
    如果有 → 立即切换（跳过滞后机制）
    如果没有 → 保持当前（没有更好的选择）
```

#### 代码位置

- **边缘判断**：`sperm_registry.py` → `_is_bbox_at_edge()` 静态方法
- **强制切换**：`sperm_registry.py` → `get_best_sperm()` 方法开头

---

## 7. 最优精子选择逻辑

### 7.1 综合评分公式

系统用一个综合分数来评价每个精子的"好坏"：

```
综合分 = 0.5 × 运动学分 + 0.5 × 形态学分 × 置信度

其中:
  运动学分 = kinematic_continuous（基于 EMA 平滑后的 VSL 和 ALH）
  形态学分 = morphology_continuous（基于投票后的稳定等级）
  置信度 = min(测量次数 / 20, 1.0)
```

#### 运动学连续分

不使用离散的等级，而是用连续值：

```
vsl_score = min(vsl_smooth / 30.0, 1.0)    # VSL 归一化，30 μm/s = 满分
alh_score = max(0, 1 - alh_smooth / 2.0)   # ALH 惩罚，2.0 μm = 零分

运动学连续分 = 0.7 × vsl_score + 0.3 × alh_score
```

#### 形态学连续分

```
形态学连续分 = max(0, (6 - stable_grade) / 5.0)

Grade 1 → 1.0（满分）
Grade 2 → 0.8
Grade 3 → 0.6
Grade 4 → 0.4
Grade 5 → 0.2
Grade 6 或 -1 → 0.0
```

#### 置信度权重

测量次数越多，形态学分越可靠：
```
confidence = min(测量次数 / 20, 1.0)

测量 1 次 → confidence = 0.05（几乎不信任）
测量 10 次 → confidence = 0.5（半信半疑）
测量 20+ 次 → confidence = 1.0（完全信任）
```

### 7.2 加权轮转调度

每帧要选择 10 个精子进行分割分析。选择策略是**加权轮转**：

```
对每个活跃精子，计算调度分:
  morph_need = 1.0 - min(测量次数 / 10, 1.0)  # 测量需求（未测过的优先）
  vsl_norm = min(vsl / 30.0, 1.0)              # VSL 归一化
  edge_norm = min(到边缘距离 / 200, 1.0)       # 中心位置优先

调度分 = 0.4 × morph_need + 0.3 × vsl_norm + 0.3 × edge_norm

修正:
  如果已在候选池中 → 调度分 × 0.3（降低优先级，给其他精子机会）
  如果 30 帧未被测量 → 调度分 × 2.0（提高优先级，防止被遗忘）
```

**通俗理解**：系统优先安排"还没怎么测过的、游得快的、在画面中间的"精子。已经在候选池里的精子降低优先级，但如果很久没被测到，又会提高优先级。

### 7.3 候选池管理

#### 进入候选池的条件

```
形态学测量次数 ≥ 10 次 → 自动进入候选池
```

#### 候选池排序规则

当需要选择"最优精子"时，从候选池中按以下优先级排序：

1. **综合分**（composite_score）— 高分优先
2. **测量次数**（morphology_measurement_count）— 数据多的优先
3. **颈部角度之和**（neck_bent_angle + neck_head_angle）— 角度小的优先

---

## 8. 数据库设计

### 8.1 数据库表结构

系统使用 SQLite 数据库，包含 3 张表：

#### 表 1: `sperm_records`（精子记录表）

每行代表一个被追踪的精子的最新状态。主键为 `track_id`。

| 列名 | 类型 | 说明 |
|------|------|------|
| track_id | INTEGER | 轨迹 ID（主键） |
| vsl | REAL | 直线运动速度（原始值） |
| alh | REAL | 头部侧向振幅（原始值） |
| kinematic_grade | INTEGER | 运动学等级 |
| pos_x, pos_y | REAL | 最后位置 |
| box_x1, box_y1, box_x2, box_y2 | REAL | 检测框坐标 |
| head_length | REAL | 头部长度 |
| head_width | REAL | 头部宽度 |
| head_ratio | REAL | 头部长宽比 |
| head_area | REAL | 头部面积 |
| neck_width | REAL | 颈部宽度 |
| neck_length | REAL | 颈部长度 |
| neck_head_angle | REAL | 头颈角度 |
| neck_bent_angle | REAL | 颈部弯曲角 |
| morphology_grade | INTEGER | 形态学等级 |
| morphology_measurement_count | INTEGER | 形态学测量次数 |
| last_measurement_frame | INTEGER | 最后测量帧号 |
| in_candidate_pool | INTEGER | 是否在候选池 (0/1) |
| scheduling_score | REAL | 调度分 |
| composite_score | REAL | 综合评分 |
| first_seen | REAL | 首次出现时间戳 |
| last_updated | REAL | 最后更新时间戳 |
| is_active | INTEGER | 是否活跃 (0/1) |
| vsl_smooth | REAL | EMA 平滑后的 VSL |
| alh_smooth | REAL | EMA 平滑后的 ALH |
| head_length_smooth | REAL | EMA 平滑后的头部长度 |
| head_width_smooth | REAL | EMA 平滑后的头部宽度 |
| head_ratio_smooth | REAL | EMA 平滑后的头部长宽比 |
| head_area_smooth | REAL | EMA 平滑后的头部面积 |
| neck_width_smooth | REAL | EMA 平滑后的颈部宽度 |
| neck_bent_angle_smooth | REAL | EMA 平滑后的颈部弯曲角 |
| neck_head_angle_smooth | REAL | EMA 平滑后的头颈角 |
| grade_history | TEXT | 等级投票历史（JSON 列表） |
| stable_morph_grade | INTEGER | 投票后的稳定等级 |
| kinematic_continuous | REAL | 运动学连续分 |
| morphology_continuous | REAL | 形态学连续分 |

#### 表 2: `morphology_history`（形态学历史表）

每次形态学测量都会插入一行新记录（追加式日志）。

| 列名 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 自增主键 |
| track_id | INTEGER | 轨迹 ID |
| timestamp | REAL | 测量时间戳 |
| frame_number | INTEGER | 帧号 |
| head_length | REAL | 头部长度 |
| head_width | REAL | 头部宽度 |
| head_ratio | REAL | 头部长宽比 |
| head_area | REAL | 头部面积 |
| neck_width | REAL | 颈部宽度 |
| neck_length | REAL | 颈部长度 |
| neck_head_angle | REAL | 头颈角度 |
| neck_bent_angle | REAL | 颈部弯曲角 |
| vsl | REAL | 当时的 VSL |
| alh | REAL | 当时的 ALH |
| grade | INTEGER | 测量时的等级 |

#### 表 3: `experiment_meta`（实验元数据表）

| 列名 | 类型 | 说明 |
|------|------|------|
| key | TEXT | 键名 |
| value | TEXT | 值 |

目前存储 `start_time`（实验开始时间）。

### 8.2 数据持久化策略

| 策略 | 说明 |
|------|------|
| WAL 模式 | `PRAGMA journal_mode=WAL`，支持并发读写 |
| 批量写入 | 每 10 帧执行一次 `flush_to_db()` |
| 脏标记 | 只有被修改的记录才写入（`_dirty_ids` 集合） |
| INSERT OR REPLACE | 使用 upsert 语义，避免重复插入 |
| 追加式日志 | `morphology_history` 表只追加不修改 |

---

## 9. 代码文件详解

### 9.1 main.py — 程序入口

**文件大小**：686 B（26 行）  
**作用**：程序的启动文件，设置环境变量、创建 Qt 应用、显示主窗口。

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决 OpenMP 重复库问题

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from app import MainWindow

if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    # 自动切换到"实时分析"标签页
    if hasattr(window, 'tabWidget'):
        window.tabWidget.setCurrentIndex(1)
    sys.exit(app.exec())
```

**关键点**：
- `KMP_DUPLICATE_LIB_OK=TRUE`：解决 PyTorch 和 OpenCV 的 OpenMP 冲突
- 高 DPI 支持：确保在高分辨率屏幕上正常显示
- 自动切换到标签页索引 1（实时分析页面）

---

### 9.2 app.py — 主窗口与 UI 控制

**文件大小**：41 KB（1034 行）  
**作用**：主窗口类，管理 UI 布局、线程初始化、信号连接、用户交互。

#### 主要类

| 类 | 作用 |
|----|------|
| `FullScreenVideoLabel` | 全屏视频显示标签，支持双击/ESC 退出 |
| `StatsLabel` | 统计信息显示标签（黑底绿字） |
| `MainWindow` | 主窗口，核心控制类 |

#### MainWindow 初始化流程

```
__init__()
  ├── 设置工作目录
  ├── 加载 UI 文件 (interfacedesign.ui)
  ├── setup_window()           # 窗口尺寸、位置
  ├── configure_video_label()  # 视频标签配置
  ├── manually_set_icons()     # 设置图标
  ├── add_stats_display()      # 创建统计标签
  ├── reorganize_realtime_analysis_layout()  # 重新组织布局
  ├── start_video_thread()     # 启动视频线程
  ├── initialize_detection_thread()  # 初始化检测线程
  ├── initialize_segment_thread()    # 初始化分割线程
  ├── initialize_camera_thread()     # 初始化相机线程
  └── setup_connections()      # 连接信号和槽
```

#### 信号/槽连接

| 信号源 | 信号 | 接收方 | 槽函数 |
|--------|------|--------|--------|
| DetectionThread | stats_updated | MainWindow | update_detection_stats |
| DetectionThread | pipeline_ready | MainWindow | _on_pipeline_ready |
| SegmentThread | stats_updated | MainWindow | update_segment_stats |
| VideoTestThread | fps_changed | MainWindow | update_video_fps |
| VideoTestThread | new_frame | DetectionThread | add_frame |
| CameraThread | error_signal | MainWindow | handle_camera_error |
| CameraThread | fps_signal | MainWindow | update_camera_fps |

#### 按钮功能

| 按钮 | 方法 | 功能 |
|------|------|------|
| 开始 (pushButton) | on_start_clicked | 启动检测和分割 |
| 停止 (pushButton_3) | on_stop_clicked | 暂停检测和分割 |
| 录制 (pushButton_4) | on_record_clicked | 切换轨迹显示 |
| 清除 (pushButton_2) | on_clear_clicked | 切换视频/相机模式 |

---

### 9.3 video_thread.py — 视频采集线程

**文件大小**：11 KB  
**作用**：读取视频文件，输出帧图像到检测线程和显示标签。

#### 核心流程

```
run()
  ├── 打开视频文件 (cv2.VideoCapture)
  ├── 获取 FPS 和总帧数
  └── 循环:
      ├── 读取帧
      ├── 如果检测启用:
      │   ├── 发送 new_frame 信号（原始 BGR 帧）
      │   └── 如果 pipeline 未就绪 → 直接显示原始帧
      └── 如果检测未启用:
          └── 直接显示原始帧
```

#### Pipeline Ready 机制

这是一个防止视频闪烁的设计：

1. 用户点击"开始" → 视频线程开始向检测线程发送帧
2. 但此时分割线程还没准备好 → 如果不显示原始帧，屏幕会黑
3. 解决方案：在 pipeline 未就绪时，视频线程同时显示原始帧
4. 当分割线程处理完第一帧 → 发出 `pipeline_ready` 信号
5. 收到信号后 → 视频线程停止直接显示，改由分割线程的输出显示

---

### 9.4 camera_thread.py — 相机采集线程

**文件大小**：8 KB  
**作用**：通过 pypylon SDK 控制 Basler 工业相机，实时采集图像。

#### 相机配置

| 参数 | 值 |
|------|-----|
| 触发模式 | Off（连续采集） |
| 帧率 | 30 FPS |
| 像素格式 | Mono8（灰度） |
| ROI | 1600×1200 |
| 心跳超时 | 1000ms（GigE 相机） |

#### 信号

| 信号 | 类型 | 说明 |
|------|------|------|
| new_image_signal | np.ndarray | 新帧图像（BGR 格式） |
| error_signal | str | 错误信息 |
| fps_signal | float | 当前帧率 |

---

### 9.5 detection_thread.py — 检测线程

**文件大小**：42 KB  
**作用**：YOLOv8 推理 + JPDAF 跟踪 + 精子注册表更新 + 最优精子选择。

#### 核心循环（`run()` 方法）

```
对每一帧:
  1. 预处理: BGR→RGB, 缩放到 640×640, 归一化, CHW 转换
  2. TensorRT 推理: 拷贝到 GPU → 执行 → 拷贝回 CPU
  3. 后处理: 转置, 置信度过滤, xywh→xyxy, 缩放回原尺寸, NMS
  4. JPDAF 跟踪: predict() → correct() → get_active_tracks()
  5. 注册表更新: 对每个活跃轨迹, 匹配最近的检测框, 更新运动学参数
  6. 调度: select_for_segmentation(K=10) 选择 10 个精子供分割
  7. 最优选择: get_best_sperm(frame_w, frame_h)
  8. 数据库写入: 每 10 帧 flush_to_db()
  9. 构造 DetectionResult, 发送给分割线程
  10. 发送统计信号
```

#### DetectionResult 数据结构

```python
class DetectionResult:
    frame          # 原始帧图像
    boxes          # 检测框列表 [x1,y1,x2,y2]
    scores         # 置信度分数
    class_ids      # 类别 ID
    tracks         # 活跃轨迹列表
    best_track_id  # 最优精子的轨迹 ID
    top_candidates # 调度选出的 10 个候选 ID
    show_tracking  # 是否显示轨迹
```

---

### 9.6 segment_thread.py — 分割线程

**文件大小**：60 KB（项目中最大的文件）  
**作用**：ADSCNet 分割推理 + 形态学测量 + 精子分级 + 可视化。

#### 核心循环（`run()` 方法）

```
对每个 DetectionResult:
  1. 遍历 top_candidates（10 个候选精子）
  2. 对每个候选:
     a. 提取 64×64 ROI
     b. 预处理: albumentations Resize+Normalize
     c. TensorRT 推理
     d. 后处理: sigmoid → 阈值 0.5 → 二值 mask
     e. 计算头部形态学（椭圆拟合）
     f. 计算颈部形态学（骨架化 + 距离变换）
     g. 精子分级
     h. 更新注册表（形态学数据 + 等级）
  3. 查询最优精子: get_best_sperm()
  4. 绘制可视化结果
  5. 发送显示信号
```

#### 可视化模式

| 模式 | 条件 | 显示内容 |
|------|------|---------|
| 普通模式 | show_tracking=False | 只显示最优精子的绿色框 + 分割叠加 |
| 录制模式 | show_tracking=True | 所有候选精子的彩色框 + 轨迹 + 分割叠加 + 文字标签 |

---

### 9.7 jpdaf_tracker.py — JPDAF 跟踪器

**文件大小**：24 KB（617 行）  
**作用**：实现完整的 JPDAF 多目标跟踪算法。

#### 类结构

```
JPDATrack (单个轨迹)
  ├── 卡尔曼滤波器状态 (x, P, F, H, Q, R)
  ├── 轨迹管理 (id, age, trajectory, consecutive_misses)
  ├── 运动学计算 (vsl, alh, grade)
  └── 方法: predict(), correct(), calculate_motion_parameters()

JPDAFilter (多轨迹管理器)
  ├── 轨迹列表 (self.tracks)
  ├── 方法: predict(), correct(), validate_measurements()
  ├── 聚类: cluster_measurements()
  ├── 概率: calc_joint_prob()
  └── 管理: delete_tracks(), init_unassociate_tracks()
```

---

### 9.8 sperm_registry.py — 精子注册表

**文件大小**：25 KB（566 行）  
**作用**：管理所有精子的数据，实现综合评分、调度、最优选择。

#### SpermRecord 数据类

33 个字段，涵盖：
- 身份信息（track_id）
- 运动学原始值（vsl, alh, pos_x, pos_y）
- 检测框（box_x1~y2）
- 形态学原始值（8 个参数）
- 调度状态（测量次数、候选池标记等）
- EMA 平滑值（9 个参数）
- 投票结果（grade_history, stable_morph_grade）
- 连续评分（kinematic_continuous, morphology_continuous）

#### SpermRegistry 主要方法

| 方法 | 调用者 | 作用 |
|------|--------|------|
| update_kinematics() | DetectionThread | 更新运动学参数（每帧） |
| update_morphology() | SegmentThread | 更新形态学参数（每精子每轮） |
| select_for_segmentation() | DetectionThread | 选择 K 个精子供分割 |
| get_best_sperm() | DetectionThread/SegmentThread | 获取最优精子 |
| flush_to_db() | DetectionThread | 批量写入数据库 |
| mark_inactive() | DetectionThread | 标记消失的精子 |

---

### 9.9 modules.py — 神经网络基础模块

**文件大小**：8 KB（193 行）  
**作用**：提供各种卷积-BN-激活函数的组合模块。

| 模块 | 说明 |
|------|------|
| ConvBNAct | 标准卷积 + BN + 激活 |
| DSConvBNAct | 深度可分离卷积 |
| DWConvBNAct | 深度卷积 |
| PWConvBNAct | 逐点卷积 (1×1) |
| DeConvBNAct | 转置卷积（上采样） |
| Activation | 激活函数工厂（支持 16 种激活函数） |
| PyramidPoolingModule | 金字塔池化模块 |

---

### 9.10 model_zoo/ — 模型定义

#### adscnet.py — ADSCNet

轻量级实时分割网络，使用非对称深度可分离卷积。

| 模块 | 说明 |
|------|------|
| ADSCModule | 非对称深度可分离卷积模块 (3,1)+(1,3) |
| DDCC | 密集膨胀卷积上下文模块 |
| ADSCNet | 完整编码器-解码器网络 |

#### edanet.py — EDANet

高效密集非对称卷积网络。

| 模块 | 说明 |
|------|------|
| EDAModule | 高效密集非对称模块 |
| EDABlock | 密集连接块 |
| DownsamplingBlock | 下采样块 |
| EDANet | 完整网络 |

---

## 10. 线程架构与信号机制

### 线程间通信

```
VideoThread ──new_frame──→ DetectionThread ──detection_result──→ SegmentThread
     │                           │                                    │
     │                           │                                    │
     ▼                           ▼                                    ▼
  frame_ready               frame_ready                          frame_ready
  (显示在 UI)              (显示在 UI)                          (显示在 UI)
```

### Pipeline Ready 信号流

```
SegmentThread 处理完第一帧
    ↓
emit pipeline_ready()
    ↓
DetectionThread 转发信号
    ↓
MainWindow._on_pipeline_ready()
    ↓
VideoThread.enable_detection(True, pipeline_active=True)
    ↓
VideoThread 停止直接显示原始帧，改由 SegmentThread 输出显示
```

### 线程安全

| 保护对象 | 保护方式 | 说明 |
|---------|---------|------|
| 帧缓冲区 | threading.Lock | DetectionThread 和 SegmentThread 的 deque |
| 精子注册表 | threading.Lock | SpermRegistry 的所有读写操作 |
| QLabel 引用 | QMutex | set_output_label() 方法 |
| 数据库连接 | check_same_thread=False | SQLite 允许多线程访问 |

---

## 11. 模型文件说明

### YOLOv8 检测模型

| 文件 | 大小 | 说明 |
|------|------|------|
| `yolo_weights/best.pt` | 6 MB | PyTorch 原始训练权重 |
| `yolo_weights/best.onnx` | 6 MB | ONNX 中间格式（便于跨平台） |
| `yolo_weights/best.engine` | 16 MB | TensorRT 引擎（运行时使用） |

**转换命令**：
```bash
# PyTorch → ONNX
python yolov8_onnx2engine.py

# ONNX → TensorRT
# 在 yolov8_onnx2engine.py 中自动完成
```

### ADSCNet 分割模型

| 文件 | 大小 | 说明 |
|------|------|------|
| `seg_weights/ADSCNet_.../model.pth` | 2.8 MB | PyTorch 训练权重 |
| `seg_weights/ADSCNet_.../model.onnx` | 2.8 MB | ONNX 格式 |
| `seg_weights/ADSCNet_.../model_fp32.engine` | 3.6 MB | TensorRT FP32 引擎 |

**转换命令**：
```bash
python convert_segModel_to_tensorrt.py
```

### 模型精度

| 模型 | 精度 | 输入尺寸 | 推理时间（参考） |
|------|------|---------|----------------|
| YOLOv8 检测 | FP16 | 640×640 | ~2ms |
| ADSCNet 分割 | FP32 | 64×64 | ~0.5ms |

---

## 12. 使用指南

### 启动程序

```bash
# 激活虚拟环境
conda activate spermselectionv2

# 运行程序
python main.py
```

### 界面操作

程序启动后自动进入"实时分析"标签页。

#### 按钮说明

| 按钮 | 功能 | 使用场景 |
|------|------|---------|
| **开始** | 启动检测和分割流水线 | 开始分析视频 |
| **停止** | 暂停检测和分割 | 暂停分析 |
| **录制** | 切换轨迹可视化显示 | 需要查看精子轨迹时 |
| **清除** | 切换视频/相机模式 | 切换输入源 |

#### 操作流程

1. **启动程序** → 自动加载视频并显示
2. **点击"开始"** → 启动 AI 检测和分割
3. **观察结果** → 绿色方框标出最优精子
4. **点击"录制"** → 查看所有精子的轨迹和等级
5. **点击"停止"** → 暂停分析
6. **双击视频** → 进入全屏模式
7. **ESC 或双击** → 退出全屏

#### 统计信息

右侧面板实时显示：
- 各线程 FPS
- 缓冲区使用情况
- 检测到的精子数量
- 各等级精子数量
- 候选池大小
- 平均测量次数
- 检测/分割耗时
- 数据库路径

### 数据库查看

每次运行会在 `SpermDatabase/` 目录创建新的数据库文件：
```
SpermDatabase/sperm_experiment_20260523_154206.db
```

可以使用 `SpermDatabaseVisualization/index.html` 查看数据库内容。

---

## 13. 验证脚本说明

项目包含多个验证脚本，用于独立测试各个模块：

| 脚本 | 作用 |
|------|------|
| `val_ROI_video_tensorrt.py` | 测试 YOLOv8 检测 + 基本跟踪 |
| `val_ROI_video_JPDFAF_tensorrt.py` | 测试 YOLOv8 + JPDAF 跟踪 |
| `val_ROI_video_sperm_VSL_selection.py` | 测试 VSL 选择策略 |
| `val_ROI_video_sperm_VSLMorph_selection.py` | 测试 VSL+形态学选择策略 |
| `val_folder_seg_tensorrt.py` | 批量文件分割测试 |
| `val_video_seg_tensorrt.py` | 视频分割测试 |
| `val_video_tracking_tensorrt.py` | 视频跟踪测试 |

---

## 14. 常见问题与排查

### Q: 程序启动时报 "加载UI界面失败"

**原因**：找不到 `interfacedesign.ui` 文件  
**解决**：确保在项目根目录运行 `python main.py`

### Q: TensorRT 引擎加载失败

**原因**：TensorRT 版本不匹配或 CUDA 版本不对  
**解决**：
1. 检查 CUDA 版本：`nvcc --version`
2. 检查 TensorRT 版本：`python -c "import tensorrt; print(tensorrt.__version__)"`
3. 如果版本不匹配，需要重新转换模型：`python yolov8_onnx2engine.py`

### Q: 视频播放但没有检测结果

**原因**：没有点击"开始"按钮  
**解决**：点击"开始"按钮启动检测流水线

### Q: 检测到精子但没有绿色框

**原因**：候选池中没有满足条件的精子  
**解决**：等待足够多的形态学测量（至少 10 次），或检查视频质量

### Q: 程序运行很慢

**可能原因**：
1. GPU 显存不足 → 减少 max_detections
2. CPU 瓶颈 → 检查是否有其他程序占用 CPU
3. 视频分辨率太高 → 使用较低分辨率的视频

### Q: 数据库文件在哪里？

每次运行自动创建在 `SpermDatabase/` 目录，按时间戳命名：
```
sperm_experiment_YYYYMMDD_HHMMSS.db
```

---

## 附录 A: 关键常量速查表

| 常量 | 值 | 文件 | 说明 |
|------|-----|------|------|
| EMA_ALPHA | 0.15 | sperm_registry.py | EMA 平滑系数 |
| GRADE_VOTE_WINDOW | 20 | sperm_registry.py | 等级投票窗口 |
| GRADE_VOTE_THRESHOLD | 0.6 | sperm_registry.py | 投票采纳阈值 |
| BEST_HOLD_FRAMES | 30 | sperm_registry.py | 滞后切换帧数 |
| BEST_SWITCH_MARGIN | 0.05 | sperm_registry.py | 切换分数差距 |
| CONFIDENCE_THRESHOLD | 10 | sperm_registry.py | 候选池进入阈值 |
| K_PER_FRAME | 10 | sperm_registry.py | 每帧分割数 |
| COMPOSITE_ALPHA | 0.5 | sperm_registry.py | 运动学/形态学权重 |
| conf_threshold | 0.25 | detection_thread.py | 检测置信度阈值 |
| iou_threshold | 0.45 | detection_thread.py | NMS IoU 阈值 |
| max_detections | 150 | detection_thread.py | 最大检测数 |
| pixel_to_micron | 0.7 | 多文件 | 像素到微米转换 |

---

## 附录 B: 术语表

| 术语 | 英文 | 含义 |
|------|------|------|
| VSL | Velocity of Straight Line | 直线运动速度 |
| ALH | Amplitude of Lateral Head displacement | 头部侧向振幅 |
| JPDAF | Joint Probabilistic Data Association Filter | 联合概率数据关联滤波器 |
| EMA | Exponential Moving Average | 指数移动平均 |
| NMS | Non-Maximum Suppression | 非极大值抑制 |
| IoU | Intersection over Union | 交并比 |
| ROI | Region of Interest | 感兴趣区域 |
| TensorRT | - | NVIDIA 推理优化引擎 |
| CUDA | Compute Unified Device Architecture | NVIDIA GPU 计算架构 |
| FP16/FP32 | Float16/Float32 | 半精度/单精度浮点数 |
| BGR | Blue-Green-Red | OpenCV 默认颜色格式 |
| CHW | Channel-Height-Width | PyTorch 张量格式 |
| HWC | Height-Width-Channel | OpenCV 图像格式 |
| WAL | Write-Ahead Logging | SQLite 日志模式 |
| Upsert | Update or Insert | 存在则更新，不存在则插入 |

---

*文档版本 2.0 — 2026-05-23*
