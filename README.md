# 实时单精子优选分级系统 V2.0

## 技术文档

---

## 目录

1. [系统概述](#1-系统概述)
2. [Overview — 优选逻辑一图流](#2-overview--优选逻辑一图流)
3. [系统架构](#3-系统架构)
4. [核心模块详解](#4-核心模块详解)
5. [优选流程详解](#5-优选流程详解)
6. [平滑与稳定机制](#6-平滑与稳定机制)
7. [分级体系](#7-分级体系)
8. [调度算法](#8-调度算法)
9. [关键参数汇总](#9-关键参数汇总)
10. [数据持久化](#10-数据持久化)
11. [用户界面与操作](#11-用户界面与操作)
12. [使用指南](#12-使用指南)
13. [技术依赖](#13-技术依赖)

---

## 1. 系统概述

### 1.1 系统目标

本系统是一个基于深度学习的实时单精子优选分级系统，用于从显微镜视频中自动检测、跟踪、分析和筛选出最优精子。系统综合考虑精子的**运动学参数**（VSL、ALH）和**形态学参数**（头部形态、颈部形态），通过多维度评分实现精子的自动化优选。

### 1.2 核心能力

| 能力 | 说明 |
|------|------|
| 实时检测 | 基于 YOLOv8 的精子目标检测，TensorRT 加速推理 |
| 多目标跟踪 | JPDAF（联合概率数据关联滤波）算法，处理精子交叉、遮挡 |
| 语义分割 | ADSCNet 网络分割精子头部（顶体、细胞核）和颈部 |
| 形态学分析 | 自动测量头部长度/宽度/面积、颈部宽度/长度/角度 |
| 运动学分析 | 计算 VSL（直线速度）、ALH（侧摆幅度） |
| 智能调度 | 加权轮询算法，优先分析最有价值的精子 |
| 综合评分 | 运动学 + 形态学复合评分，选出最优精子 |
| 稳定选择 | EMA 平滑 + 等级投票 + 滞后切换，避免频繁跳动 |
| 数据持久化 | SQLite 数据库记录所有精子数据和形态学历史 |

### 1.3 应用场景

- 辅助生殖技术（ART）中的精子筛选
- 精子质量评估和分级
- 精子运动学和形态学的自动化分析

---

## 2. Overview — 优选逻辑一图流

> **一句话概括**：系统在每一帧画面里找到所有精子，持续跟踪它们的运动轨迹和形态特征，给每个精子打一个综合分数，分数最高且稳定的那个就是"最优精子"。

整个优选过程可以分为 **三步**：

```
┌──────────────────────────────────────────────────────────┐
│  第 1 步：看见 — YOLOv8 检测每一帧画面里的所有精子        │
│  每个精子获得一个检测框 (bounding box) 和置信度分数        │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  第 2 步：认识 — JPDAF 跟踪 + 运动学/形态学分析           │
│  ● 跟踪器给每个精子分配一个唯一 ID，记录运动轨迹          │
│  ● 从轨迹算出「跑得多快」(VSL) 和「摆得多大」(ALH)        │
│  ● 对精子头部和颈部进行分割，量出长宽、角度等形态指标      │
│  ● 每个精子得到两组分数：运动学分 + 形态学分              │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│  第 3 步：选优 — 综合评分 + 稳定选择                       │
│  ● 综合分 = 0.5 × 运动学分 + 0.5 × 形态学分              │
│  ● 综合分最高的精子成为「最优精子」                        │
│  ● 为防止选中结果频繁跳动，系统使用：                      │
│    - EMA 平滑：让各项指标不会因单帧抖动而剧烈变化          │
│    - 等级投票：形态学等级取最近 20 次测量的多数结果        │
│    - 滞后切换：新精子必须连续 30 帧都优于当前最优才替换    │
│    - 边缘检测：最优精子的检测框碰到画面边缘时立即切换      │
└──────────────────────────────────────────────────────────┘
```

**通俗理解**：就像老师在操场上观察一群跑步的学生——先看到所有人（检测），然后记住每个人是谁、跑得多快、姿势怎么样（跟踪+分析），最后选出跑得最快、姿势最好的那个（优选）。为了不让评选结果每秒钟变来变去，需要连续观察一段时间、综合多次打分才能最终确定。

---

## 3. 系统架构

### 3.1 多线程架构

系统采用 PyQt6 多线程架构，各模块通过 Qt 信号机制通信：

```
┌─────────────────────────────────────────────────────────────────┐
│                        MainWindow (GUI)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ 视频显示  │  │ 分割显示  │  │ 统计信息  │  │ 用户控制按钮  │   │
│  └────┬─────┘  └────▲─────┘  └────▲─────┘  └──────┬───────┘   │
└───────┼─────────────┼─────────────┼────────────────┼───────────┘
        │             │             │                │
        ▼             │             │                │
┌───────────────┐     │             │                │
│ VideoThread   │     │             │                │
│ (视频采集)     │     │             │                │
└───────┬───────┘     │             │                │
        │ new_frame   │             │                │
        ▼             │             │                │
┌───────────────┐     │             │                │
│DetectionThread│     │             │                │
│ (YOLOv8检测)  │     │             │                │
│ + JPDAF跟踪   │     │             │                │
│ + 运动学分级   │     │             │                │
└───────┬───────┘     │             │                │
        │ detection_result          │                │
        ▼             │             │                │
┌───────────────┐     │             │                │
│ SegmentThread │─────┘             │                │
│ (语义分割)     │                   │                │
│ + 形态学分析   │                   │                │
│ + 形态学分级   │                   │                │
└───────┬───────┘                   │                │
        │ writes to                 │                │
        ▼                           │                │
┌───────────────┐                   │                │
│ SpermRegistry │───────────────────┘                │
│ (精子注册表)   │                                    │
│ + 调度算法     │                                    │
│ + 候选池管理   │                                    │
│ + SQLite持久化 │                                    │
└───────────────┘                                    │
```

### 3.2 数据流

```
视频/相机输入
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 检测线程 (DetectionThread)                                   │
│  1. YOLOv8 检测 → 边界框 + 置信度                            │
│  2. JPDAF 跟踪 → 轨迹 ID + 位置预测                          │
│  3. 运动学计算 → VSL, ALH                                    │
│  4. 运动学分级 → Grade 1-6                                   │
│  5. 调度请求 → 选择 K=10 个精子进行分割                       │
│  6. 输出: DetectionResult (frame, tracks, top_candidates)    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 分割线程 (SegmentThread)                                     │
│  1. ROI 提取 → 以精子位置为中心的 64×64 区域                   │
│  2. ADSCNet 分割 → 4 类概率图 (顶体/细胞核/不可测头/颈部)      │
│  3. 头部形态学 → 长度/宽度/长宽比/面积                         │
│  4. 颈部形态学 → 宽度/长度/弯曲角度/头颈角度                   │
│  5. 形态学分级 → Grade 4/5/-1                                │
│  6. 更新注册表 → 写入形态学数据                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 精子注册表 (SpermRegistry)                                   │
│  1. EMA 平滑 → 平滑运动学和形态学参数                         │
│  2. 等级投票 → 稳定形态学等级                                 │
│  3. 候选池管理 → 测量次数 ≥ 10 进入候选池                     │
│  4. 复合评分 → 0.5×运动学 + 0.5×形态学                       │
│  5. 最优选择 → 候选池中复合分最高 + 滞后稳定的精子            │
│  6. 边缘检测 → 检测框触碰边缘时立即切换                       │
│  7. SQLite 持久化 → 每 10 帧批量写入                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 核心模块详解

### 4.1 检测线程 (DetectionThread)

**文件**: `detection_thread.py`

#### 4.1.1 YOLOv8 检测引擎

| 参数 | 值 | 说明 |
|------|-----|------|
| 引擎文件 | `yolo_weights/best.engine` | TensorRT 格式 |
| 输入尺寸 | 640×640×3 | RGB 图像，float16 |
| 置信度阈值 | 0.25 | 低于此值的检测结果被过滤 |
| NMS IoU 阈值 | 0.45 | 非极大值抑制的重叠阈值 |
| 最大检测数 | 150 | 每帧最多检测 150 个精子 |
| 类别 | `['sperm']` | 单类别检测 |

#### 4.1.2 检测流程

```
输入帧 → 预处理 (BGR→RGB, 缩放640×640, 归一化, CHW)
    → TensorRT 推理 (异步CUDA流)
    → 后处理 (转置, 置信度过滤, xywh→xyxy, 缩放原尺寸, NMS)
    → 输出: 边界框坐标 (xyxy格式) + 中心点
```

#### 4.1.3 JPDAF 跟踪器

跟踪器使用联合概率数据关联滤波（JPDAF）算法，核心参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| 过程噪声 | 20.0 | 卡尔曼滤波器过程噪声协方差 |
| 测量噪声 | 2.0 | 卡尔曼滤波器测量噪声协方差 |
| 检测概率 | 0.7 | 目标被检测到的概率 |
| 门限概率 | 0.95 | 马氏距离门限概率 |
| 状态向量 | [x, y, vx, vy] | 4 维：位置 + 速度 |
| 测量向量 | [x, y] | 2 维：仅位置 |
| 最小激活帧数 | 5 | 跟踪帧数 ≥ 5 才算活跃 |
| 最大连续丢失 | 10 | 连续 10 帧未匹配则删除轨迹 |

#### 4.1.4 运动学参数计算

**VSL（直线速度, Straight Line Velocity）**:
- 使用最近 30 个轨迹点（最少需要 5 个）
- 计算首尾点的欧氏距离，转换为微米
- 除以时间跨度得到速度（μm/s）

**ALH（侧摆幅度, Amplitude of Lateral Head Displacement）**:
- 对最近轨迹点进行最小二乘直线拟合
- 计算每个点到拟合直线的垂直距离
- ALH = 平均垂直距离 × 2（表示全振幅）

#### 4.1.5 运动学分级

| 等级 | VSL (μm/s) | ALH (μm) | 说明 |
|------|-----------|----------|------|
| Grade 1 | ≥ 13.5 | ≤ 0.6 | 优秀：快速直线运动 |
| Grade 2 | ≥ 10.0 | ≤ 0.8 | 良好：较快且稳定 |
| Grade 3 | ≥ 7.5 | ≤ 1.0 | 一般：中等速度 |
| Grade 4 | ≥ 5.0 | - | 较慢 |
| Grade 5 | ≥ 2.0 | - | 慢 |
| Grade 6 | < 2.0 | - | 极慢或不动 |

### 4.2 分割线程 (SegmentThread)

**文件**: `segment_thread.py`

#### 4.2.1 分割引擎

| 参数 | 值 | 说明 |
|------|-----|------|
| 引擎文件 | `seg_weights/ADSCNet_sperm_ROINAHead_250707/model_fp32.engine` | TensorRT 格式 |
| 网络架构 | ADSCNet | 非对称深度可分离卷积网络 |
| 输入尺寸 | 64×64×3 | ROI 裁剪区域 |
| 输出类别 | 4 类 | 语义分割 |
| 概率阈值 | 0.5 | 二值化阈值 |

#### 4.2.2 分割类别定义

| 类别索引 | 名称 | 说明 |
|----------|------|------|
| 0 | 顶体 (Acrosome) | 精子头部前端 |
| 1 | 细胞核 (Nucleus) | 精子头部主体 |
| 2 | 不可测头 (Non-measurable Head) | 异常头部 |
| 3 | 颈部 (Neck) | 连接头部和尾部的区域 |

#### 4.2.3 头部形态学计算

`_calculate_head_morphology(masks)`:

1. **掩码合并**:
   - 正常头部：合并类别 0（顶体）+ 类别 1（细胞核）
   - 异常头部：使用类别 2（不可测头）

2. **轮廓提取**: 在合并掩码中找到最大轮廓

3. **椭圆拟合**: 对轮廓拟合椭圆（需要 ≥ 5 个轮廓点）

4. **参数提取**:
   - `head_length`：椭圆长轴长度（μm）
   - `head_width`：椭圆短轴长度（μm）
   - `head_ratio`：长宽比 = 长度 / 宽度
   - `head_area`：轮廓面积（μm²）

5. **单位转换**: 像素 → 微米，转换系数 0.7

#### 4.2.4 颈部形态学计算

`_calculate_neck_morphology(masks)`:

1. **掩码提取**: 使用类别 3（颈部）的二值掩码

2. **骨架化**: 使用 Guo-Hall 细化算法 (`cv2.ximgproc.thinning`) 提取中心线

3. **骨架点排序**: 最近邻方法连接骨架点（最大间距 5 像素）

4. **颈部长度**: 沿骨架点的距离总和 × 转换系数

5. **颈部宽度**:
   - 计算距离变换 (Distance Transform)
   - 沿骨架采样宽度（每 3 个点采样一次）
   - 取第 95 百分位数 × 2 × 转换系数

6. **颈部弯曲角度**:
   - 将骨架分成若干段
   - 计算相邻段的方向向量
   - 计算方向变化角度
   - 取最大角度变化

7. **头颈角度**:
   - 拟合头部椭圆获取主轴方向
   - 计算颈部起始段方向
   - 计算两方向的夹角（限制在 0-90°）

### 4.3 精子注册表 (SpermRegistry)

**文件**: `sperm_registry.py`

#### 4.3.1 数据结构

每个被跟踪的精子维护一个 `SpermRecord`：

```python
SpermRecord:
    # 原始运动学参数（每帧更新）
    track_id: int           # 轨迹 ID
    vsl: float              # 直线速度 (μm/s)
    alh: float              # 侧摆幅度 (μm)
    kinematic_grade: int    # 运动学等级 (1-6)
    pos_x: float            # 当前 X 坐标
    pos_y: float            # 当前 Y 坐标

    # 检测框 (xyxy 格式, 每帧更新)
    box_x1: float           # 左上角 X
    box_y1: float           # 左上角 Y
    box_x2: float           # 右下角 X
    box_y2: float           # 右下角 Y

    # 原始形态学参数（分割后更新）
    head_length: float      # 头部长度 (μm)
    head_width: float       # 头部宽度 (μm)
    head_ratio: float       # 头部长宽比
    head_area: float        # 头部面积 (μm²)
    neck_width: float       # 颈部宽度 (μm)
    neck_length: float      # 颈部长度 (μm)
    neck_head_angle: float  # 头颈角度 (°)
    neck_bent_angle: float  # 颈部弯曲角度 (°)
    morphology_grade: int   # 形态学等级 (4/5/-1)

    # 平滑后的参数 (EMA 指数移动平均)
    vsl_smooth: float       # 平滑直线速度
    alh_smooth: float       # 平滑侧摆幅度
    head_length_smooth: float
    head_width_smooth: float
    head_ratio_smooth: float
    head_area_smooth: float
    neck_width_smooth: float
    neck_bent_angle_smooth: float
    neck_head_angle_smooth: float

    # 等级投票
    grade_history: list     # 最近 20 次形态学等级记录
    stable_morph_grade: int # 投票后的稳定形态学等级

    # 连续评分
    kinematic_continuous: float   # 连续运动学分 (0-1)
    morphology_continuous: float  # 连续形态学分 (0-1)

    # 调度状态
    morphology_measurement_count: int  # 形态学测量次数
    last_measurement_frame: int        # 最后测量帧号
    in_candidate_pool: bool            # 是否进入候选池
    scheduling_score: float            # 调度得分
    composite_score: float             # 复合评分
```

#### 4.3.2 关键常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `CONFIDENCE_THRESHOLD` | 10 | 进入候选池的最小测量次数 |
| `K_PER_FRAME` | 10 | 每帧分割的精子数 |
| `FRESHNESS_INTERVAL` | 30 | 重新测量间隔（帧） |
| `W_MORPH_COUNT` | 0.4 | 调度权重：测量需求 |
| `W_VSL` | 0.3 | 调度权重：VSL 速度 |
| `W_EDGE_DISTANCE` | 0.3 | 调度权重：边缘距离 |
| `VSL_NORMALIZATION` | 30.0 | VSL 归一化因子 (μm/s) |
| `EDGE_NORMALIZATION` | 200.0 | 边缘距离归一化因子 (px) |
| `COMPOSITE_ALPHA` | 0.5 | 运动学 vs 形态学平衡系数 |
| `POOL_PENALTY` | 0.3 | 已达标精子的调度降权系数 |
| `DB_FLUSH_INTERVAL` | 10 | 数据库刷新间隔（帧） |
| `EMA_ALPHA` | 0.15 | EMA 平滑系数 (越小越平滑) |
| `GRADE_VOTE_WINDOW` | 20 | 等级投票窗口大小 |
| `GRADE_VOTE_THRESHOLD` | 0.6 | 投票阈值 (60% 以上一致才采纳) |
| `BEST_HOLD_FRAMES` | 30 | 最佳精子最小保持帧数 |
| `BEST_SWITCH_MARGIN` | 0.05 | 切换所需超越幅度 |

---

## 5. 优选流程详解

### 5.1 整体流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                     视频/相机输入帧                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: YOLOv8 检测                                             │
│  • 输入: 640×640 RGB 图像                                        │
│  • 输出: 边界框(xyxy) + 置信度                                    │
│  • 过滤: conf > 0.25, NMS IoU = 0.45                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: JPDAF 跟踪                                              │
│  • 预测: 卡尔曼滤波预测所有轨迹位置                                │
│  • 匹配: 马氏距离门限 + 联合概率数据关联                          │
│  • 更新: 加权修正轨迹状态                                         │
│  • 匹配检测框: 将最近的检测框关联到每个轨迹                       │
│  • 删除: 连续丢失 10 帧或门限椭圆过大则删除                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: 运动学参数计算 (EMA 平滑)                                │
│  • VSL_raw = 直线距离 / 时间跨度 (最近30点)                      │
│  • ALH_raw = 平均垂直位移 × 2 (相对拟合直线)                     │
│  • VSL_smooth = 0.15 × VSL_raw + 0.85 × VSL_smooth_prev        │
│  • ALH_smooth = 0.15 × ALH_raw + 0.85 × ALH_smooth_prev        │
│  • 连续运动学分 = 0.7×(VSL_smooth/30) + 0.3×(1-ALH_smooth/2)   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: 加权调度                                                 │
│  • 计算每个精子的调度得分                                         │
│  • score = 0.4×测量需求 + 0.3×VSL + 0.3×边缘距离                 │
│  • 已达标精子降权 (×0.3), 长时间未测量提升 (×2.0)                 │
│  • 选择 Top K=10 个精子进行分割                                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: 对每个调度精子进行分割                                    │
│  5a. ROI 提取: 以精子位置为中心的 64×64 区域                      │
│  5b. ADSCNet 分割: 4 类概率图 → 二值掩码                         │
│  5c. 头部形态学: 椭圆拟合 → 长度/宽度/长宽比/面积                 │
│  5d. 颈部形态学: 骨架化 → 宽度/长度/弯曲角度/头颈角度             │
│  5e. 形态学参数 EMA 平滑                                         │
│  5f. 形态学等级投票 (最近20次取多数)                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 6: 候选池管理                                               │
│  • 测量次数 ≥ 10 → 进入候选池                                    │
│  • 复合评分 = 0.5×运动学连续分 + 0.5×形态学分×置信度             │
│  • 候选池按复合分降序排列                                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 7: 最优精子选择 (带稳定机制)                                │
│  • 检查当前最优精子的检测框是否触碰画面边缘                       │
│    → 触碰边缘: 立即切换到非边缘的最佳候选                         │
│  • 检查是否有新的挑战者复合分更高                                 │
│    → 新精子必须连续 30 帧超越当前最优 (幅度 > 0.05) 才替换        │
│  • 最终输出: 最优精子的检测框 + 分割叠加                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 8: 数据持久化                                               │
│  • 每 10 帧批量写入 SQLite 数据库                                 │
│  • 存储: 精子记录 + 形态学历史 + 实验元数据                       │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 详细步骤说明

#### Step 1: 视频帧采集

- **视频模式**: 从 `test-videoes/` 目录读取视频文件，按原始帧率播放
- **相机模式**: 通过 Basler pypylon SDK 采集实时图像，30 FPS
- **缓冲区**: 检测线程帧缓冲区最大 100 帧

#### Step 2: YOLOv8 目标检测

```
输入帧 (原始尺寸)
    → BGR 转 RGB
    → 缩放到 640×640
    → 归一化到 [0, 1]
    → 转换为 CHW 格式 (float16)
    → 添加 batch 维度
    → TensorRT 异步推理
    → 输出: (1, 5, 8400) 格式
    → 转置: (8400, 5)
    → 提取: boxes[:, :4], scores[:, 4:]
    → 置信度过滤: > 0.25
    → 坐标转换: xywh → xyxy
    → 缩放回原图尺寸
    → NMS: IoU = 0.45
    → 输出: 边界框 (xyxy) + 中心点
```

#### Step 3: JPDAF 多目标跟踪

**预测阶段**:
```
对每个已有轨迹:
    x_pred = F × x_post  (状态转移预测)
    P_pred = F × P_post × F^T + Q  (协方差预测)
```

**匹配阶段**:
```
1. 计算马氏距离矩阵 (轨迹 vs 检测)
2. 门限过滤: 距离 < GateThreshold
3. 聚类: 将关联的轨迹和检测分组
4. 枚举所有可行联合关联事件
5. 计算联合概率
6. 计算边缘关联概率 (betta)
```

**检测框匹配**:
```
对每个活跃轨迹:
    1. 取轨迹最新位置 (tx, ty)
    2. 遍历所有检测框，计算框中心到轨迹位置的距离
    3. 选择最近的检测框作为该轨迹的 bbox
    4. 将 bbox 传入精子注册表用于边缘检测
```

**更新阶段**:
```
对每个轨迹:
    x_post = x_pred + K × Σ(betta_i × (z_i - H × x_pred))
    其中 K 为卡尔曼增益, betta 为边缘关联概率
```

**轨迹管理**:
- 新建: 未关联的检测创建新轨迹
- 删除: 连续丢失 > 10 帧 或 门限椭圆过大
- 激活: 跟踪帧数 ≥ 5

#### Step 4: 运动学参数计算

**VSL 计算**:
```python
# 使用最近 30 个轨迹点
points = trajectory[-30:]
if len(points) >= 5:
    distance = euclidean(points[-1], points[0]) * pixel_to_micron
    time_span = times[-1] - times[0]
    vsl = distance / time_span  # μm/s
```

**ALH 计算**:
```python
# 最小二乘拟合直线
k, b = least_squares_fit(points)

# 计算垂直距离
for point in points:
    perpendicular_dist = abs(k * point.x - point.y + b) / sqrt(k² + 1)
    distances.append(perpendicular_dist)

alh = mean(distances) * pixel_to_micron * 2  # μm
```

**EMA 平滑** (详见 [第 6 章](#6-平滑与稳定机制)):
```python
vsl_smooth = 0.15 × vsl + 0.85 × vsl_smooth_prev
alh_smooth = 0.15 × alh + 0.85 × alh_smooth_prev
```

**运动学分级**:
```python
if vsl >= 13.5 and alh <= 0.6:
    grade = 1
elif vsl >= 10.0 and alh <= 0.8:
    grade = 2
elif vsl >= 7.5 and alh <= 1.0:
    grade = 3
elif vsl >= 5.0:
    grade = 4
elif vsl >= 2.0:
    grade = 5
else:
    grade = 6
```

#### Step 5: 加权调度

**调度得分计算**:
```python
morph_need = 1.0 - min(measurement_count / 10, 1.0)  # 测量需求
vsl_norm = min(vsl / 30.0, 1.0)                      # VSL 归一化
edge_norm = min(edge_distance / 200.0, 1.0)           # 边缘距离归一化

score = 0.4 * morph_need + 0.3 * vsl_norm + 0.3 * edge_norm

# 已达标精子降权
if in_candidate_pool:
    score *= 0.3
    # 长时间未测量则提升
    if (current_frame - last_measurement_frame) > 30:
        score *= 2.0
```

**调度策略**:
1. **测量需求优先**: 新出现或测量次数不足的精子优先级更高
2. **速度偏好**: VSL 更快的精子优先级更高
3. **位置偏好**: 远离边缘的精子优先级更高
4. **公平性**: 已达标精子降权，但长时间未测量会重新提升

#### Step 6: TensorRT 语义分割

**ROI 提取**:
```python
center_x, center_y = track.trajectory[-1]
half_size = 32  # 64/2
x1 = max(0, center_x - half_size)
y1 = max(0, center_y - half_size)
x2 = min(frame_width, center_x + half_size)
y2 = min(frame_height, center_y + half_size)
roi = frame[y1:y2, x1:x2]
# 如果 ROI 不足 64×64，用零填充
```

**分割流程**:
```
ROI (64×64×3)
    → 缩放到 64×64
    → 归一化
    → TensorRT 推理
    → 输出: (1, 4, 64, 64) 概率图
    → Sigmoid 激活
    → 阈值 0.5 二值化
    → 4 个二值掩码
```

#### Step 7: 形态学参数计算

**头部形态学**:
```python
# 合并头部掩码
if has_acrosome and has_nucleus:
    head_mask = acrosome_mask OR nucleus_mask
elif has_nonmeasurable:
    head_mask = nonmeasurable_mask

# 椭圆拟合
contours = findContours(head_mask)
largest_contour = max(contours, key=area)
ellipse = fitEllipse(largest_contour)

# 参数提取
head_length = max(ellipse.axes) * pixel_to_micron  # μm
head_width = min(ellipse.axes) * pixel_to_micron   # μm
head_ratio = head_length / head_width
head_area = contourArea(largest_contour) * pixel_to_micron²  # μm²
```

**颈部形态学**:
```python
# 骨架化
skeleton = thinning(neck_mask, GUOHALL)
ordered_points = nearest_neighbor_chain(skeleton_points)

# 颈部长度
neck_length = sum(distances(ordered_points)) * pixel_to_micron

# 颈部宽度 (距离变换)
dist_transform = distanceTransform(neck_mask)
widths = [2 * dist_transform[point] for point in ordered_points[::3]]
neck_width = percentile(widths, 95) * pixel_to_micron

# 颈部弯曲角度
segments = split_into_segments(ordered_points)
angles = [angle_between(segments[i], segments[i+1]) for i in range(len(segments)-1)]
neck_bent_angle = max(angles)

# 头颈角度
head_direction = ellipse_major_axis_direction
neck_direction = ordered_points[5] - ordered_points[0]
neck_head_angle = angle_between(head_direction, neck_direction)
neck_head_angle = min(neck_head_angle, 180 - neck_head_angle)  # 限制在 0-90°
```

#### Step 8: 形态学分级

**当前生效的分级逻辑**:

**Grade 4** (需满足所有条件):
```
VSL ≥ 11.0 μm/s
AND 1.5 ≤ head_ratio ≤ 2.2
AND 0.5 < neck_width ≤ 1.0 μm
AND 20.0 < neck_bent_angle ≤ 50.0°
AND 10.0 < neck_head_angle ≤ 30.0°
AND neck_bent_angle ≥ 20.0°
AND neck_head_angle ≥ 10.0°
AND 10.0 < head_area ≤ 90.0 μm²
```

**Grade 5** (需满足所有条件):
```
0 < neck_bent_angle < 34.0°
AND 0 < neck_head_angle < 34.0°
AND 10.0 < head_area ≤ 110.0 μm²
AND 5.0 ≤ VSL ≤ 80.0 μm/s
排除: 特定异常 VSL 范围 (78.0-78.3, 30.4-32.5, 5.0-5.5)
```

**Grade -1** (未分级): 不满足上述条件的所有情况

#### Step 9: 候选池管理与最优精子选择

**进入候选池条件**:
```python
if morphology_measurement_count >= 10:
    in_candidate_pool = True
```

**复合评分**:
```python
# 运动学: 使用连续分数 (基于 EMA 平滑后的 VSL/ALH)
kinematic_score = 0.7 × min(vsl_smooth / 30.0, 1.0) + 0.3 × max(0, 1 - alh_smooth / 2.0)

# 形态学: 使用投票后的稳定等级
morphology_score = (6 - stable_morph_grade) / 5.0

# 测量置信度: 测量越多，形态学权重越高
confidence = min(measurement_count / 20.0, 1.0)

# 复合分
composite_score = 0.5 × kinematic_score + 0.5 × morphology_score × confidence
```

**最优精子选择** (带稳定机制):
```python
# 1. 检查当前最优精子是否到达边缘
if current_best.bbox_touches_frame_edge:
    switch_to(non_edge_candidate)  # 立即切换

# 2. 检查是否有更优的挑战者
if challenger.composite_score > current_best.composite_score + 0.05:
    hold_count += 1
    if hold_count >= 30:  # 连续 30 帧都更优
        switch_to(challenger)

# 3. 否则保持当前最优
return current_best
```

#### Step 10: 结果输出

- 在视频帧上绘制最优精子的绿色边界框
- 叠加分割结果（彩色掩码，透明度 0.5）
- 实时显示统计信息（FPS、缓冲区、分级分布等）

---

## 6. 平滑与稳定机制

优选系统的核心挑战之一是：**如何让选中的最优精子保持稳定，不会因单帧数据抖动而频繁跳变？** 系统采用了四层稳定机制：

### 6.1 指数移动平均 (EMA) — 参数平滑

**目的**: 让 VSL、ALH 以及各项形态学参数不会因单次测量的噪声而剧烈波动。

**原理**: 每次新测量值只占 15% 权重，历史值占 85% 权重。

```python
EMA_ALPHA = 0.15

# 运动学平滑
vsl_smooth = 0.15 × vsl_new + 0.85 × vsl_smooth_prev
alh_smooth = 0.15 × alh_new + 0.85 × alh_smooth_prev

# 形态学平滑 (头部、颈部各项参数同理)
head_length_smooth = 0.15 × head_length_new + 0.85 × head_length_smooth_prev
neck_bent_angle_smooth = 0.15 × neck_bent_new + 0.85 × neck_bent_prev
# ... 其余参数类推
```

**效果**: 参数变化曲线变得平滑，单帧异常值被大幅削弱。

```
原始数据:   10 → 12 → 8 → 15 → 11 → 13   (剧烈波动)
EMA 平滑后: 10 → 10.3 → 9.9 → 10.7 → 10.7 → 11.1  (平滑过渡)
```

### 6.2 等级投票 — 形态学等级稳定

**目的**: 形态学分级 (Grade 4/5/-1) 可能因分割结果微小差异而在相邻等级间跳动。投票机制确保等级变化需要多数一致。

**原理**: 保存最近 20 次形态学分级结果，只有当某个等级占比超过 60% 时才采纳。

```python
GRADE_VOTE_WINDOW = 20    # 投票窗口
GRADE_VOTE_THRESHOLD = 0.6  # 采纳阈值

def vote_grade(history):
    valid = [g for g in history if g > 0]  # 过滤掉 -1
    counts = Counter(valid)
    top_grade, top_count = counts.most_common(1)[0]
    if top_count / len(valid) >= 0.6:
        return top_grade    # 多数一致，采纳
    return -1               # 没有明确多数，保持未分级
```

**效果**:
- 偶尔一两次分级错误不会影响最终结果
- 等级变化需要至少 60% 的一致性，大幅减少跳变
- 只有持续稳定的形态学特征才能获得明确的等级

### 6.3 滞后切换 — 最优精子防抖

**目的**: 当另一个精子的复合分开始超过当前最优精子时，不立即切换，而是等待确认。

**原理**: 挑战者必须**连续 30 帧**都以超过 0.05 分的幅度超越当前最优精子，才会触发切换。

```python
BEST_HOLD_FRAMES = 30      # 最小保持帧数
BEST_SWITCH_MARGIN = 0.05  # 超越幅度阈值

def get_best_sperm():
    top = candidate_pool[0]  # 当前最高分

    if top.score > current_best.score + 0.05:
        hold_count += 1
        if hold_count >= 30:
            # 连续 30 帧都更优，确认切换
            switch_to(top)
    else:
        hold_count = 0  # 中间断了，重新计数

    return current_best  # 保持当前
```

**效果**:
- 短暂的分数波动不会导致切换
- 只有真正持续更优的精子才会被选中
- 避免了"乒乓球效应"（两个精子交替成为最优）

### 6.4 边缘检测 — 检测框触碰边界时立即切换

**目的**: 当最优精子游到画面边缘、检测框碰到边界时，说明这个精子即将离开视野，需要立即切换到下一个候选。

**原理**: 检查最优精子的检测框 (bounding box) 是否触碰画面四条边中的任意一条。

```python
def is_bbox_at_edge(rec, frame_w, frame_h):
    """检测框是否触碰画面边缘"""
    # bbox 用 xyxy 格式: (x1, y1, x2, y2)
    return (rec.box_x1 <= 0 or       # 左边碰到
            rec.box_y1 <= 0 or       # 上边碰到
            rec.box_x2 >= frame_w or # 右边碰到
            rec.box_y2 >= frame_h)   # 下边碰到

def get_best_sperm(frame_w, frame_h):
    if is_bbox_at_edge(current_best, frame_w, frame_h):
        # 立即从候选池中选一个不在边缘的精子
        non_edge = [r for r in pool if not is_bbox_at_edge(r)]
        if non_edge:
            return non_edge[0]  # 立即切换
```

**效果**:
- 最优精子到达画面边缘时立即被替换
- 不需要等待 30 帧的滞后确认（边缘情况优先级最高）
- 检测框比中心点更准确地反映精子是否接近边界

### 6.5 四层机制协同工作

```
                    ┌─────────────────────────────────┐
                    │         原始测量数据              │
                    └────────────┬────────────────────┘
                                 ▼
                    ┌─────────────────────────────────┐
                    │  第 1 层: EMA 平滑               │
                    │  消除单帧噪声，让参数曲线平滑      │
                    └────────────┬────────────────────┘
                                 ▼
                    ┌─────────────────────────────────┐
                    │  第 2 层: 等级投票               │
                    │  稳定形态学等级，消除分类跳变      │
                    └────────────┬────────────────────┘
                                 ▼
                    ┌─────────────────────────────────┐
                    │  第 3 层: 滞后切换               │
                    │  新精子需连续 30 帧更优才替换     │
                    └────────────┬────────────────────┘
                                 ▼
                    ┌─────────────────────────────────┐
                    │  第 4 层: 边缘检测               │
                    │  检测框碰边 → 立即切换 (最高优先) │
                    └─────────────────────────────────┘
```

---

## 7. 分级体系

### 7.1 运动学分级

运动学分级基于 VSL（直线速度）和 ALH（侧摆幅度）：

| 等级 | VSL (μm/s) | ALH (μm) | 临床意义 |
|------|-----------|----------|----------|
| Grade 1 | ≥ 13.5 | ≤ 0.6 | 优秀：快速前向运动，轨迹稳定 |
| Grade 2 | ≥ 10.0 | ≤ 0.8 | 良好：较快运动，略有摆动 |
| Grade 3 | ≥ 7.5 | ≤ 1.0 | 一般：中等速度，摆动明显 |
| Grade 4 | ≥ 5.0 | - | 较慢：速度较低 |
| Grade 5 | ≥ 2.0 | - | 慢：速度很低 |
| Grade 6 | < 2.0 | - | 极慢/不动 |

### 7.2 形态学分级

形态学分级基于头部和颈部的形态学参数：

| 等级 | 条件 | 说明 |
|------|------|------|
| Grade 4 | VSL≥11, head_ratio∈[1.5,2.2], neck_width∈(0.5,1.0], neck_bent∈(20,50], neck_head∈(10,30], head_area∈(10,90] | 形态正常，颈部有适度弯曲 |
| Grade 5 | neck_bent∈(0,34), neck_head∈(0,34), head_area∈(10,110], VSL∈[5,80] | 形态基本正常，参数较宽松 |
| Grade -1 | 不满足上述条件 | 形态异常或无法分级 |

### 7.3 复合评分

复合评分综合运动学和形态学两个维度：

```
composite_score = 0.5 × kinematic_continuous + 0.5 × morphology_score × confidence

其中:
  kinematic_continuous = 0.7 × min(vsl_smooth / 30.0, 1.0) + 0.3 × max(0, 1 - alh_smooth / 2.0)
  morphology_score = (6 - stable_morph_grade) / 5.0
  confidence = min(measurement_count / 20.0, 1.0)
```

**评分范围**:
- 最高分: 1.0 (运动学优秀 + 形态学 Grade 1 + 充分测量)
- 最低分: 0.0 (运动学极差 + 形态学 Grade -1)

**示例**:
| 运动学等级 | 形态学等级 | 测量次数 | 运动学分 | 形态学分 | 置信度 | 复合分 |
|-----------|-----------|---------|---------|---------|--------|--------|
| 1 | 4 | 20 | 1.0 | 0.4 | 1.0 | 0.70 |
| 2 | 5 | 15 | 0.8 | 0.2 | 0.75 | 0.48 |
| 3 | -1 | 10 | 0.6 | 0.0 | 0.5 | 0.30 |
| 1 | 5 | 20 | 1.0 | 0.2 | 1.0 | 0.60 |

---

## 8. 调度算法

### 8.1 调度目标

在每帧处理中，系统需要从所有活跃精子中选择 K=10 个进行形态学分割。调度算法的目标是：

1. **公平性**: 确保每个精子都能获得足够的测量次数
2. **效率**: 优先分析有潜力的精子（速度快、远离边缘）
3. **新鲜度**: 定期重新测量已达标精子，确保数据准确

### 8.2 调度得分计算

```python
def compute_scheduling_score(sperm, current_frame):
    # 1. 测量需求 (权重 0.4)
    morph_need = 1.0 - min(sperm.measurement_count / 10, 1.0)

    # 2. VSL 速度 (权重 0.3)
    vsl_norm = min(sperm.vsl / 30.0, 1.0)

    # 3. 边缘距离 (权重 0.3)
    edge_distance = min(sperm.pos_x, sperm.pos_y,
                       frame_width - sperm.pos_x,
                       frame_height - sperm.pos_y)
    edge_norm = min(edge_distance / 200.0, 1.0)

    # 综合得分
    score = 0.4 * morph_need + 0.3 * vsl_norm + 0.3 * edge_norm

    # 已达标精子降权
    if sperm.in_candidate_pool:
        score *= 0.3
        # 长时间未测量则提升
        if (current_frame - sperm.last_measurement_frame) > 30:
            score *= 2.0

    return score
```

### 8.3 调度流程

```
每帧处理:
1. 遍历所有活跃精子
2. 计算每个精子的调度得分
3. 按得分降序排列
4. 选择 Top K=10 个精子
5. 返回这 10 个精子的 track_id 列表
```

---

## 9. 关键参数汇总

### 9.1 检测参数

| 参数 | 值 | 文件位置 |
|------|-----|----------|
| YOLOv8 置信度阈值 | 0.25 | detection_thread.py |
| NMS IoU 阈值 | 0.45 | detection_thread.py |
| 最大检测数 | 150 | detection_thread.py |
| YOLOv8 输入尺寸 | 640×640 | detection_thread.py |
| 像素到微米转换系数 | 0.7 | detection_thread.py |

### 9.2 跟踪参数

| 参数 | 值 | 文件位置 |
|------|-----|----------|
| 过程噪声 | 20.0 | detection_thread.py |
| 测量噪声 | 2.0 | detection_thread.py |
| 检测概率 | 0.7 | detection_thread.py |
| 门限概率 | 0.95 | detection_thread.py |
| 最小激活帧数 | 5 | jpdaf_tracker.py |
| 最大连续丢失 | 10 | jpdaf_tracker.py |

### 9.3 分割参数

| 参数 | 值 | 文件位置 |
|------|-----|----------|
| 分割输入尺寸 | 64×64 | segment_thread.py |
| 分割类别数 | 4 | segment_thread.py |
| 概率阈值 | 0.5 | segment_thread.py |

### 9.4 调度参数

| 参数 | 值 | 文件位置 |
|------|-----|----------|
| 每帧分割精子数 K | 10 | sperm_registry.py |
| 候选池阈值 | 10 次测量 | sperm_registry.py |
| 新鲜度间隔 | 30 帧 | sperm_registry.py |
| 调度权重 (测量/VSL/边缘) | 0.4/0.3/0.3 | sperm_registry.py |
| VSL 归一化因子 | 30.0 μm/s | sperm_registry.py |
| 边缘归一化因子 | 200.0 px | sperm_registry.py |
| 复合评分 α | 0.5 | sperm_registry.py |
| 已达标降权系数 | 0.3 | sperm_registry.py |

### 9.5 稳定性参数

| 参数 | 值 | 文件位置 |
|------|-----|----------|
| EMA 平滑系数 | 0.15 | sperm_registry.py |
| 等级投票窗口 | 20 帧 | sperm_registry.py |
| 投票采纳阈值 | 60% | sperm_registry.py |
| 最佳精子保持帧数 | 30 帧 | sperm_registry.py |
| 切换超越幅度 | 0.05 | sperm_registry.py |

### 9.6 界面参数

| 参数 | 值 | 文件位置 |
|------|-----|----------|
| 检测帧缓冲区 | 100 帧 | detection_thread.py |
| 分割帧缓冲区 | 50 帧 | segment_thread.py |
| VSL 上限过滤 | 90.0 μm/s | detection_thread.py |
| 统计更新间隔 | 100 ms | app.py |
| 数据库刷新间隔 | 10 帧 | sperm_registry.py |

---

## 10. 数据持久化

### 10.1 数据库结构

数据库文件位置: `SpermDatabase/sperm_experiment_<timestamp>.db`

**表 1: sperm_records** (精子记录)

| 字段 | 类型 | 说明 |
|------|------|------|
| track_id | INTEGER PRIMARY KEY | 轨迹 ID |
| vsl | REAL | 直线速度 |
| alh | REAL | 侧摆幅度 |
| kinematic_grade | INTEGER | 运动学等级 |
| pos_x | REAL | X 坐标 |
| pos_y | REAL | Y 坐标 |
| head_length | REAL | 头部长度 |
| head_width | REAL | 头部宽度 |
| head_ratio | REAL | 头部长宽比 |
| head_area | REAL | 头部面积 |
| neck_width | REAL | 颈部宽度 |
| neck_length | REAL | 颈部长度 |
| neck_head_angle | REAL | 头颈角度 |
| neck_bent_angle | REAL | 颈部弯曲角度 |
| morphology_grade | INTEGER | 形态学等级 |
| morphology_measurement_count | INTEGER | 测量次数 |
| in_candidate_pool | INTEGER | 是否在候选池 |
| scheduling_score | REAL | 调度得分 |
| composite_score | REAL | 复合评分 |
| first_seen | REAL | 首次出现时间 |
| last_updated | REAL | 最后更新时间 |
| is_active | INTEGER | 是否活跃 |
| vsl_smooth | REAL | EMA 平滑直线速度 |
| alh_smooth | REAL | EMA 平滑侧摆幅度 |
| head_length_smooth | REAL | EMA 平滑头部长度 |
| head_width_smooth | REAL | EMA 平滑头部宽度 |
| head_ratio_smooth | REAL | EMA 平滑头部长宽比 |
| head_area_smooth | REAL | EMA 平滑头部面积 |
| neck_width_smooth | REAL | EMA 平滑颈部宽度 |
| neck_bent_angle_smooth | REAL | EMA 平滑颈部弯曲角度 |
| neck_head_angle_smooth | REAL | EMA 平滑头颈角度 |
| stable_morph_grade | INTEGER | 投票后的稳定形态学等级 |
| kinematic_continuous | REAL | 连续运动学分数 |
| morphology_continuous | REAL | 连续形态学分数 |

**表 2: morphology_history** (形态学历史)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PRIMARY KEY AUTOINCREMENT | 自增 ID |
| track_id | INTEGER | 轨迹 ID |
| timestamp | REAL | 时间戳 |
| frame_number | INTEGER | 帧号 |
| head_length | REAL | 头部长度 |
| head_width | REAL | 头部宽度 |
| head_ratio | REAL | 头部长宽比 |
| head_area | REAL | 头部面积 |
| neck_width | REAL | 颈部宽度 |
| neck_length | REAL | 颈部长度 |
| neck_head_angle | REAL | 头颈角度 |
| neck_bent_angle | REAL | 颈部弯曲角度 |
| vsl | REAL | 直线速度 |
| alh | REAL | 侧摆幅度 |
| grade | INTEGER | 形态学等级 |

**表 3: experiment_meta** (实验元数据)

| 字段 | 类型 | 说明 |
|------|------|------|
| key | TEXT PRIMARY KEY | 键名 |
| value | TEXT | 值 |

### 10.2 持久化策略

- **写入频率**: 每 10 帧批量写入一次
- **写入模式**: WAL 日志模式，NORMAL 同步
- **批量写入**: 使用 `executemany` 批量插入
- **缓冲机制**: 形态学历史先缓存，与精子记录一起批量提交
- **Schema 迁移**: 启动时自动检测并新增缺失的平滑字段列

---

## 11. 用户界面与操作

### 11.1 界面布局

```
┌─────────────────────────────────────────────────────────────────┐
│  实时分析标签页                                                   │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │                         │  │                             │  │
│  │      视频显示区域        │  │      分割结果显示区域        │  │
│  │    (原始检测帧)          │  │    (分割叠加帧)             │  │
│  │                         │  │                             │  │
│  └─────────────────────────┘  └─────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 统计信息显示区域                                              ││
│  │ FPS / 缓冲区 / 分级分布 / 候选池 / 数据库路径                 ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐      │
│  │   开始     │ │   停止    │ │   录制    │ │   清除    │      │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2 控制按钮

| 按钮 | 功能 | 说明 |
|------|------|------|
| **开始** | 启动检测和分割 | 同时启动 DetectionThread 和 SegmentThread |
| **停止** | 暂停检测和分割 | 暂停两个线程，保留当前状态 |
| **录制** | 切换显示模式 | 切换正常模式/录制模式（显示所有候选精子） |
| **清除** | 切换视频/相机 | 停止当前模式，切换到另一种输入源 |

### 11.3 显示模式

**正常模式**:
- 仅显示最优精子的边界框和分割结果
- 分割叠加: 彩色半透明掩码

**录制模式**:
- 显示所有 K=10 个候选精子的完整信息
- 边界框颜色: 按等级着色
  - Grade 1: 绿色
  - Grade 2: 黄色
  - Grade 3: 橙色
  - Grade 4: 红色
  - Grade 5: 紫色
  - Grade -1: 灰色
- 显示: ID、等级、VSL、轨迹、门限椭圆

---

## 12. 使用指南

### 12.1 环境准备

**硬件要求**:
- NVIDIA GPU (支持 CUDA 12.1，显存 ≥ 4GB)
- 内存 ≥ 8GB

**软件依赖**:
```bash
pip install PyQt6 pycuda opencv-python numpy scipy albumentations PyYAML
# Basler 相机支持 (可选)
pip install pypylon
```

**TensorRT 安装**:
- 安装 TensorRT 10.9.0.34
- 确保 CUDA 12.1 已安装

### 12.2 启动程序

```bash
cd D:\SimonWorkspace\SpermSelectionV2_ForDemo_20250815
python main.py
```

程序启动后会自动切换到"实时分析"标签页。

### 12.3 操作流程

#### 使用视频文件

1. 将测试视频放入 `test-videoes/` 目录
2. 启动程序后，视频自动加载并播放
3. 点击 **开始** 按钮，启动检测和分割
4. 观察左侧视频显示区域的检测框和右侧的分割结果
5. 统计信息区域实时显示 FPS、分级分布、候选池状态
6. 点击 **停止** 暂停处理

#### 使用 Basler 相机

1. 连接 Basler 工业相机
2. 启动程序，点击 **清除** 切换到相机模式
3. 相机开始采集 (30 FPS)
4. 点击 **开始** 启动检测

#### 录制模式

1. 点击 **录制** 按钮进入录制模式
2. 录制模式下显示所有候选精子的详细信息
3. 可以观察每个精子的等级、VSL、轨迹
4. 再次点击 **录制** 回到正常模式

### 12.4 查看结果

**实时统计**:
```
FPS: 视频 30.0 | 检测 28.5 | 分割 25.2
缓冲: 检测 15/100 | 分割 8/50
检测数: 45 | 跟踪数: 32
分级: G1:0 G2:3 G3:8 G4:12 G5:6 G6:3
候选池: 5 | 本轮调度: 10 | 平均测量: 7.3 | 最高复合分: 0.60
```

**数据库文件**:
- 位置: `SpermDatabase/sperm_experiment_<timestamp>.db`
- 格式: SQLite
- 可用 SQLite Browser 或 Python sqlite3 模块查看

### 12.5 注意事项

- 每次启动程序会自动创建新的数据库文件
- 检测和分割使用 TensorRT 引擎，首次加载需要数秒预热
- 视频模式下暂停/恢复不会丢失已有的跟踪数据
- 最佳精子的切换遵循稳定机制：不会因单帧波动而跳变
- 最优精子到达画面边缘时会自动切换到下一个候选

---

## 13. 技术依赖

### 13.1 软件环境

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | 3.9 | 运行环境 |
| PyQt6 | - | GUI 框架 |
| PyTorch | - | 深度学习框架 |
| TensorRT | 10.9.0.34 | 推理加速 |
| CUDA | 12.1 | GPU 计算 |
| pycuda | - | CUDA Python 绑定 |
| OpenCV | - | 图像处理 |
| NumPy | - | 数值计算 |
| SciPy | - | 科学计算 |
| albumentations | - | 图像增强 |
| PyYAML | - | 配置文件解析 |
| pypylon | - | Basler 相机 SDK (可选) |

### 13.2 硬件要求

| 组件 | 要求 |
|------|------|
| GPU | NVIDIA GPU，支持 CUDA 12.1 |
| 显存 | ≥ 4 GB |
| 内存 | ≥ 8 GB |

### 13.3 模型文件

| 模型 | 路径 | 格式 |
|------|------|------|
| YOLOv8 检测模型 | `yolo_weights/best.engine` | TensorRT |
| YOLOv8 检测模型 | `yolo_weights/best.onnx` | ONNX |
| YOLOv8 检测模型 | `yolo_weights/best.pt` | PyTorch |
| ADSCNet 分割模型 | `seg_weights/ADSCNet_sperm_ROINAHead_250707/model_fp32.engine` | TensorRT |
| EDANet 分割模型 | `seg_weights/EDANet_sperm_NAHeadLatest250610_0610/` | TensorRT |

---

## 附录 A: 术语表

| 术语 | 英文 | 说明 |
|------|------|------|
| VSL | Straight Line Velocity | 直线速度，精子首尾连线距离/时间 |
| ALH | Amplitude of Lateral Head Displacement | 侧摆幅度，精子头部相对拟合直线的横向位移 |
| JPDAF | Joint Probabilistic Data Association Filter | 联合概率数据关联滤波器 |
| NMS | Non-Maximum Suppression | 非极大值抑制 |
| IoU | Intersection over Union | 交并比 |
| ROI | Region of Interest | 感兴趣区域 |
| EMA | Exponential Moving Average | 指数移动平均 |
| TensorRT | - | NVIDIA 推理优化引擎 |
| ADSCNet | Asymmetric Depthwise Separable Convolution Network | 非对称深度可分离卷积网络 |
| EDANet | Efficient Dense Asymmetric Network | 高效密集非对称网络 |
| 顶体 | Acrosome | 精子头部前端结构 |
| 细胞核 | Nucleus | 精子头部主体结构 |
| 颈部 | Neck | 连接头部和尾部的区域 |
| Bounding Box | 检测框 | YOLOv8 输出的精子矩形检测区域 (xyxy格式) |

---

## 附录 B: 文件结构

```
SpermSelectionV2_ForDemo_20250815/
├── main.py                          # 入口文件 (OMP修复, 高DPI支持)
├── app.py                           # PyQt6 主窗口 (管线就绪信号)
├── detection_thread.py              # 检测线程 (YOLOv8 + JPDAF + 运动学)
├── segment_thread.py                # 分割线程 (ADSCNet + 形态学 + 管线就绪)
├── video_thread.py                  # 视频读取线程 (管线活跃标志)
├── camera_thread.py                 # 相机采集线程
├── jpdaf_tracker.py                 # JPDAF 跟踪器
├── sperm_registry.py                # 精子注册表 (EMA + 投票 + 滞后 + 边缘检测)
├── modules.py                       # 神经网络基础模块
├── model_zoo/
│   ├── adscnet.py                   # ADSCNet 网络定义
│   └── edanet.py                    # EDANet 网络定义
├── yolo_weights/
│   ├── best.engine                  # YOLOv8 TensorRT 引擎
│   ├── best.onnx                    # YOLOv8 ONNX 模型
│   └── best.pt                      # YOLOv8 PyTorch 权重
├── seg_weights/
│   └── ADSCNet_sperm_ROINAHead_250707/
│       ├── model_fp32.engine        # 分割 TensorRT 引擎
│       └── config.yml               # 分割配置文件
├── test-videoes/                    # 测试视频
├── SpermDatabase/                   # 精子数据库
├── resource-files/                  # GUI 资源
├── interfacedesign.ui               # UI 设计文件
└── SPERM_SELECTION_DOCUMENTATION.md # 本文档
```

---

*文档版本: 2.0*
*更新日期: 2026-05-23*
