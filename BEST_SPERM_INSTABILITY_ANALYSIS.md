# 精子优选 "最佳精子跳变" 现象分析报告

## 问题描述

启动精子优选系统后，界面上显示的"最佳精子"(best sperm) 不断跳变，不是固定指向同一个精子。本报告从数据库实测数据和代码逻辑两个维度分析根因，并给出解决方案。

---

## 一、数据库实测分析

**分析对象**: `SpermDatabase/sperm_experiment_20260522_180713.db` (1.6MB, 最完整的数据库)

### 1.1 总体统计

| 指标 | 数值 |
|------|------|
| 总记录数 | 389 |
| 活跃记录 (is_active=1) | 334 |
| 候选池记录 (in_candidate_pool=1) | 366 |
| 形态学未分级 (morphology_grade=-1) | **364 (93.6%)** |
| 有形态学分级的记录 | **25 (6.4%)** |

### 1.2 复合评分分布 — 严重的分数离散化

| 统计量 | 值 |
|--------|-----|
| 均值 | 0.2861 |
| 标准差 | 0.2163 |
| 最小值 | 0.0000 |
| 最大值 | 0.8000 |
| P25 / P50 / P75 | 0.20 / 0.20 / 0.30 |

**关键发现**: 复合评分只取离散值 {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8}，因为评分公式将整数等级 (1-6) 通过 `(6-grade)/5.0` 映射为有限个离散分数。

### 1.3 Top 20 精子排名 — 核心问题

| 排名 | track_id | 运动学等级 | 形态学等级 | 测量次数 | 复合评分 |
|------|----------|-----------|-----------|---------|---------|
| 1 | 138 | -1 | **5** | 10 | **0.8000** |
| 2 | 24 | -1 | -1 | 59 | **0.7000** |
| 3 | 29 | -1 | -1 | 8 | **0.7000** |
| 4 | 39 | -1 | -1 | 10 | **0.7000** |
| 5-20 | 43,44,48,60,61,66,75,93,107,119,124,147,150,151,160,176 | -1 | -1 | 8-59 | **0.7000** |

**排名差距**:
```
Rank 1 vs 2:  差距 = 0.1000
Rank 2 vs 3:  差距 = 0.0000  ← 完全相同
Rank 3 vs 4:  差距 = 0.0000  ← 完全相同
... (一直到第20名全部相同)
```

**19 个精子的复合评分完全相同 (0.7000)**，系统无法区分它们，"最佳精子"在这些平局精子之间任意跳变。

### 1.4 形态学等级振荡 — 39.5% 的轨迹存在等级翻转

| 指标 | 数值 |
|------|------|
| 有多次测量的轨迹 | 387 |
| 存在等级变化的轨迹 | **153 (39.5%)** |
| 总等级翻转次数 | 1,536 |

最严重的振荡案例:
| track_id | 测量次数 | 等级翻转次数 | 翻转率 |
|----------|---------|-------------|--------|
| 229 | 440 | 66 | 15.0% |
| 215 | 226 | 46 | 20.4% |
| 134 | 288 | 44 | 15.3% |

**振荡模式**: 所有翻转都是在 **5 和 -1 之间** 来回切换，即形态学分类器有时能识别精子形态 (grade=5)，有时无法识别 (grade=-1)，逐帧交替。

### 1.5 运动学等级分布

| 等级 | 数量 | 占比 |
|------|------|------|
| -1 (未分级) | 68 | 17.5% |
| 1 (最优) | 5 | 1.3% |
| 2 | 17 | 4.4% |
| 3 | 33 | 8.5% |
| 4 | **205** | **52.7%** |
| 5 | 10 | 2.6% |
| 6 | 51 | 13.1% |

超过一半的精子运动学等级为 4，进一步加剧分数聚集。

---

## 二、代码层面根因分析

### 2.1 根因 1: 复合评分公式产生大量平局

**文件**: `sperm_registry.py:206-210`

```python
def _update_composite_score(self, rec: SpermRecord):
    kin_norm = _kinematic_grade_to_score(rec.kinematic_grade)    # (6-grade)/5.0
    morph_norm = _morphology_grade_to_score(rec.morphology_grade) # (6-grade)/5.0
    rec.composite_score = COMPOSITE_ALPHA * kin_norm + (1 - COMPOSITE_ALPHA) * morph_norm
```

**问题**:
- 运动学等级只有 6 个整数值 (1-6)，映射后只有 {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} 六个离散分数
- 形态学等级实际只产生 3 个值 (4→0.4, 5→0.2, -1→0.0)
- 两个离散分量的组合最多产生 18 个不同的复合评分值
- **93.6% 的精子形态学等级为 -1 (分数=0.0)**，复合评分完全由运动学等级决定
- 运动学等级分布集中在 4 (52.7%)，导致大量精子具有完全相同的复合评分

### 2.2 根因 2: 运动学等级每帧跳变，无平滑

**文件**: `jpdaf_tracker.py:212-294`

VSL 和 ALH 每帧从滑动窗口重新计算，无任何滤波:
```python
recent_points = self.trajectory[-30:]  # 每帧滑动一个点
# VSL = 首末点距离 / 时间跨度  ← 对窗口端点的异常值极其敏感
# ALH = 轨迹点到拟合直线的平均垂直距离 × 2
```

等级判定使用硬阈值，无滞后:
```python
if self.vsl >= 13.5 and self.alh <= 0.6:     # Grade 1
elif self.vsl >= 10.0 and self.alh <= 0.8:   # Grade 2
elif self.vsl >= 7.5 and self.alh <= 1.0:    # Grade 3
elif self.vsl >= 5.0:                         # Grade 4
elif self.vsl >= 2.0:                         # Grade 5
else:                                          # Grade 6
```

**关键问题**: VSL=10.1 + ALH=0.79 → Grade 2; VSL=9.9 + ALH=0.81 → Grade 4。微小的测量波动导致 **2 个等级的跳变**，复合评分瞬间变化 0.2。

### 2.3 根因 3: 形态学等级受分割噪声影响，逐帧振荡

**文件**: `segment_thread.py:652-727`

形态学分级依赖多个边界条件:
```python
# Grade 4 的判定条件 (全部必须满足):
if (VSL >= 11.0 and
    1.5 <= head_ratio <= 2.2 and
    0.5 < neck_width <= 1.0 and
    20 < neck_bent_angle <= 50 and      # ← neck_bent_angle=19.8 vs 20.2 决定 grade 4 还是 5
    10 < neck_head_angle <= 30 and
    neck_bent_angle >= 20.0 and          # ← 重复检查，同样的硬边界
    neck_head_angle >= 10.0 and
    10.0 < head_area <= 90.0):
    return 4
```

**形态学测量噪声来源**:
1. **神经网络输出波动**: 分割模型 (ADSCNet) 的 Sigmoid 输出在 0.5 阈值附近波动，导致 mask 在像素级别逐帧变化
2. **骨架提取不稳定**: Guo-Hall 细化算法对 mask 边缘变化敏感，导致 neck_bent_angle 和 neck_head_angle 波动
3. **椭圆拟合不稳定**: `cv2.fitEllipse` 对轮廓点变化敏感，head_ratio 和 head_area 随分割结果波动
4. **VSL 被用于形态学分级**: 形态学分级条件中包含 VSL >= 11.0，而 VSL 本身每帧波动

### 2.4 根因 4: 每次测量直接覆盖，无平滑/投票机制

**文件**: `sperm_registry.py:167-200`

```python
def update_morphology(self, track_id: int, morphology, current_frame: int = 0):
    rec.head_length = morphology.head_length   # 直接覆盖
    rec.head_width = morphology.head_width     # 直接覆盖
    ...
    rec.morphology_grade = morphology.grade    # 直接覆盖 ← 无平滑
    rec.morphology_measurement_count += 1      # 只增计数，不用于加权
```

**没有**: 滑动平均、指数平滑、中值滤波、投票机制。measurement_count 只用于判断是否达到候选池阈值 (10次)，不参与评分或稳定化。

### 2.5 根因 5: 最佳精子选择无稳定性保障

**文件**: `detection_thread.py:627-628`

```python
candidate_pool = self.registry.get_candidate_pool()
self.best_track_id = candidate_pool[0].track_id if candidate_pool else None
```

**文件**: `sperm_registry.py:267-273`

```python
def get_candidate_pool(self):
    pool = [r for r in self._records.values()
            if r.in_candidate_pool and r.is_active]
    pool.sort(key=lambda r: r.composite_score, reverse=True)
    return pool
```

**没有**: 最小保持时间、平局打破机制、滞后切换阈值。只要另一个精子的 composite_score 高出 0.0001，立即切换。

### 2.6 根因 6: 轨迹丢失导致精子重新初始化

**文件**: `detection_thread.py:614-618`

当精子暂时未被检测到 (遮挡、检测遗漏)，会被标记为 inactive 并移出候选池。重新出现时获得新的 track_id，从零开始 (grade=6, 无形态学历史, 不在候选池中)。

---

## 三、现象是否正常？

**结论: 在当前算法设计下，这个现象是"可预期的"，但不是"可接受的"。**

原因:
1. 分数离散化 + 大量平局 → 最佳精子在平局精子间随机跳变是数学上的必然结果
2. 分割模型输出噪声 + 硬边界阈值 → 等级振荡是统计上的高概率事件
3. 无任何平滑或稳定性机制 → 系统对噪声零容忍

这不意味着分割算法"有问题"——任何深度学习分割模型在像素级别都会有输出波动。问题在于 **评分和选择系统没有对这种噪声进行鲁棒性设计**。

---

## 四、解决方案

### 方案 1: 引入形态学等级投票/平滑机制 (推荐，影响最大)

**修改文件**: `sperm_registry.py`

**思路**: 用最近 N 次测量的多数投票替代单次测量结果。

```python
# 在 SpermRecord 中新增:
grade_history: list = field(default_factory=list)  # 最近 N 次形态学等级

# 在 update_morphology 中:
rec.grade_history.append(morphology.grade)
if len(rec.grade_history) > 20:  # 保留最近 20 次
    rec.grade_history.pop(0)
# 多数投票
from collections import Counter
valid_grades = [g for g in rec.grade_history if g > 0]
if valid_grades:
    rec.morphology_grade = Counter(valid_grades).most_common(1)[0][0]
```

**效果**: 即使分割结果逐帧波动，投票后的等级保持稳定。需要 12/20 次以上的测量给出相同等级才会改变。

### 方案 2: 使用连续分数替代离散等级 (推荐，效果最好)

**修改文件**: `sperm_registry.py`

**思路**: 不将运动学/形态学映射为离散等级再转分数，而是直接使用连续指标计算复合评分。

```python
def _update_composite_score(self, rec: SpermRecord):
    # 运动学: 直接用 VSL/ALH 的连续值
    vsl_score = min(rec.vsl / 30.0, 1.0)           # VSL 归一化到 [0, 1]
    alh_score = max(0, 1.0 - rec.alh / 2.0)        # ALH 越小越好
    kin_score = 0.7 * vsl_score + 0.3 * alh_score

    # 形态学: 用测量置信度加权
    morph_score = _morphology_grade_to_score(rec.morphology_grade)
    confidence = min(rec.morphology_measurement_count / 20.0, 1.0)
    morph_weighted = morph_score * confidence

    rec.composite_score = COMPOSITE_ALPHA * kin_score + (1 - COMPOSITE_ALPHA) * morph_weighted
```

**效果**: 
- 不同精子的 VSL 即使只差 0.1 μm/s，也会产生不同的分数
- 消除平局问题
- 测量次数越多，形态学权重越高

### 方案 3: 添加平局打破机制 (最小改动)

**修改文件**: `sperm_registry.py`

**思路**: 在 get_candidate_pool 排序时增加二级排序键。

```python
def get_candidate_pool(self):
    pool = [r for r in self._records.values()
            if r.in_candidate_pool and r.is_active]
    pool.sort(key=lambda r: (
        r.composite_score,                                    # 主排序: 复合评分
        r.morphology_measurement_count,                       # 二级: 测量次数多的优先
        -(r.neck_bent_angle + r.neck_head_angle) if r.morphology_grade > 0 else 0  # 三级: 颈部角度小的优先
    ), reverse=True)
    return pool
```

**效果**: 平局时优先选择测量次数多、形态学更稳定的精子。改动最小。

### 方案 4: 添加最佳精子切换滞后 (推荐，防止频繁跳变)

**修改文件**: `sperm_registry.py` 或 `detection_thread.py`

**思路**: 当前最佳精子只有在被另一个精子 **持续超越** 一定帧数后才切换。

```python
# 在 SpermRegistry 中新增:
_best_sperm_id: Optional[int] = None
_best_sperm_hold_count: int = 0
HOLD_THRESHOLD = 30  # 需要持续 30 帧才切换

def get_best_sperm(self):
    pool = self.get_candidate_pool()
    if not pool:
        self._best_sperm_id = None
        return None

    top = pool[0]
    if top.track_id == self._best_sperm_id:
        self._best_sperm_hold_count = 0
    else:
        self._best_sperm_hold_count += 1
        if self._best_sperm_hold_count < self.HOLD_THRESHOLD:
            # 还没持续够，保持当前最佳
            for r in pool:
                if r.track_id == self._best_sperm_id:
                    return r
        # 持续超越，切换
        self._best_sperm_id = top.track_id
        self._best_sperm_hold_count = 0
    return top
```

**效果**: 防止因瞬时噪声导致的最佳精子频繁跳变。只有当一个精子 **稳定地** 比当前最佳精子更好时才切换。

### 方案 5: 提高候选池置信度阈值

**修改文件**: `sperm_registry.py`

```python
CONFIDENCE_THRESHOLD = 20  # 从 10 提高到 20
```

**效果**: 精子需要更多测量才能进入候选池，评分更可靠。但会延迟候选池的形成。

### 方案 6: 添加运动学等级滞后

**修改文件**: `jpdaf_tracker.py`

```python
def update_grade(self):
    new_grade = self._calculate_raw_grade()
    if not hasattr(self, '_prev_grade'):
        self._prev_grade = new_grade
    # 只有当新等级与前一帧不同时，且连续 N 帧都给出新等级，才切换
    if new_grade != self._prev_grade:
        if not hasattr(self, '_grade_change_count'):
            self._grade_change_count = 0
        self._grade_change_count += 1
        if self._grade_change_count >= 5:  # 连续 5 帧一致才切换
            self._prev_grade = new_grade
            self._grade_change_count = 0
    else:
        self._grade_change_count = 0
    self.kinematic_grade = self._prev_grade
```

**效果**: 防止运动学等级在边界值附近逐帧跳变。

---

## 五、推荐实施优先级

| 优先级 | 方案 | 改动量 | 效果 | 说明 |
|--------|------|--------|------|------|
| **P0** | 方案 4: 切换滞后 | 小 | 高 | 立即解决跳变问题，不改变评分逻辑 |
| **P0** | 方案 3: 平局打破 | 极小 | 中 | 解决平局时的随机跳变 |
| **P1** | 方案 1: 等级投票 | 中 | 高 | 解决形态学等级振荡的根本问题 |
| **P1** | 方案 6: 运动学滞后 | 小 | 中 | 解决运动学等级边界跳变 |
| **P2** | 方案 2: 连续分数 | 中 | 最高 | 彻底消除离散化问题，但改动较大 |
| **P2** | 方案 5: 提高阈值 | 极小 | 低 | 提高评分可靠性，但延迟候选池形成 |

**建议组合**: 方案 4 + 方案 3 + 方案 1 一起实施，可以在不大幅重构的情况下显著改善稳定性。

---

## 六、关键文件索引

| 文件 | 关键行 | 内容 |
|------|--------|------|
| `sperm_registry.py` | 206-210 | 复合评分计算 |
| `sperm_registry.py` | 57-67 | 等级→分数映射 |
| `sperm_registry.py` | 167-200 | 形态学更新 (直接覆盖，无平滑) |
| `sperm_registry.py` | 267-273 | 候选池查询 (无平局打破) |
| `sperm_registry.py` | 226-262 | 调度算法 |
| `sperm_registry.py` | 12-22 | 常量配置 |
| `jpdaf_tracker.py` | 212-276 | VSL/ALH 计算 (每帧重算，无滤波) |
| `jpdaf_tracker.py` | 281-294 | 运动学等级判定 (硬阈值，无滞后) |
| `segment_thread.py` | 652-727 | 形态学分级 (硬边界，无投票) |
| `segment_thread.py` | 428-588 | 颈部形态学测量 (骨架化，噪声敏感) |
| `segment_thread.py` | 590-650 | 头部形态学测量 (椭圆拟合，噪声敏感) |
| `detection_thread.py` | 603-612 | 运动学数据推送到注册表 |
| `detection_thread.py` | 627-628 | 最佳精子选择 (瞬时，无稳定性) |
| `detection_thread.py` | 614-618 | 轨迹丢失/重新初始化 |

---

## 七、总结

"最佳精子跳变" 是 **算法设计层面的问题**，不是分割模型的 bug。核心原因是：

1. **评分系统离散化** — 整数等级只有 6 级，大量精子分数完全相同
2. **无任何平滑机制** — 所有测量值都是单帧原始值，噪声直接传导到评分
3. **无选择稳定性保障** — 最佳精子每帧重新计算，无滞后、无平局打破

这三个因素叠加，导致系统对分割和跟踪的微小噪声零容忍，表现为最佳精子不断跳变。解决方案的核心思想是：**在噪声和决策之间加入缓冲层**（平滑、投票、滞后、连续化）。
