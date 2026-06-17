import os
import time
import threading
import sqlite3
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ============================================================
# 常量配置
# ============================================================
CONFIDENCE_THRESHOLD = 10      # 形态学测量次数阈值，达到后进入候选池
K_PER_FRAME = 8                # 每帧分割的精子数
FRESHNESS_INTERVAL = 30        # 候选池精子重新测量间隔(帧)
W_MORPH_COUNT = 0.4            # 调度权重：测量需求
W_VSL = 0.3                    # 调度权重：VSL
W_EDGE_DISTANCE = 0.3          # 调度权重：边缘距离
VSL_NORMALIZATION = 30.0       # VSL 归一化因子 (um/s)
EDGE_NORMALIZATION = 200.0     # 边缘距离归一化因子 (pixels)
COMPOSITE_ALPHA = 0.5          # 复合评分中运动学 vs 形态学的平衡
POOL_PENALTY = 0.3             # 已达标精子的调度降权系数
DB_FLUSH_INTERVAL = 10         # 每 N 帧批量写入 sperm_records
EMA_ALPHA = 0.15               # 指数移动平均系数 (越小越平滑)
GRADE_VOTE_WINDOW = 20         # 等级投票窗口大小
GRADE_VOTE_THRESHOLD = 0.6     # 投票阈值 (60% 以上一致才采纳)
BEST_HOLD_FRAMES = 8           # 最佳精子最小保持帧数（~0.8秒 @10fps，原30帧/40fps≈0.75秒）
BEST_SWITCH_MARGIN = 0.05      # 切换所需超越幅度


@dataclass
class SpermRecord:
    """单个精子的完整记录"""
    track_id: int
    # 运动学 (DetectionThread 每帧更新)
    vsl: float = 0.0
    alh: float = 0.0
    kinematic_grade: int = 6
    pos_x: float = 0.0
    pos_y: float = 0.0
    # 检测框 (xyxy 格式, DetectionThread 每帧更新)
    box_x1: float = 0.0
    box_y1: float = 0.0
    box_x2: float = 0.0
    box_y2: float = 0.0
    # 形态学 (SegmentThread 分割后更新)
    head_length: float = 0.0
    head_width: float = 0.0
    head_ratio: float = 0.0
    head_area: float = 0.0
    neck_width: float = 0.0
    neck_length: float = 0.0
    neck_head_angle: float = 0.0
    neck_bent_angle: float = 0.0
    morphology_grade: int = -1
    # 调度状态
    morphology_measurement_count: int = 0
    last_measurement_frame: int = 0
    in_candidate_pool: bool = False
    scheduling_score: float = 0.0
    composite_score: float = 0.0
    # 元数据
    first_seen: float = 0.0
    last_updated: float = 0.0
    is_active: bool = True
    # 平滑后的运动学参数 (指数移动平均)
    vsl_smooth: float = 0.0
    alh_smooth: float = 0.0
    # 平滑后的形态学参数
    head_length_smooth: float = 0.0
    head_width_smooth: float = 0.0
    head_ratio_smooth: float = 0.0
    head_area_smooth: float = 0.0
    neck_width_smooth: float = 0.0
    neck_bent_angle_smooth: float = 0.0
    neck_head_angle_smooth: float = 0.0
    # 形态学等级投票历史
    grade_history: list = field(default_factory=list)
    # 稳定形态学等级 (投票结果)
    stable_morph_grade: int = -1
    # 连续评分分量
    kinematic_continuous: float = 0.0
    morphology_continuous: float = 0.0


def _kinematic_grade_to_score(grade: int) -> float:
    """运动学分级转归一化分数: 1→1.0, 6→0.0"""
    return max(0.0, (6 - grade) / 5.0)


def _morphology_grade_to_score(grade: int) -> float:
    """形态学分级转归一化分数: 4→0.6, 5→0.4, -1→0.0"""
    if grade <= 0:
        return 0.0
    return max(0.0, (6 - grade) / 5.0)


def _continuous_kinematic_score(vsl: float, alh: float) -> float:
    """直接用 VSL/ALH 连续值计算运动学分数，避免离散化"""
    vsl_score = min(vsl / 30.0, 1.0)            # VSL 归一化 (30 um/s 满分)
    alh_score = max(0.0, 1.0 - alh / 2.0)       # ALH 越小越好 (2.0 um 为零分)
    return 0.7 * vsl_score + 0.3 * alh_score


def _edge_distance(x: float, y: float, frame_w: int, frame_h: int) -> float:
    """计算点到图像边缘的最短距离"""
    return min(x, y, frame_w - x, frame_h - y)


class SpermRegistry:
    """
    线程安全的精子注册表。
    管理所有被跟踪精子的运动学/形态学数据，实现加权轮询调度和候选池管理。
    支持 SQLite 持久化。
    """

    def __init__(self, db_path: Optional[str] = None):
        self._records: Dict[int, SpermRecord] = {}
        self._lock = threading.Lock()
        self._db_path = db_path
        self._db_conn: Optional[sqlite3.Connection] = None
        self._dirty_ids: set = set()
        self._morph_history_buffer: list = []
        self._flush_counter = 0
        self._best_sperm_id: Optional[int] = None
        self._best_sperm_hold_count: int = 0

        if db_path:
            self._init_db(db_path)

    # ----------------------------------------------------------
    # SQLite 初始化
    # ----------------------------------------------------------
    def _init_db(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        self._db_conn = sqlite3.connect(db_path, check_same_thread=False)
        self._db_conn.execute("PRAGMA journal_mode=WAL")
        self._db_conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()
        self._write_experiment_meta()

    def _create_tables(self):
        cur = self._db_conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS sperm_records (
                track_id INTEGER PRIMARY KEY,
                vsl REAL, alh REAL, kinematic_grade INTEGER,
                pos_x REAL, pos_y REAL,
                head_length REAL, head_width REAL, head_ratio REAL, head_area REAL,
                neck_width REAL, neck_length REAL, neck_head_angle REAL, neck_bent_angle REAL,
                morphology_grade INTEGER,
                morphology_measurement_count INTEGER,
                in_candidate_pool INTEGER,
                scheduling_score REAL, composite_score REAL,
                first_seen REAL, last_updated REAL, is_active INTEGER
            );
            CREATE TABLE IF NOT EXISTS morphology_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                timestamp REAL,
                frame_number INTEGER,
                head_length REAL, head_width REAL, head_ratio REAL, head_area REAL,
                neck_width REAL, neck_length REAL, neck_head_angle REAL, neck_bent_angle REAL,
                vsl REAL, alh REAL,
                grade INTEGER
            );
            CREATE TABLE IF NOT EXISTS experiment_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        # 迁移: 新增平滑字段列
        cur.execute("PRAGMA table_info(sperm_records)")
        existing_cols = {row[1] for row in cur.fetchall()}
        new_cols = {
            'vsl_smooth': 'REAL', 'alh_smooth': 'REAL',
            'head_length_smooth': 'REAL', 'head_width_smooth': 'REAL',
            'head_ratio_smooth': 'REAL', 'head_area_smooth': 'REAL',
            'neck_width_smooth': 'REAL', 'neck_bent_angle_smooth': 'REAL',
            'neck_head_angle_smooth': 'REAL',
            'stable_morph_grade': 'INTEGER',
            'kinematic_continuous': 'REAL', 'morphology_continuous': 'REAL',
        }
        for col, typ in new_cols.items():
            if col not in existing_cols:
                cur.execute(f"ALTER TABLE sperm_records ADD COLUMN {col} {typ}")
        self._db_conn.commit()

    def _write_experiment_meta(self):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._db_conn.execute(
            "INSERT OR REPLACE INTO experiment_meta (key, value) VALUES (?, ?)",
            ("start_time", now)
        )
        self._db_conn.commit()

    # ----------------------------------------------------------
    # 运动学更新 (DetectionThread 每帧调用)
    # ----------------------------------------------------------
    def update_kinematics(self, track_id: int, vsl: float, alh: float,
                          kinematic_grade: int, pos_x: float, pos_y: float,
                          bbox=None):
        with self._lock:
            if track_id not in self._records:
                self._records[track_id] = SpermRecord(
                    track_id=track_id, first_seen=time.time()
                )
            rec = self._records[track_id]
            # 原始值保留
            rec.vsl = vsl
            rec.alh = alh
            rec.kinematic_grade = kinematic_grade
            rec.pos_x = pos_x
            rec.pos_y = pos_y
            # 检测框
            if bbox is not None:
                rec.box_x1, rec.box_y1, rec.box_x2, rec.box_y2 = bbox
            # 指数移动平均平滑
            if rec.vsl_smooth == 0.0 and rec.morphology_measurement_count == 0:
                rec.vsl_smooth = vsl
                rec.alh_smooth = alh
            else:
                rec.vsl_smooth = EMA_ALPHA * vsl + (1 - EMA_ALPHA) * rec.vsl_smooth
                rec.alh_smooth = EMA_ALPHA * alh + (1 - EMA_ALPHA) * rec.alh_smooth
            # 用平滑值计算连续运动学分数
            rec.kinematic_continuous = _continuous_kinematic_score(rec.vsl_smooth, rec.alh_smooth)
            rec.last_updated = time.time()
            rec.is_active = True
            self._update_composite_score(rec)

    # ----------------------------------------------------------
    # 形态学更新 (SegmentThread 分割后调用)
    # ----------------------------------------------------------
    def update_morphology(self, track_id: int, morphology, current_frame: int = 0):
        """
        morphology: SpermMorphology 对象 (来自 segment_thread.py)
        """
        with self._lock:
            if track_id not in self._records:
                return
            rec = self._records[track_id]
            old_grade = rec.stable_morph_grade

            # 指数移动平均平滑形态学参数
            rec.head_length_smooth = self._ema(rec.head_length_smooth, morphology.head_length)
            rec.head_width_smooth = self._ema(rec.head_width_smooth, morphology.head_width)
            rec.head_ratio_smooth = self._ema(rec.head_ratio_smooth, morphology.head_ratio)
            rec.head_area_smooth = self._ema(rec.head_area_smooth, morphology.head_area)
            rec.neck_width_smooth = self._ema(rec.neck_width_smooth, morphology.neck_width)
            rec.neck_bent_angle_smooth = self._ema(rec.neck_bent_angle_smooth, morphology.neck_bent_angle)
            rec.neck_head_angle_smooth = self._ema(rec.neck_head_angle_smooth, morphology.neck_head_angle)

            # 保留原始值 (用于历史记录)
            rec.head_length = morphology.head_length
            rec.head_width = morphology.head_width
            rec.head_ratio = morphology.head_ratio
            rec.head_area = morphology.head_area
            rec.neck_width = morphology.neck_width
            rec.neck_length = morphology.neck_length
            rec.neck_head_angle = morphology.neck_head_angle
            rec.neck_bent_angle = morphology.neck_bent_angle
            rec.morphology_measurement_count += 1
            rec.last_measurement_frame = current_frame

            # 等级投票
            rec.grade_history.append(morphology.grade)
            if len(rec.grade_history) > GRADE_VOTE_WINDOW:
                rec.grade_history.pop(0)
            rec.stable_morph_grade = self._vote_grade(rec.grade_history)
            rec.morphology_grade = rec.stable_morph_grade

            # 检查是否达到置信度阈值
            if rec.morphology_measurement_count >= CONFIDENCE_THRESHOLD and not rec.in_candidate_pool:
                rec.in_candidate_pool = True
                print(f"[Registry] 精子 ID={track_id} 进入候选池 "
                      f"(测量次数={rec.morphology_measurement_count}, 稳定分级={rec.stable_morph_grade})")

            # 退化检测
            if rec.in_candidate_pool and old_grade > 0 and rec.stable_morph_grade > old_grade:
                print(f"[Registry] 精子 ID={track_id} 分级恶化: {old_grade} -> {rec.stable_morph_grade}")

            # 用平滑参数计算连续形态学分数
            rec.morphology_continuous = _morphology_grade_to_score(rec.stable_morph_grade)

            self._update_composite_score(rec)
            self._dirty_ids.add(track_id)

            # 写入形态学历史
            if self._db_conn:
                self._flush_morphology_history(track_id, morphology, current_frame)

    def _update_composite_score(self, rec: SpermRecord):
        """更新复合排序分数 (使用连续值，避免离散化平局)"""
        # 运动学: 优先用连续分数，否则回退到等级映射
        if rec.vsl_smooth > 0:
            kin_score = rec.kinematic_continuous
        else:
            kin_score = _kinematic_grade_to_score(rec.kinematic_grade)

        # 形态学: 用投票后的稳定等级
        morph_score = _morphology_grade_to_score(rec.stable_morph_grade)

        # 测量置信度: 测量越多，形态学权重越高
        confidence = min(rec.morphology_measurement_count / 20.0, 1.0)
        morph_weighted = morph_score * confidence

        rec.composite_score = COMPOSITE_ALPHA * kin_score + (1 - COMPOSITE_ALPHA) * morph_weighted

    def _ema(self, old: float, new: float) -> float:
        """指数移动平均"""
        if old == 0.0:
            return new
        return EMA_ALPHA * new + (1 - EMA_ALPHA) * old

    def _vote_grade(self, history: list) -> int:
        """多数投票决定稳定等级，忽略 -1 (未分级)"""
        from collections import Counter
        valid = [g for g in history if g > 0]
        if not valid:
            return -1
        counts = Counter(valid)
        top_grade, top_count = counts.most_common(1)[0]
        if top_count / len(valid) >= GRADE_VOTE_THRESHOLD:
            return top_grade
        return -1

    def _flush_morphology_history(self, track_id: int, morphology, frame_number: int):
        """缓冲形态学测量历史，由 flush_to_db 统一写入"""
        self._morph_history_buffer.append(
            (track_id, time.time(), frame_number,
             morphology.head_length, morphology.head_width,
             morphology.head_ratio, morphology.head_area,
             morphology.neck_width, morphology.neck_length,
             morphology.neck_head_angle, morphology.neck_bent_angle,
             morphology.vsl, morphology.alh, morphology.grade)
        )

    # ----------------------------------------------------------
    # 调度算法 (DetectionThread 每帧调用)
    # ----------------------------------------------------------
    def select_for_segmentation(self, K: int, frame_shape: Tuple[int, int],
                                current_frame: int = 0) -> List[int]:
        """
        加权轮询调度：选择 K 个精子进行分割。
        返回 track_id 列表。
        """
        frame_h, frame_w = frame_shape

        with self._lock:
            candidates = []
            for rec in self._records.values():
                if not rec.is_active:
                    continue

                # 计算调度得分
                morph_need = 1.0 - min(rec.morphology_measurement_count / CONFIDENCE_THRESHOLD, 1.0)
                vsl_norm = min(rec.vsl / VSL_NORMALIZATION, 1.0)
                edge_dist = _edge_distance(rec.pos_x, rec.pos_y, frame_w, frame_h)
                edge_norm = min(edge_dist / EDGE_NORMALIZATION, 1.0)

                score = (W_MORPH_COUNT * morph_need
                         + W_VSL * vsl_norm
                         + W_EDGE_DISTANCE * edge_norm)

                # 已进入候选池的精子降低调度优先级
                if rec.in_candidate_pool:
                    score *= POOL_PENALTY
                    # 但如果长时间未测量，提升优先级以刷新数据
                    if (current_frame - rec.last_measurement_frame) > FRESHNESS_INTERVAL:
                        score *= 2.0

                rec.scheduling_score = score
                candidates.append((rec.track_id, score))

        # 按得分降序排序，取 top-K
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [tid for tid, _ in candidates[:K]]

    # ----------------------------------------------------------
    # 候选池查询
    # ----------------------------------------------------------
    def get_candidate_pool(self) -> List[SpermRecord]:
        """返回已达标精子列表，按 composite_score 降序 (带平局打破)"""
        with self._lock:
            pool = [r for r in self._records.values()
                    if r.in_candidate_pool and r.is_active]
            pool.sort(key=lambda r: (
                r.composite_score,                          # 主排序: 复合评分
                r.morphology_measurement_count,             # 二级: 测量次数多优先
                -(r.neck_bent_angle_smooth + r.neck_head_angle_smooth)  # 三级: 颈部角度小优先
            ), reverse=True)
            return pool

    @staticmethod
    def _is_bbox_at_edge(rec: SpermRecord, frame_w: int, frame_h: int) -> bool:
        """检测框是否触碰画面边缘"""
        if rec.box_x2 == 0.0 and rec.box_y2 == 0.0:
            # bbox 未设置，回退到中心点判断
            return (rec.pos_x <= 0 or rec.pos_y <= 0 or
                    rec.pos_x >= frame_w or rec.pos_y >= frame_h)
        return (rec.box_x1 <= 0 or rec.box_y1 <= 0 or
                rec.box_x2 >= frame_w or rec.box_y2 >= frame_h)

    def get_best_sperm(self, frame_w: int = 0, frame_h: int = 0) -> Optional[SpermRecord]:
        """获取最佳精子，带滞后和平局打破。frame_w/frame_h 用于边缘检测。"""
        pool = self.get_candidate_pool()
        if not pool:
            self._best_sperm_id = None
            return None

        # 检查当前最佳精子是否到达边缘 (检测框触碰画面边界)
        if self._best_sperm_id is not None and frame_w > 0 and frame_h > 0:
            current = next((r for r in pool if r.track_id == self._best_sperm_id), None)
            if current and self._is_bbox_at_edge(current, frame_w, frame_h):
                # 当前最佳精子到达边缘，立即切换到非边缘的最佳候选
                non_edge = [r for r in pool
                            if r.track_id != self._best_sperm_id
                            and not self._is_bbox_at_edge(r, frame_w, frame_h)]
                if non_edge:
                    self._best_sperm_id = non_edge[0].track_id
                    self._best_sperm_hold_count = 0
                    return non_edge[0]
                # 没有非边缘候选，保持当前
                return current

        top = pool[0]

        if self._best_sperm_id is None:
            self._best_sperm_id = top.track_id
            self._best_sperm_hold_count = 0
            return top

        if top.track_id == self._best_sperm_id:
            self._best_sperm_hold_count = 0
            return top

        # 有新的挑战者
        current = next((r for r in pool if r.track_id == self._best_sperm_id), None)
        if current is None:
            # 当前最佳已不在候选池
            self._best_sperm_id = top.track_id
            self._best_sperm_hold_count = 0
            return top

        # 需要持续超越 + 超越幅度足够
        if top.composite_score > current.composite_score + BEST_SWITCH_MARGIN:
            self._best_sperm_hold_count += 1
            if self._best_sperm_hold_count >= BEST_HOLD_FRAMES:
                self._best_sperm_id = top.track_id
                self._best_sperm_hold_count = 0
                return top
        else:
            self._best_sperm_hold_count = 0

        # 保持当前最佳
        return current

    # ----------------------------------------------------------
    # 记录查询
    # ----------------------------------------------------------
    def get_record(self, track_id: int) -> Optional[SpermRecord]:
        with self._lock:
            return self._records.get(track_id)

    def get_all_active(self) -> List[SpermRecord]:
        with self._lock:
            return [r for r in self._records.values() if r.is_active]

    def get_stats(self) -> dict:
        """获取注册表统计信息"""
        with self._lock:
            active = [r for r in self._records.values() if r.is_active]
            pool = [r for r in active if r.in_candidate_pool]
            total_measurements = sum(r.morphology_measurement_count for r in active)
            avg_measurements = total_measurements / len(active) if active else 0.0
            top_score = pool[0].composite_score if pool else 0.0
            return {
                'total_tracked': len(self._records),
                'active_count': len(active),
                'candidate_pool_size': len(pool),
                'avg_measurement_count': avg_measurements,
                'composite_score_top': top_score,
                'db_path': self._db_path or '',
            }

    # ----------------------------------------------------------
    # 活跃状态管理
    # ----------------------------------------------------------
    def mark_inactive(self, track_ids: List[int]):
        with self._lock:
            for tid in track_ids:
                if tid in self._records:
                    self._records[tid].is_active = False

    # ----------------------------------------------------------
    # SQLite 批量刷写
    # ----------------------------------------------------------
    def flush_to_db(self):
        """批量将脏记录和形态学历史写入数据库"""
        if not self._db_conn:
            return

        with self._lock:
            dirty = list(self._dirty_ids)
            self._dirty_ids.clear()
            history = list(self._morph_history_buffer)
            self._morph_history_buffer.clear()

        if not dirty and not history:
            return

        try:
            cur = self._db_conn.cursor()
            for tid in dirty:
                with self._lock:
                    rec = self._records.get(tid)
                if not rec:
                    continue
                cur.execute(
                    """INSERT OR REPLACE INTO sperm_records
                       (track_id, vsl, alh, kinematic_grade, pos_x, pos_y,
                        head_length, head_width, head_ratio, head_area,
                        neck_width, neck_length, neck_head_angle, neck_bent_angle,
                        morphology_grade, morphology_measurement_count,
                        in_candidate_pool, scheduling_score, composite_score,
                        first_seen, last_updated, is_active,
                        vsl_smooth, alh_smooth,
                        head_length_smooth, head_width_smooth, head_ratio_smooth, head_area_smooth,
                        neck_width_smooth, neck_bent_angle_smooth, neck_head_angle_smooth,
                        stable_morph_grade, kinematic_continuous, morphology_continuous)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (rec.track_id, rec.vsl, rec.alh, rec.kinematic_grade,
                     rec.pos_x, rec.pos_y,
                     rec.head_length, rec.head_width, rec.head_ratio, rec.head_area,
                     rec.neck_width, rec.neck_length, rec.neck_head_angle, rec.neck_bent_angle,
                     rec.morphology_grade, rec.morphology_measurement_count,
                     int(rec.in_candidate_pool), rec.scheduling_score, rec.composite_score,
                     rec.first_seen, rec.last_updated, int(rec.is_active),
                     rec.vsl_smooth, rec.alh_smooth,
                     rec.head_length_smooth, rec.head_width_smooth, rec.head_ratio_smooth, rec.head_area_smooth,
                     rec.neck_width_smooth, rec.neck_bent_angle_smooth, rec.neck_head_angle_smooth,
                     rec.stable_morph_grade, rec.kinematic_continuous, rec.morphology_continuous)
                )
            if history:
                cur.executemany(
                    """INSERT INTO morphology_history
                       (track_id, timestamp, frame_number,
                        head_length, head_width, head_ratio, head_area,
                        neck_width, neck_length, neck_head_angle, neck_bent_angle,
                        vsl, alh, grade)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    history
                )
            self._db_conn.commit()
        except Exception as e:
            print(f"[Registry] 批量写入数据库失败: {e}")

    # ----------------------------------------------------------
    # 关闭
    # ----------------------------------------------------------
    def close(self):
        """关闭数据库连接"""
        self.flush_to_db()
        if self._db_conn:
            self._db_conn.close()
            self._db_conn = None
