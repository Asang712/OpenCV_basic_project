"""运动分析核心: 状态常量、MotionAnalyzer、ShakeSegmentRecorder。

MotionAnalyzer 基于 Shi-Tomasi 角点 + Lucas-Kanade 稀疏光流, 用仿射模型 + RANSAC
估计镜头整体运动 (自动分离背景运动与前景运动物体), 并按「位移 / 波动」分类状态。
"""

import csv

import cv2
import numpy as np

# ---------------- 运动状态定义 ----------------
STATE_STATIC = "静止"
STATE_H_MOVE = "横向移动"
STATE_V_MOVE = "纵向移动"
STATE_SHAKE = "抖动"

STATE_EN = {
    STATE_STATIC: "STATIC",
    STATE_H_MOVE: "HORIZONTAL",
    STATE_V_MOVE: "VERTICAL",
    STATE_SHAKE: "SHAKE",
}

# BGR 颜色: 静止-绿 / 横向-橙 / 纵向-蓝 / 抖动-红
STATE_COLORS = {
    STATE_STATIC: (80, 220, 80),
    STATE_H_MOVE: (40, 160, 255),
    STATE_V_MOVE: (255, 170, 40),
    STATE_SHAKE: (60, 60, 235),
}

# Shi-Tomasi 角点检测参数 (cv2.goodFeaturesToTrack)
FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7)
# Lucas-Kanade 稀疏光流参数 (cv2.calcOpticalFlowPyrLK)
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

MIN_TRACK_POINTS = 10  # 有效跟踪点低于该数量时重新检测特征点

# 离群点剔除 (RANSAC 失败时的回退策略): 位移超过「中值 × OUTLIER_MEDIAN_SCALE」丢弃
OUTLIER_MEDIAN_SCALE = 3.0
OUTLIER_MIN_MAG = 2.0

# 防抖指数平滑系数 (0~1): 越大越跟随原始偏移, 越小越平滑
STAB_SMOOTH_ALPHA = 0.08

# 可视化常量
ARROW_SCALE = 8       # 平均运动箭头放大倍数
PANEL_W, PANEL_H = 340, 140   # HUD 半透明面板尺寸
REF_RES = 720.0       # 阈值参考分辨率 (长边像素), 用于跨分辨率归一化

# 仅使用画面边缘区域的特征点估计镜头运动 (中心区域易被前景物体占据, 造成误判)
BORDER_RATIO = 0.2    # 边缘占短边的比例 (0.2 = 外圈 20%)
MIN_INLIER_RATIO = 0.3  # RANSAC 内点占比低于该值时认为估计不可靠


class MotionKalman:
    """运动向量的卡尔曼平滑器。

    状态 x = [dx, dy, ddx, ddy] (运动向量 + 其变化率), 常加速度模型。
    相比滑动窗口均值, 卡尔曼滤波在平滑的同时响应更快, 且能处理加减速。
    输出 smooth 为滤波后的运动向量, jitter 为测量残差 (innovation)。
    """

    def __init__(self, proc_noise=1e-3, meas_noise=0.5):
        # 状态转移: x(k+1) = F x(k); 速度项积分到位置
        self.F = np.eye(4)
        self.F[0, 2] = self.F[1, 3] = 1.0
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)  # 只观测 (dx, dy)
        self.x = np.zeros(4)
        self.P = np.eye(4) * 10.0
        self.Q = np.eye(4) * proc_noise
        self.R = np.eye(2) * meas_noise

    def update(self, measurement):
        """measurement: 本帧原始运动向量 (2,)。返回 (smooth, jitter)。"""
        # ---------- predict ----------
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # ---------- update ----------
        y = measurement - self.H @ self.x                 # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        smooth = self.x[:2].copy()
        jitter = float(np.linalg.norm(y))
        return smooth, jitter


class MotionAnalyzer:
    """基于 LK 稀疏光流 + 仿射 RANSAC 的镜头运动分析器。

    Args:
        track_width: 跟踪用灰度图宽度, >0 时将原图等比缩放到该宽度再做特征检测
            与光流跟踪 (显著提速), 0 表示不缩放。
    """

    def __init__(self, static_thresh=0.5, shake_thresh=2.5, severe_thresh=8.0,
                 track_width=640, use_clahe=True,
                 kalman_proc_noise=1e-2, kalman_meas_noise=0.5,
                 border_ratio=BORDER_RATIO):
        self.static_thresh = static_thresh   # 平均位移低于该值视为静止 (px/帧 @720p)
        self.shake_thresh = shake_thresh     # 波动高于该值判定为抖动 (px/帧 @720p)
        self.severe_thresh = severe_thresh   # 波动高于该值标记为剧烈抖动 (@720p)
        self.track_width = track_width
        self.use_clahe = use_clahe
        self.border_ratio = border_ratio     # 仅边缘区域的点参与镜头运动估计

        self.prev_gray = None
        self.points = None                   # 当前跟踪的特征点 (跟踪坐标系)
        self.last_tracks = (None, None)      # (旧点, 新点), 原始坐标系, 用于绘制

        self.cum_offset = np.zeros(2)        # 累计画面偏移 (防抖用)
        self.stab_offset = np.zeros(2)       # 平滑后的补偿偏移 (防抖用)
        self._scale = 1.0                    # 阈值归一化系数
        self._draw_scale = 1.0               # 跟踪坐标 -> 原始坐标
        self._prev_vector = np.zeros(2)      # 上一帧运动向量 (估计不可靠时复用)

        self.kalman = MotionKalman(proc_noise=kalman_proc_noise,
                                   meas_noise=kalman_meas_noise)
        # CLAHE 光照归一化, 减弱曝光突变对光流的干扰
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None

    def process(self, gray):
        """处理一帧灰度图。

        返回: (状态, 平滑运动向量, 平均位移大小, 运动波动)
        """
        # ---- 1. 跟踪降分辨率: 同物理运动在更低分辨率下像素位移更小, 计算更快 ----
        orig_h, orig_w = gray.shape[:2]
        if self.track_width and orig_w != self.track_width:
            scale = self.track_width / orig_w
            track_gray = cv2.resize(gray, (self.track_width, int(orig_h * scale)))
            self._draw_scale = 1.0 / scale
        else:
            track_gray = gray
            self._draw_scale = 1.0

        # ---- 2. CLAHE 光照归一化: 减弱曝光/光照突变对光流的干扰 ----
        if self.clahe is not None:
            track_gray = self.clahe.apply(track_gray)

        vector = np.zeros(2)
        if self.prev_gray is not None and self.points is not None \
                and len(self.points) >= MIN_TRACK_POINTS:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, track_gray, self.points, None, **LK_PARAMS)
            new_pts = p1[st == 1]
            old_pts = self.points[st == 1]
            if len(new_pts) >= MIN_TRACK_POINTS:
                # 仅取画面边缘区域的点估计镜头运动 (中心易被前景物体占据造成误判)
                h, w = track_gray.shape[:2]
                b = int(min(h, w) * self.border_ratio)
                border_mask = (
                    (old_pts[:, 0] < b) | (old_pts[:, 0] >= w - b) |
                    (old_pts[:, 1] < b) | (old_pts[:, 1] >= h - b))
                est_old = old_pts[border_mask]
                est_new = new_pts[border_mask]
                # 边缘点不足时回退到全部点
                if len(est_old) < MIN_TRACK_POINTS:
                    est_old, est_new = old_pts, new_pts

                result = self._estimate_motion(est_old, est_new)
                if result is not None:
                    vector, old_in, new_in = result
                    self.last_tracks = (old_in * self._draw_scale,
                                        new_in * self._draw_scale)
                    self.points = new_in.reshape(-1, 1, 2)
                    self._prev_vector = vector
                else:
                    # 估计不可靠 (内点占比过低), 复用上一帧向量避免抖动误报
                    vector = self._prev_vector
                    self.points = new_pts.reshape(-1, 1, 2)
            else:
                self.points = None
        else:
            self.points = None

        if self.points is None:
            pts = cv2.goodFeaturesToTrack(track_gray, mask=None, **FEATURE_PARAMS)
            if pts is not None:
                self.points = pts
            self.last_tracks = (None, None)

        self.prev_gray = track_gray
        # 阈值按跟踪分辨率归一化, 保证跨分辨率分类一致
        self._scale = max(track_gray.shape) / REF_RES
        # 卡尔曼平滑: 输出滤波后的运动向量与测量残差(抖动强度)
        smooth, jitter = self.kalman.update(vector)
        magnitude = float(np.linalg.norm(smooth))
        state = self._classify(smooth, magnitude, jitter)
        return state, smooth, magnitude, jitter

    def _estimate_motion(self, old_pts, new_pts):
        """用仿射模型 + RANSAC 估计整体运动。

        Returns: (vector, old_inliers, new_inliers) 或 None (估计不可靠)
            vector: 仿射平移量 (跟踪坐标系); 若 RANSAC 失败则返回均值向量。
            当 RANSAC 内点占比过低 (前景物体主导) 时返回 None, 调用方应复用旧向量。
        """
        M, inliers = cv2.estimateAffinePartial2D(
            old_pts, new_pts, method=cv2.RANSAC,
            ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99)
        if M is not None and inliers is not None:
            mask = inliers.ravel().astype(bool)
            inlier_ratio = mask.sum() / max(len(old_pts), 1)
            if mask.sum() >= MIN_TRACK_POINTS and inlier_ratio >= MIN_INLIER_RATIO:
                vector = np.array([M[0, 2], M[1, 2]], dtype=float)
                return vector, old_pts[mask], new_pts[mask]
            # 内点占比过低 -> 很可能前景物体主导了"最大一致集", 估计不可靠
            return None

        # 回退: 中值离群剔除 + 均值向量
        vectors = new_pts - old_pts
        mags = np.linalg.norm(vectors, axis=1)
        keep = mags <= max(OUTLIER_MEDIAN_SCALE * float(np.median(mags)), OUTLIER_MIN_MAG)
        if keep.any():
            vectors = vectors[keep]
            new_pts, old_pts = new_pts[keep], old_pts[keep]
        return vectors.mean(axis=0), old_pts, new_pts

    def _classify(self, smooth, magnitude, jitter):
        s = self._scale
        # 先看波动 (方向来回变 -> 抖动), 再看位移 (整体快慢 -> 静止/移动)
        if jitter >= self.shake_thresh * s:
            return STATE_SHAKE
        if magnitude < self.static_thresh * s:
            return STATE_STATIC
        return STATE_H_MOVE if abs(smooth[0]) >= abs(smooth[1]) else STATE_V_MOVE

    def is_severe(self, jitter):
        return jitter >= self.severe_thresh * self._scale

    def stabilized_view(self, frame, smooth):
        """简易防抖预览: 反向补偿(指数平滑后的)累计画面偏移。"""
        self.cum_offset += smooth
        self.stab_offset = (1 - STAB_SMOOTH_ALPHA) * self.stab_offset + STAB_SMOOTH_ALPHA * self.cum_offset
        h, w = frame.shape[:2]
        m = np.float32([[1, 0, -self.stab_offset[0]],
                        [0, 1, -self.stab_offset[1]]])
        return cv2.warpAffine(frame, m, (w, h))


class ShakeSegmentRecorder:
    """自动记录剧烈抖动片段 (开始/结束时间与帧号)。"""

    def __init__(self, severe_thresh, scale=1.0):
        self.severe_thresh = severe_thresh
        self.scale = scale            # 阈值归一化系数 (与 analyzer._scale 保持一致)
        self.segments = []   # (起始帧, 起始时间, 结束帧, 结束时间, 峰值波动)
        self.current = None

    def set_scale(self, scale):
        """同步 analyzer 的分辨率归一化系数。"""
        self.scale = scale

    def update(self, frame_id, time_s, state, jitter):
        if state == STATE_SHAKE and self.is_severe_state(jitter):
            if self.current is None:
                self.current = [frame_id, time_s, jitter]
            else:
                self.current[2] = max(self.current[2], jitter)
        elif self.current is not None:
            self.segments.append((*self.current[:2], frame_id - 1, time_s, self.current[2]))
            self.current = None

    def is_severe_state(self, jitter):
        return jitter >= self.severe_thresh * self.scale

    def close(self, frame_id, time_s):
        if self.current is not None:
            self.segments.append((*self.current[:2], frame_id, time_s, self.current[2]))
            self.current = None

    def save(self, path):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["开始时间(s)", "结束时间(s)", "起始帧", "结束帧", "峰值波动(px/帧)"])
            for s in self.segments:
                writer.writerow([f"{s[1]:.3f}", f"{s[3]:.3f}", s[0], s[2], f"{s[4]:.3f}"])
