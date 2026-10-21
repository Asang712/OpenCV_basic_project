#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""项目3: 视频镜头运动分析器

读取本地视频文件或 USB 摄像头实时流，使用 Shi-Tomasi 角点检测 + Lucas-Kanade
稀疏光流计算帧间运动向量，判断镜头运动状态（静止 / 横向移动 / 纵向移动 / 抖动），
在画面上叠加可视化信息（特征点、运动箭头、状态文本），并输出 CSV 运动日志。

功能:
    1. 支持本地视频文件、USB 摄像头实时流输入
    2. 稀疏光流 Lucas-Kanade 特征点跟踪 (cv2.calcOpticalFlowPyrLK)
    3. 计算平均运动向量，分类: 静止 / 横向移动 / 纵向移动 / 抖动
    4. 实时在画面叠加特征点、运动箭头、运动状态与平均位移
    5. 输出 CSV 日志: 时间戳、运动类型、平均位移大小
    6. 扩展: 简易防抖预览窗口、自动标记剧烈抖动片段(shake_segments.csv)

用法:
    python3 video_motion_analyzer.py --source test.mp4                # 本地视频
    python3 video_motion_analyzer.py --source 0 --stabilize           # USB 摄像头
    python3 video_motion_analyzer.py --source test.mp4 --no-display   # 无窗口运行

快捷键:
    q / ESC  退出      s  开关防抖预览窗口
"""

import argparse
import csv
import os
import sys
import time
from collections import deque

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

MAIN_WINDOW = "Motion Analyzer"
STAB_WINDOW = "Stabilized Preview"

# 常见中文字体路径，用于画面中文显示（找不到则退回英文）
FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


def find_cjk_font():
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


class TextRenderer:
    """在 OpenCV 图像上绘制文字。

    若系统安装了 Pillow 且能找到中文字体，则支持中文；
    否则退回 cv2.putText 并使用英文标签。
    """

    def __init__(self):
        self.pil_mod = None
        self.pil_font = None
        try:
            from PIL import Image, ImageDraw, ImageFont
            font_path = find_cjk_font()
            if font_path:
                self.pil_mod = (Image, ImageDraw, ImageFont)
                self.pil_font = ImageFont.truetype(font_path, 24)
        except Exception:
            pass

    def supports_cn(self):
        return self.pil_font is not None

    def draw(self, img, items):
        """items: [(x, y, 文本, BGR颜色, 字号px), ...]"""
        if self.pil_font is None:
            for x, y, text, color, size in items:
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.5, size / 24.0), color, 2, cv2.LINE_AA)
            return img

        Image, ImageDraw, ImageFont = self.pil_mod
        canvas = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(canvas)
        for x, y, text, color, size in items:
            font = self.pil_font if size == 24 else ImageFont.truetype(
                self.pil_font.path, int(size))
            draw.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))
        img[:] = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
        return img


class MotionAnalyzer:
    """基于 LK 稀疏光流的镜头运动分析器。"""

    def __init__(self, window=5, static_thresh=0.5, shake_thresh=2.5, severe_thresh=8.0):
        self.static_thresh = static_thresh   # 平均位移低于该值视为静止 (px/帧)
        self.shake_thresh = shake_thresh     # 窗口内运动波动高于该值判定为抖动 (px/帧)
        self.severe_thresh = severe_thresh   # 波动高于该值标记为剧烈抖动 (px/帧)

        self.window = deque(maxlen=window)   # 最近 N 帧平均运动向量, 用于平滑
        self.prev_gray = None
        self.points = None                   # 当前跟踪的特征点
        self.last_tracks = (None, None)      # (旧点, 新点), 用于绘制轨迹

        self.cum_offset = np.zeros(2)        # 累计画面偏移 (防抖用)
        self.stab_offset = np.zeros(2)       # 平滑后的补偿偏移 (防抖用)

    def process(self, gray):
        """处理一帧灰度图。

        返回: (状态, 平滑运动向量, 平均位移大小, 运动波动)
        """
        vector = np.zeros(2)
        if self.prev_gray is not None and self.points is not None \
                and len(self.points) >= MIN_TRACK_POINTS:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.points, None, **LK_PARAMS)
            new_pts = p1[st == 1]
            old_pts = self.points[st == 1]
            if len(new_pts) >= MIN_TRACK_POINTS:
                vectors = new_pts - old_pts
                # 简单离群剔除: 幅度远大于中值的跟踪视为误匹配
                mags = np.linalg.norm(vectors, axis=1)
                keep = mags <= max(3.0 * float(np.median(mags)), 2.0)
                if keep.any():
                    vectors = vectors[keep]
                    new_pts, old_pts = new_pts[keep], old_pts[keep]
                vector = vectors.mean(axis=0)
                self.last_tracks = (old_pts, new_pts)
                self.points = new_pts.reshape(-1, 1, 2)
            else:
                self.points = None  # 跟踪丢失, 下一帧重新检测
        else:
            self.points = None

        if self.points is None:
            pts = cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)
            if pts is not None:
                self.points = pts
            self.last_tracks = (None, None)

        self.prev_gray = gray
        self.window.append(vector)
        arr = np.asarray(self.window)
        smooth = arr.mean(axis=0)
        magnitude = float(np.linalg.norm(smooth))
        jitter = float(np.linalg.norm(arr - smooth, axis=1).mean())
        state = self._classify(smooth, magnitude, jitter)
        return state, smooth, magnitude, jitter

    def _classify(self, smooth, magnitude, jitter):
        if jitter >= self.shake_thresh:
            return STATE_SHAKE
        if magnitude < self.static_thresh:
            return STATE_STATIC
        return STATE_H_MOVE if abs(smooth[0]) >= abs(smooth[1]) else STATE_V_MOVE

    def is_severe(self, jitter):
        return jitter >= self.severe_thresh

    def stabilized_view(self, frame, smooth):
        """简易防抖预览: 反向补偿(指数平滑后的)累计画面偏移。"""
        self.cum_offset += smooth
        self.stab_offset = 0.92 * self.stab_offset + 0.08 * self.cum_offset
        h, w = frame.shape[:2]
        m = np.float32([[1, 0, -self.stab_offset[0]],
                        [0, 1, -self.stab_offset[1]]])
        return cv2.warpAffine(frame, m, (w, h))


class ShakeSegmentRecorder:
    """自动记录剧烈抖动片段 (开始/结束时间与帧号)。"""

    def __init__(self, severe_thresh):
        self.severe_thresh = severe_thresh
        self.segments = []   # (起始帧, 起始时间, 结束帧, 结束时间, 峰值波动)
        self.current = None

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
        return jitter >= self.severe_thresh

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="视频镜头运动分析器 (Shi-Tomasi 角点 + LK 稀疏光流)")
    parser.add_argument("--source", required=True,
                        help="视频文件路径或摄像头索引 (如 0)")
    parser.add_argument("--csv", default="output/motion_log.csv",
                        help="运动日志 CSV 输出路径 (默认 output/motion_log.csv)")
    parser.add_argument("--window", type=int, default=5,
                        help="运动平滑窗口帧数 (默认 5)")
    parser.add_argument("--static-thresh", type=float, default=0.5,
                        help="静止判断阈值, 平均位移 px/帧 (默认 0.5)")
    parser.add_argument("--shake-thresh", type=float, default=2.5,
                        help="抖动判断阈值, 窗口内运动波动 px/帧 (默认 2.5)")
    parser.add_argument("--severe-thresh", type=float, default=8.0,
                        help="剧烈抖动标记阈值, 运动波动 px/帧 (默认 8.0)")
    parser.add_argument("--stabilize", action="store_true",
                        help="开启简易防抖预览窗口 (运行中按 s 可开关)")
    parser.add_argument("--no-display", action="store_true",
                        help="无显示窗口运行, 仅输出日志 (适合服务器/后台)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最多处理帧数, 0 表示不限制")
    parser.add_argument("--width", type=int, default=0,
                        help="将画面等比缩放到指定宽度, 0 表示不缩放")
    return parser.parse_args()


def open_source(source):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        name = f"摄像头 {source}"
    else:
        if not os.path.isfile(source):
            sys.exit(f"[错误] 找不到视频文件: {source}")
        cap = cv2.VideoCapture(source)
        name = source
    if not cap.isOpened():
        sys.exit(f"[错误] 无法打开视频源: {name}")
    return cap


def draw_overlay(frame, analyzer, state, smooth, magnitude, jitter,
                 frame_id, fps, renderer, severe):
    """在画面上叠加特征点轨迹、运动箭头与状态 HUD。"""
    h, w = frame.shape[:2]
    cn = renderer.supports_cn()

    # 1. 特征点与逐点运动轨迹
    old_pts, new_pts = analyzer.last_tracks
    if old_pts is not None:
        for o, n in zip(old_pts, new_pts):
            o = (int(o[0]), int(o[1]))
            n = (int(n[0]), int(n[1]))
            cv2.line(frame, o, n, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, n, 3, (0, 255, 0), -1)

    # 2. 画面中心的平均运动箭头 (放大 8 倍便于观察)
    center = (w // 2, h // 2)
    tip = (int(center[0] + smooth[0] * 8), int(center[1] + smooth[1] * 8))
    cv2.arrowedLine(frame, center, tip, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.15)

    # 3. 半透明信息面板
    panel_w, panel_h = 340, 140
    sub = frame[0:panel_h, 0:panel_w]
    frame[0:panel_h, 0:panel_w] = cv2.addWeighted(sub, 0.35,
                                                  np.zeros_like(sub), 0.65, 0)

    # 4. 状态文本 (支持中文则中文, 否则英文)
    items = [
        (10, 34, f"状态: {state}" if cn else f"State: {STATE_EN[state]}",
         STATE_COLORS[state], 28),
        (10, 72, f"平均位移: {magnitude:.2f} px/帧" if cn
         else f"Disp: {magnitude:.2f} px/f", (255, 255, 255), 22),
        (10, 104, f"波动: {jitter:.2f} | 帧: {frame_id} | FPS: {fps:.1f}" if cn
         else f"Jitter: {jitter:.2f} | Frame: {frame_id} | FPS: {fps:.1f}",
         (200, 200, 200), 20),
    ]
    if severe:
        items.append((10, h - 56, "!! 剧烈抖动 !!" if cn else "!! SEVERE SHAKE !!",
                      (60, 60, 235), 30))
    renderer.draw(frame, items)


def main():
    args = parse_args()
    cap = open_source(args.source)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if not fps_in or fps_in <= 0:
        fps_in = 30.0

    # 输出目录与 CSV 日志
    csv_path = os.path.abspath(args.csv)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    shake_path = os.path.join(os.path.dirname(csv_path), "shake_segments.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8-sig")
    writer = csv.writer(csv_file)
    writer.writerow(["时间戳(s)", "帧号", "运动类型",
                     "平均位移dx(px)", "平均位移dy(px)", "平均位移大小(px)", "运动波动(px)"])

    analyzer = MotionAnalyzer(args.window, args.static_thresh,
                              args.shake_thresh, args.severe_thresh)
    recorder = ShakeSegmentRecorder(args.severe_thresh)
    renderer = TextRenderer()

    state_counts = {STATE_STATIC: 0, STATE_H_MOVE: 0, STATE_V_MOVE: 0, STATE_SHAKE: 0}
    last_state = None
    frame_id = 0
    t0 = time.time()
    show_stab = args.stabilize

    print(f"视频源: {args.source} | 分辨率: {int(cap.get(3))}x{int(cap.get(4))} "
          f"| FPS: {fps_in:.1f} | 中文显示: {'是' if renderer.supports_cn() else '否(缺字体)'}")
    print("按 q / ESC 退出, 按 s 开关防抖预览\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.width and frame.shape[1] != args.width:
            scale = args.width / frame.shape[1]
            frame = cv2.resize(frame, (args.width, int(frame.shape[0] * scale)))

        frame_id += 1
        time_s = frame_id / fps_in
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        state, smooth, magnitude, jitter = analyzer.process(gray)
        state_counts[state] += 1
        severe = state == STATE_SHAKE and recorder.is_severe_state(jitter)
        recorder.update(frame_id, time_s, state, jitter)

        writer.writerow([f"{time_s:.3f}", frame_id, state,
                         f"{smooth[0]:.3f}", f"{smooth[1]:.3f}",
                         f"{magnitude:.3f}", f"{jitter:.3f}"])

        # 状态变化时在控制台打印一行
        if state != last_state:
            print(f"[{time_s:8.2f}s] 帧 {frame_id:>5} | 状态: {state} "
                  f"| 平均位移: {magnitude:.2f} px/帧 | 波动: {jitter:.2f}")
            last_state = state

        if not args.no_display:
            draw_overlay(frame, analyzer, state, smooth, magnitude, jitter,
                         frame_id, fps_in, renderer, severe)
            if show_stab:
                stab = analyzer.stabilized_view(frame, smooth)
                renderer.draw(stab, [(10, 34, "防抖预览" if renderer.supports_cn()
                                      else "Stabilized", (80, 220, 80), 26)])
                cv2.imshow(STAB_WINDOW, stab)
            cv2.imshow(MAIN_WINDOW, frame)

            if cv2.getWindowProperty(MAIN_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                show_stab = not show_stab
                if not show_stab:
                    try:
                        cv2.destroyWindow(STAB_WINDOW)
                    except cv2.error:
                        pass

        if args.max_frames and frame_id >= args.max_frames:
            break

    # ---------- 收尾: 保存日志并汇总 ----------
    recorder.close(frame_id, frame_id / fps_in)
    recorder.save(shake_path)
    csv_file.close()
    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    total = max(frame_id, 1)
    print("\n===== 分析完成 =====")
    print(f"总帧数: {frame_id} | 处理耗时: {time.time() - t0:.1f}s | 输出: {csv_path}")
    for st, cnt in state_counts.items():
        print(f"  {st}: {cnt} 帧 ({cnt / total * 100:.1f}%)")
    print(f"剧烈抖动片段: {len(recorder.segments)} 段 | 日志: {shake_path}")
    for seg in recorder.segments:
        print(f"  [{seg[1]:.2f}s ~ {seg[3]:.2f}s] 帧 {seg[0]}~{seg[2]} "
              f"| 峰值波动 {seg[4]:.1f} px/帧")


if __name__ == "__main__":
    main()
