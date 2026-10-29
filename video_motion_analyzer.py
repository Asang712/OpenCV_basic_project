#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""项目3: 视频镜头运动分析器 (入口脚本)

读取本地视频文件或 USB 摄像头实时流, 使用 Shi-Tomasi 角点 + Lucas-Kanade
稀疏光流 + 仿射 RANSAC 估计镜头整体运动, 分类为 静止 / 横向移动 / 纵向移动 / 抖动,
在画面上叠加可视化信息, 并输出 CSV 运动日志。

核心实现见 analyzer.py (算法) 与 viz.py (可视化)。

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

import cv2

from analyzer import (MotionAnalyzer, ShakeSegmentRecorder, STATE_H_MOVE,
                      STATE_SHAKE, STATE_STATIC, STATE_V_MOVE)
from io_utils import VideoStream, open_source
from viz import MAIN_WINDOW, STAB_WINDOW, TextRenderer, draw_overlay


def parse_args():
    parser = argparse.ArgumentParser(
        description="视频镜头运动分析器 (Shi-Tomasi 角点 + LK 稀疏光流 + 仿射 RANSAC + 卡尔曼平滑)")
    parser.add_argument("--source", required=True,
                        help="视频文件路径或摄像头索引 (如 0)")
    parser.add_argument("--csv", default="output/motion_log.csv",
                        help="运动日志 CSV 输出路径 (默认 output/motion_log.csv)")
    parser.add_argument("--static-thresh", type=float, default=0.5,
                        help="静止判断阈值, 平均位移 px/帧 @720p (默认 0.5)")
    parser.add_argument("--shake-thresh", type=float, default=2.5,
                        help="抖动判断阈值, 卡尔曼残差 px/帧 @720p (默认 2.5)")
    parser.add_argument("--severe-thresh", type=float, default=8.0,
                        help="剧烈抖动标记阈值, 残差 px/帧 @720p (默认 8.0)")
    parser.add_argument("--track-width", type=int, default=640,
                        help="跟踪用灰度图宽度 (默认 640, 0=不缩放)")
    parser.add_argument("--no-clahe", action="store_true",
                        help="关闭 CLAHE 光照归一化 (默认开启)")
    parser.add_argument("--kalman-q", type=float, default=1e-2,
                        help="卡尔曼过程噪声 (默认 1e-2, 越大越跟随原始测量)")
    parser.add_argument("--kalman-r", type=float, default=0.5,
                        help="卡尔曼测量噪声 (默认 0.5, 越大越平滑)")
    parser.add_argument("--stabilize", action="store_true",
                        help="开启简易防抖预览窗口 (运行中按 s 可开关)")
    parser.add_argument("--no-display", action="store_true",
                        help="无显示窗口运行, 仅输出日志 (适合服务器/后台)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最多处理帧数, 0 表示不限制")
    parser.add_argument("--width", type=int, default=0,
                        help="将画面等比缩放到指定宽度, 0 表示不缩放")
    return parser.parse_args()


def main():
    args = parse_args()
    cap = open_source(args.source)
    stream = VideoStream(cap)   # 后台线程读帧, 主循环不再阻塞
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

    analyzer = MotionAnalyzer(args.static_thresh, args.shake_thresh, args.severe_thresh,
                              track_width=args.track_width, use_clahe=not args.no_clahe,
                              kalman_proc_noise=args.kalman_q, kalman_meas_noise=args.kalman_r)
    recorder = ShakeSegmentRecorder(args.severe_thresh)
    renderer = TextRenderer()

    state_counts = {STATE_STATIC: 0, STATE_H_MOVE: 0, STATE_V_MOVE: 0, STATE_SHAKE: 0}
    last_state = None
    frame_id = 0
    t0 = time.time()
    show_stab = args.stabilize

    print(f"视频源: {args.source} | 分辨率: {int(cap.get(3))}x{int(cap.get(4))} "
          f"| 跟踪宽度: {args.track_width} | CLAHE: {'开' if not args.no_clahe else '关'} "
          f"| FPS: {fps_in:.1f} | 中文显示: {'是' if renderer.supports_cn() else '否(缺字体)'}")
    print("按 q / ESC 退出, 按 s 开关防抖预览\n")

    while True:
        ok, frame = stream.read()
        if not ok or frame is None:
            break
        if args.width and frame.shape[1] != args.width:
            scale = args.width / frame.shape[1]
            frame = cv2.resize(frame, (args.width, int(frame.shape[0] * scale)))

        frame_id += 1
        time_s = frame_id / fps_in
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        state, smooth, magnitude, jitter = analyzer.process(gray)
        recorder.set_scale(analyzer._scale)
        state_counts[state] += 1
        severe = state == STATE_SHAKE and recorder.is_severe_state(jitter)
        recorder.update(frame_id, time_s, state, jitter)

        writer.writerow([f"{time_s:.3f}", frame_id, state,
                         f"{smooth[0]:.3f}", f"{smooth[1]:.3f}",
                         f"{magnitude:.3f}", f"{jitter:.3f}"])
        # 定期刷新, 防止进程崩溃时丢失最近一批日志
        if frame_id % 30 == 0:
            csv_file.flush()

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
    stream.release()
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
