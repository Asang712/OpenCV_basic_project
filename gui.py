#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视频镜头运动分析器 - 桌面 GUI

基于 Tkinter 的图形界面, 封装 analyzer.py 的核心算法。
用法: python3 gui.py
打包: ./build.sh  (生成 dist/MotionAnalyzer 可执行文件)
"""

import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from analyzer import (MotionAnalyzer, ShakeSegmentRecorder, STATE_H_MOVE,
                      STATE_SHAKE, STATE_STATIC, STATE_V_MOVE)
from io_utils import VideoStream, open_source
from viz import TextRenderer, draw_overlay


class App:
    """主窗口: 左侧参数控制, 右侧视频预览, 底部结果输出。"""

    def __init__(self, root):
        self.root = root
        root.title("视频镜头运动分析器")
        root.geometry("1120x700")
        root.minsize(900, 560)

        # ---- 控制变量 ----
        self.source_var = tk.StringVar()
        self.static_var = tk.DoubleVar(value=0.5)
        self.shake_var = tk.DoubleVar(value=2.5)
        self.severe_var = tk.DoubleVar(value=8.0)
        self.track_var = tk.IntVar(value=640)
        self.clahe_var = tk.BooleanVar(value=True)
        self.stab_var = tk.BooleanVar(value=False)
        self.kq_var = tk.DoubleVar(value=1e-3)
        self.kr_var = tk.DoubleVar(value=0.5)

        self.running = False
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=2)   # 视频帧队列 (子线程 -> 主线程)
        self.result_queue = queue.Queue()           # 结果/状态队列

        self._photo = None  # 防止 PhotoImage 被垃圾回收
        self._build_ui()
        self._poll()

    # ---------------- UI 构建 ----------------

    def _build_ui(self):
        left = ttk.Frame(self.root, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="视频源", font=("", 10, "bold")).pack(anchor=tk.W)
        src_row = ttk.Frame(left)
        src_row.pack(fill=tk.X, pady=2)
        ttk.Entry(src_row, textvariable=self.source_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(src_row, text="浏览", command=self.browse).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(left, text="(或填写摄像头索引, 如 0)", foreground="gray").pack(anchor=tk.W, pady=(0, 8))

        pg = ttk.LabelFrame(left, text="参数", padding=8)
        pg.pack(fill=tk.X, pady=4)
        self._param_row(pg, "静止阈值", self.static_var, 0, 0.0, 50.0, 0.1)
        self._param_row(pg, "抖动阈值", self.shake_var, 1, 0.0, 50.0, 0.1)
        self._param_row(pg, "剧烈阈值", self.severe_var, 2, 0.0, 100.0, 0.5)
        self._param_row(pg, "跟踪宽度", self.track_var, 3, 0, 1920, 10)
        self._param_row(pg, "卡尔曼 Q", self.kq_var, 4, 1e-5, 1.0, 1e-4)
        self._param_row(pg, "卡尔曼 R", self.kr_var, 5, 0.01, 10.0, 0.01)

        ttk.Checkbutton(left, text="CLAHE 光照归一化", variable=self.clahe_var).pack(anchor=tk.W)
        ttk.Checkbutton(left, text="开启防抖预览", variable=self.stab_var).pack(anchor=tk.W)

        btn_row = ttk.Frame(left)
        btn_row.pack(fill=tk.X, pady=10)
        self.start_btn = ttk.Button(btn_row, text="开始分析", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=2)
        self.stop_btn = ttk.Button(btn_row, text="停止", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        ttk.Label(left, text="分析结果", font=("", 10, "bold")).pack(anchor=tk.W, pady=(10, 2))
        self.result_text = tk.Text(left, height=14, width=34, state=tk.DISABLED, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        right = ttk.Frame(self.root, padding=10)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(right, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(right, textvariable=self.status_var, foreground="gray").pack(anchor=tk.W, pady=4)

    def _param_row(self, parent, label, var, row, lo, hi, inc):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
        ttk.Spinbox(parent, from_=lo, to=hi, increment=inc,
                    textvariable=var, width=10).grid(row=row, column=1, padx=6)

    # ---------------- 事件 ----------------

    def browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")])
        if path:
            self.source_var.set(path)

    def start(self):
        src = self.source_var.get().strip()
        if not src:
            messagebox.showerror("错误", "请选择视频文件或填写摄像头索引")
            return
        self.running = True
        self.stop_event.clear()
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.config(state=tk.DISABLED)
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("分析中...")
        threading.Thread(target=self._run, args=(src,), daemon=True).start()

    def stop(self):
        self.stop_event.set()

    # ---------------- 后台分析线程 ----------------

    def _run(self, src):
        try:
            cap = open_source(src)
            stream = VideoStream(cap)
            fps_in = cap.get(cv2.CAP_PROP_FPS)
            if not fps_in or fps_in <= 0:
                fps_in = 30.0

            analyzer = MotionAnalyzer(
                self.static_var.get(), self.shake_var.get(), self.severe_var.get(),
                track_width=self.track_var.get(), use_clahe=self.clahe_var.get(),
                kalman_proc_noise=self.kq_var.get(), kalman_meas_noise=self.kr_var.get())
            recorder = ShakeSegmentRecorder(self.severe_var.get())
            renderer = TextRenderer()
            counts = {STATE_STATIC: 0, STATE_H_MOVE: 0, STATE_V_MOVE: 0, STATE_SHAKE: 0}
            frame_id = 0
            t0 = time.time()
            show_stab = self.stab_var.get()

            while not self.stop_event.is_set():
                ok, frame = stream.read()
                if not ok or frame is None:
                    break
                frame_id += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                state, smooth, mag, jit = analyzer.process(gray)
                recorder.set_scale(analyzer._scale)
                counts[state] += 1
                severe = state == STATE_SHAKE and recorder.is_severe_state(jit)
                recorder.update(frame_id, frame_id / fps_in, state, jit)

                draw_overlay(frame, analyzer, state, smooth, mag, jit,
                             frame_id, fps_in, renderer, severe)
                if show_stab:
                    frame = analyzer.stabilized_view(frame, smooth)

                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

                if frame_id % 15 == 0:
                    self.result_queue.put(("counts", counts, frame_id,
                                           time.time() - t0, len(recorder.segments)))

            recorder.close(frame_id, frame_id / fps_in)
            stream.release()
            self.result_queue.put(("done", counts, frame_id,
                                   time.time() - t0, recorder.segments))
        except Exception as e:
            self.result_queue.put(("error", str(e)))

    # ---------------- 主线程轮询 (Tkinter 非线程安全) ----------------

    def _poll(self):
        # 取最新一帧显示
        latest = None
        try:
            while True:
                latest = self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            self._show_frame(latest)

        # 处理结果消息
        try:
            while True:
                item = self.result_queue.get_nowait()
                kind = item[0]
                if kind == "error":
                    messagebox.showerror("运行错误", item[1])
                    self._finish()
                elif kind == "counts":
                    self._update_results(item[1], item[2], item[3], item[4])
                elif kind == "done":
                    self._update_results(item[1], item[2], item[3], len(item[4]), item[4])
                    self.status_var.set(f"分析完成 | {item[2]} 帧 | {item[3]:.1f}s")
                    self._finish()
        except queue.Empty:
            pass

        self.root.after(33, self._poll)  # ~30fps 刷新

    def _show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        img.thumbnail((cw, ch))
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

    def _update_results(self, counts, fid, dt, nseg, segs=None):
        total = max(fid, 1)
        lines = [f"总帧数: {fid}    耗时: {dt:.1f}s", ""]
        for st, c in counts.items():
            lines.append(f"{st}: {c}  ({c / total * 100:.1f}%)")
        lines.append("")
        lines.append(f"剧烈抖动片段: {nseg} 段")
        if segs:
            for s in segs[:8]:
                lines.append(f"  {s[1]:.2f}~{s[3]:.2f}s  峰值 {s[4]:.1f} px/帧")
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", "\n".join(lines))
        self.result_text.config(state=tk.DISABLED)

    def _finish(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
