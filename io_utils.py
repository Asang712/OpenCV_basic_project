"""视频输入工具: 视频源打开 + 后台线程读帧 (避免主循环阻塞在 cap.read)。"""

import os
import sys
from queue import Queue, Empty
from threading import Thread, Event

import cv2


def open_source(source):
    # 优先按文件路径解析, 避免把 "123.mp4" 这类文件名误判为摄像头索引
    if os.path.isfile(source):
        cap = cv2.VideoCapture(source)
        name = source
    elif source.isdigit():
        cap = cv2.VideoCapture(int(source))
        name = f"摄像头 {source}"
    else:
        sys.exit(f"[错误] 视频源不存在: {source}")
    if not cap.isOpened():
        sys.exit(f"[错误] 无法打开视频源: {name}")
    return cap


class VideoStream:
    """后台线程不断读取视频帧到有界队列, 主线程非阻塞取帧。

    使用有界队列 (queue_size) 防止处理慢于读帧时内存无限增长。
    """

    def __init__(self, cap, queue_size=2):
        self.cap = cap
        self._q = Queue(maxsize=queue_size)
        self._stop = Event()
        self._thread = Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self._q.put((False, None))  # 流结束哨兵
                break
            self._q.put((True, frame))

    def read(self, timeout=1.0):
        """返回 (ok, frame); 队列为空时最多等待 timeout 秒。"""
        try:
            return self._q.get(timeout=timeout)
        except Empty:
            return False, None

    def release(self):
        self._stop.set()
        # 清空队列, 让 worker 不再阻塞在 put
        try:
            while True:
                self._q.get_nowait()
        except Empty:
            pass
        self.cap.release()
