"""测试用辅助: 生成可跟踪的纹理图与平移/抖动帧序列。"""

import cv2
import numpy as np


def make_texture(w=320, h=180, seed=0):
    """含大块亮斑的纹理图, 比纯噪声更适合光流跟踪。"""
    rng = np.random.RandomState(seed)
    img = np.full((h, w), 30, dtype=np.uint8)
    for _ in range(50):
        cx, cy, r = rng.randint(0, w), rng.randint(0, h), rng.randint(8, 26)
        cv2.circle(img, (cx, cy), r, int(rng.randint(120, 255)), -1)
    return cv2.GaussianBlur(img, (5, 5), 0)


def shift(img, dx, dy=0):
    """将图像内容平移 (dx, dy) 像素, 边缘填黑。"""
    out = np.zeros_like(img)
    h, w = img.shape
    sx0, sy0 = max(0, -dx), max(0, -dy)
    sx1, sy1 = w - max(0, dx), h - max(0, dy)
    dx0, dy0 = max(0, dx), max(0, dy)
    dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)
    out[dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1]
    return out
