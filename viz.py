"""可视化模块: 中文文字渲染 (兼容 OpenCV 不支持中文的问题) 与画面 HUD 叠加。"""

import os

import cv2
import numpy as np

from analyzer import (ARROW_SCALE, PANEL_H, PANEL_W, STATE_COLORS, STATE_EN)

MAIN_WINDOW = "Motion Analyzer"
STAB_WINDOW = "Stabilized Preview"

# 常见中文字体路径, 找不到则退回英文
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

    若系统安装了 Pillow 且能找到中文字体, 则支持中文;
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

    # 2. 画面中心的平均运动箭头
    center = (w // 2, h // 2)
    tip = (int(center[0] + smooth[0] * ARROW_SCALE), int(center[1] + smooth[1] * ARROW_SCALE))
    cv2.arrowedLine(frame, center, tip, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.15)

    # 3. 半透明信息面板
    sub = frame[0:PANEL_H, 0:PANEL_W]
    frame[0:PANEL_H, 0:PANEL_W] = cv2.addWeighted(sub, 0.35,
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
