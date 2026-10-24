"""MotionAnalyzer 与 ShakeSegmentRecorder 的单元测试。

运行: python3 -m pytest tests/ -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import (MotionAnalyzer, ShakeSegmentRecorder, STATE_H_MOVE,
                      STATE_SHAKE, STATE_STATIC, STATE_V_MOVE)
from tests.helpers import make_texture as _make_texture, shift as _shift_right


# ---------------- MotionAnalyzer._classify ----------------

@pytest.fixture
def analyzer():
    a = MotionAnalyzer()
    a._scale = 1.0  # 固定归一化系数, 直接测分类逻辑
    return a


def test_classify_static(analyzer):
    assert analyzer._classify(np.array([0.0, 0.0]), 0.1, 0.1) == STATE_STATIC


def test_classify_horizontal(analyzer):
    smooth = np.array([5.0, 0.0])
    assert analyzer._classify(smooth, 5.0, 0.5) == STATE_H_MOVE


def test_classify_vertical(analyzer):
    smooth = np.array([0.0, 5.0])
    assert analyzer._classify(smooth, 5.0, 0.5) == STATE_V_MOVE


def test_classify_shake_priority(analyzer):
    # 波动大时优先判抖动, 即使位移很小
    assert analyzer._classify(np.array([0.0, 0.0]), 0.1, 3.0) == STATE_SHAKE


def test_classify_scale_normalization(analyzer):
    # 分辨率翻倍, 阈值也应翻倍
    analyzer._scale = 2.0
    # static_thresh=0.5 * 2.0 = 1.0; magnitude=0.8 < 1.0 -> 静止
    assert analyzer._classify(np.array([0.8, 0.0]), 0.8, 0.1) == STATE_STATIC
    # magnitude=1.2 > 1.0 -> 横向
    assert analyzer._classify(np.array([1.2, 0.0]), 1.2, 0.1) == STATE_H_MOVE


def test_is_severe(analyzer):
    analyzer._scale = 1.0
    assert analyzer.is_severe(8.0) is True      # severe_thresh=8.0
    assert analyzer.is_severe(7.9) is False
    analyzer._scale = 2.0
    assert analyzer.is_severe(15.9) is False     # 8.0 * 2.0 = 16.0
    assert analyzer.is_severe(16.0) is True


# ---------------- ShakeSegmentRecorder ----------------

def test_recorder_records_one_segment():
    r = ShakeSegmentRecorder(severe_thresh=8.0)
    # 进入剧烈抖动
    r.update(10, 1.0, STATE_SHAKE, 9.0)
    r.update(11, 1.1, STATE_SHAKE, 12.0)
    # 退出抖动
    r.update(12, 1.2, STATE_STATIC, 0.1)
    assert len(r.segments) == 1
    seg = r.segments[0]
    assert seg[0] == 10          # 起始帧
    assert seg[1] == 1.0         # 起始时间
    assert seg[2] == 11          # 结束帧 (frame_id - 1)
    assert seg[3] == 1.2         # 结束时间
    assert seg[4] == 12.0        # 峰值波动


def test_recorder_close_flushes_open_segment():
    r = ShakeSegmentRecorder(severe_thresh=8.0)
    r.update(5, 0.5, STATE_SHAKE, 9.0)
    r.close(10, 1.0)
    assert len(r.segments) == 1
    assert r.segments[0][2] == 10    # close 时用当前帧号结束
    assert r.segments[0][4] == 9.0


def test_recorder_ignores_non_severe_shake():
    r = ShakeSegmentRecorder(severe_thresh=8.0)
    # 抖动但未达剧烈阈值, 不记录
    r.update(1, 0.1, STATE_SHAKE, 5.0)
    r.update(2, 0.2, STATE_STATIC, 0.0)
    r.close(2, 0.2)
    assert r.segments == []


# ---------------- MotionAnalyzer.process 端到端烟测 ----------------


def test_process_detects_horizontal_motion():
    """逐帧向右递增平移, 卡尔曼收敛后应稳定判为横向移动。"""
    a = MotionAnalyzer(track_width=0)
    base = _make_texture(w=300, h=150)
    a.process(base)
    last_state = None
    for k in range(1, 10):
        frame = _shift_right(base, 3 * k)   # 每帧相对前一帧右移 3px
        last_state, smooth, mag, jit = a.process(frame)
    assert last_state == STATE_H_MOVE
    assert smooth[0] > 0      # dx 为正 (向右)


def test_process_static_frames():
    """完全相同的帧, 卡尔曼收敛后应判为静止。"""
    a = MotionAnalyzer(track_width=0)
    base = _make_texture()
    a.process(base)
    last_state = None
    for _ in range(8):
        last_state, smooth, mag, jit = a.process(base)
    assert last_state == STATE_STATIC
    assert mag < 1.0


def test_process_track_width_downscale():
    """开启 track_width 时跟踪分辨率降低, 仍能稳定检测横向运动。"""
    a = MotionAnalyzer(track_width=200)
    base = _make_texture(w=400, h=240)
    a.process(base)
    last_state = None
    for k in range(1, 15):
        frame = _shift_right(base, 5 * k)   # 每帧右移 5px (缩放到 200 宽约 2.5px)
        last_state, smooth, mag, jit = a.process(frame)
    assert last_state == STATE_H_MOVE
