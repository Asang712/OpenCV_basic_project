"""基于合成帧序列 (已知 ground truth 运动) 的端到端分类验证。

为每种运动生成一段连续帧, 喂给 MotionAnalyzer, 断言收敛后的主导状态
与 ground truth 一致。这是对算法正确性的回归保障。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import MotionAnalyzer, STATE_H_MOVE, STATE_SHAKE, STATE_STATIC, STATE_V_MOVE
from tests.helpers import make_texture, shift


def _run_sequence(frames, analyzer):
    """跑一段帧序列, 返回逐帧状态列表。"""
    states = []
    for f in frames:
        state, _, _, _ = analyzer.process(f)
        states.append(state)
    return states


def _dominant(states, tail=20):
    """取序列末尾 tail 帧的主导状态。"""
    from collections import Counter
    return Counter(states[-tail:]).most_common(1)[0][0]


def _pan_frames(base, dx, dy, n=40):
    """生成匀速平移的 n 帧 (每帧位移 dx, dy)。"""
    return [shift(base, dx * k, dy * k) for k in range(1, n + 1)]


def _shake_frames(base, amp=5, n=40):
    """生成抖动帧: 水平方向交替左右抖动。"""
    frames = []
    for k in range(1, n + 1):
        dx = amp if k % 2 == 0 else -amp
        frames.append(shift(base, dx, 0))
    return frames


@pytest.fixture
def base():
    return make_texture(w=320, h=180)


def test_synthetic_static(base):
    a = MotionAnalyzer(track_width=0)
    a.process(base)
    states = _run_sequence([base] * 40, a)
    assert _dominant(states) == STATE_STATIC


def test_synthetic_horizontal_right(base):
    a = MotionAnalyzer(track_width=0)
    a.process(base)
    states = _run_sequence(_pan_frames(base, 4, 0), a)
    assert _dominant(states) == STATE_H_MOVE


def test_synthetic_horizontal_left(base):
    a = MotionAnalyzer(track_width=0)
    a.process(base)
    states = _run_sequence(_pan_frames(base, -4, 0), a)
    assert _dominant(states) == STATE_H_MOVE


def test_synthetic_vertical_down(base):
    a = MotionAnalyzer(track_width=0)
    a.process(base)
    states = _run_sequence(_pan_frames(base, 0, 4), a)
    assert _dominant(states) == STATE_V_MOVE


def test_synthetic_shake(base):
    a = MotionAnalyzer(track_width=0)
    a.process(base)
    states = _run_sequence(_shake_frames(base, amp=6), a)
    assert _dominant(states) == STATE_SHAKE


def test_synthetic_static_with_clahe_off(base):
    """关闭 CLAHE 时静止场景仍应判为静止。"""
    a = MotionAnalyzer(track_width=0, use_clahe=False)
    a.process(base)
    states = _run_sequence([base] * 40, a)
    assert _dominant(states) == STATE_STATIC
