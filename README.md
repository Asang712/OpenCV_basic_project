# 视频镜头运动分析器

读取本地视频或 USB 摄像头实时流，使用 **Shi-Tomasi 角点检测 + Lucas-Kanade 稀疏光流**计算帧间画面运动向量，判断镜头是**静止、平移、抖动**，在画面上叠加可视化运动箭头，并输出运动日志（CSV）。

理解视频帧间运动，是视频稳定、目标跟踪的基础，适合学习光流原理。

## 功能特性

| 功能         | 说明                                                                                   |
| ---------- | ------------------------------------------------------------------------------------ |
| 多输入源       | 本地视频文件 / USB 摄像头实时流                                                                  |
| 稀疏光流跟踪     | Shi-Tomasi 角点检测（`cv2.goodFeaturesToTrack`）+ LK 光流（`cv2.calcOpticalFlowPyrLK`），带离群点剔除 |
| 运动状态分类     | 静止 / 横向移动 / 纵向移动 / 抖动（基于平滑平均向量 + 窗口内波动阈值判断）                                          |
| 画面可视化      | 特征点、逐点运动轨迹、中心平均运动箭头、状态/位移/波动 HUD                                                     |
| CSV 运动日志   | 时间戳、帧号、运动类型、平均位移 dx/dy、位移大小、运动波动                                                     |
| 防抖预览（扩展）   | 反向补偿累计画面偏移的简易稳定窗口（`--stabilize` 或按 `s` 开关）                                           |
| 剧烈抖动标记（扩展） | 自动记录剧烈抖动片段，输出 `shake_segments.csv` 并在画面红字告警                                          |

## 运动判断原理

1. 每帧对上一帧的特征点做 LK 跟踪，得到逐点位移，取均值并剔除离群点，得到帧级运动向量 `(dx, dy)`。
2. 用最近 `--window`（默认 5）帧的向量求平均，得到平滑运动向量；同时计算窗口内各帧向量相对均值的平均偏差，作为**运动波动**（抖动强度）。
3. 阈值分类（单位 px/帧，可用命令行参数调节）：
   - 波动 `>= shake-thresh(2.5)` → **抖动**（方向来回变化，均值被抵消也能检出）
   - 平均位移 `< static-thresh(0.5)` → **静止**
   - `|dx| >= |dy|` → **横向移动**，否则 → **纵向移动**
   - 波动 `>= severe-thresh(8.0)` 时额外标记为**剧烈抖动片段**

## 环境要求

- Python 3.8+
- OpenCV >= 4.5、NumPy、Pillow（可选，用于画面中文显示）

```bash
pip install -r requirements.txt
```

> 若系统无中文字体或未安装 Pillow，画面文字自动退回英文，不影响功能。

## 使用方法

```bash
# 本地视频文件
python3 video_motion_analyzer.py --source test.mp4

# USB 摄像头 + 防抖预览
python3 video_motion_analyzer.py --source 0 --stabilize

# 无显示窗口运行（仅输出日志，适合服务器）
python3 video_motion_analyzer.py --source test.mp4 --no-display

# 调节阈值示例
python3 video_motion_analyzer.py --source test.mp4 --static-thresh 0.8 --shake-thresh 3.0
```

### 参数说明

| 参数                | 默认值                     | 说明                  |
| ----------------- | ----------------------- | ------------------- |
| `--source`        | 必填                      | 视频文件路径或摄像头索引（如 `0`） |
| `--csv`           | `output/motion_log.csv` | 运动日志输出路径            |
| `--window`        | 5                       | 运动平滑窗口帧数            |
| `--static-thresh` | 0.5                     | 静止判断阈值（px/帧）        |
| `--shake-thresh`  | 2.5                     | 抖动判断阈值（窗口内波动 px/帧）  |
| `--severe-thresh` | 8.0                     | 剧烈抖动标记阈值（px/帧）      |
| `--stabilize`     | 关                       | 开启防抖预览窗口            |
| `--no-display`    | 关                       | 无窗口运行               |
| `--max-frames`    | 0                       | 最多处理帧数（0 = 不限制）     |
| `--width`         | 0                       | 等比缩放画面宽度（0 = 原始分辨率） |

### 快捷键

- `q` / `ESC`：退出
- `s`：开关防抖预览窗口

## 输出文件

- **motion\_log.csv**：逐帧运动日志，列为 `时间戳(s), 帧号, 运动类型, 平均位移dx(px), 平均位移dy(px), 平均位移大小(px), 运动波动(px)`（UTF-8 BOM，Excel 可直接打开）。
- **shake\_segments.csv**：剧烈抖动片段列表，列为 `开始时间(s), 结束时间(s), 起始帧, 结束帧, 峰值波动(px/帧)`。
- 运行结束时控制台打印各状态帧数占比与抖动片段汇总。

## 目录结构

```
.
├── video_motion_analyzer.py   # 主程序（单文件实现全部功能）
├── requirements.txt
├── README.md
└── output/                    # 运行时生成的日志目录（已 gitignore）
```

## 实现要点

- **Shi-Tomasi 角点检测**：`maxCorners=200, qualityLevel=0.01, minDistance=10`，跟踪点不足 10 个时自动重新检测。
- **LK 稀疏光流**：金字塔 3 层、窗口 21x21，仅使用 `status==1` 的有效跟踪点。
- **离群剔除**：位移幅度超过中值 3 倍（且 >2px）的跟踪点视为误匹配剔除，避免个别错配拉偏均值。
- **防抖预览**：累计画面偏移做指数平滑后用 `cv2.warpAffine` 反向平移补偿（仅平移模型）。
- **CSV 使用 UTF-8 BOM 编码**，避免 Excel 打开中文乱码。

