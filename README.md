# 视频镜头运动分析器

读取本地视频或 USB 摄像头实时流，使用 **Shi-Tomasi 角点检测 + Lucas-Kanade 稀疏光流**计算帧间画面运动向量，判断镜头是**静止、平移、抖动**，在画面上叠加可视化运动箭头，并输出运动日志（CSV）。

理解视频帧间运动，是视频稳定、目标跟踪的基础，适合学习光流原理。

## 功能特性

| 功能         | 说明                                                                                   |
| ---------- | ------------------------------------------------------------------------------------ |
| 多输入源       | 本地视频文件 / USB 摄像头实时流                                                                  |
| 稀疏光流跟踪     | Shi-Tomasi 角点检测（`cv2.goodFeaturesToTrack`）+ LK 光流（`cv2.calcOpticalFlowPyrLK`） |
| 仿射运动估计     | `cv2.estimateAffinePartial2D` + RANSAC 估计镜头整体运动，自动分离前景运动物体与背景 |
| 光照归一化       | CLAHE（`cv2.createCLAHE`）均衡对比度，减弱曝光/光照突变对光流的干扰（`--no-clahe` 可关） |
| 卡尔曼平滑       | 用常加速度卡尔曼滤波平滑运动向量，**波动定义为测量残差**，相比滑动窗口响应更快、更能处理加减速 |
| 运动状态分类     | 静止 / 横向移动 / 纵向移动 / 抖动（基于卡尔曼平滑向量 + 残差阈值判断） |
| 跟踪降分辨率     | 特征检测与光流在缩放到 `--track-width`（默认 640）的灰度图上运行，提速且不影响分类 |
| 多线程读帧       | 后台线程预读帧到有界队列，主循环不再阻塞在 `cap.read()`，提升实时性 |
| 画面可视化      | 特征点、逐点运动轨迹、中心平均运动箭头、状态/位移/波动 HUD                                                     |
| CSV 运动日志   | 时间戳、帧号、运动类型、平均位移 dx/dy、位移大小、运动波动                                                     |
| 防抖预览（扩展）   | 反向补偿累计画面偏移的简易稳定窗口（`--stabilize` 或按 `s` 开关）                                           |
| 剧烈抖动标记（扩展） | 自动记录剧烈抖动片段，输出 `shake_segments.csv` 并在画面红字告警                                          |

## 代码结构

| 文件 | 职责 |
| --- | --- |
| `video_motion_analyzer.py` | 命令行入口：参数解析、主循环编排 |
| `gui.py` | **桌面图形界面**：参数面板 + 实时视频预览 + 结果输出（Tkinter） |
| `analyzer.py` | 核心算法：`MotionKalman`、`MotionAnalyzer`（光流+仿射RANSAC+卡尔曼+分类+防抖）、`ShakeSegmentRecorder` |
| `io_utils.py` | 视频输入：`open_source`、`VideoStream`（后台线程读帧） |
| `viz.py` | 可视化：中文文字渲染 `TextRenderer`、`draw_overlay` HUD 叠加 |
| `build.sh` | 打包脚本：用 PyInstaller 生成单文件可执行程序 |
| `tests/` | 单元测试 + 合成视频 ground-truth 验证（`python3 -m pytest tests/ -v`） |

## 运动判断原理

1. **多线程读帧**：`VideoStream` 在后台线程持续解码，主线程从有界队列取帧，避免 I/O 阻塞。
2. **跟踪降分辨率**：将灰度图缩放到 `--track-width`（默认 640）做特征检测与光流跟踪，降低计算量。
3. **光照归一化**：对跟踪灰度图做 CLAHE，减弱曝光突变对光流匹配的干扰。
4. **仿射 RANSAC**：对相邻两帧的特征点做 LK 跟踪，用 `cv2.estimateAffinePartial2D(..., method=RANSAC)` 拟合整体运动，取内点（背景）的平移量作为帧级运动向量 `(dx, dy)`——运动物体等外点被自动剔除。
5. **卡尔曼平滑**：用常加速度模型卡尔曼滤波平滑运动向量；**运动波动 = 测量残差（innovation）的模**，真实反映"当前运动偏离平滑预期的程度"。
6. 阈值分类（以长边 720px 为参考分辨率，实际阈值按跟踪分辨率自动缩放，单位 px/帧，可用命令行参数调节）：
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
| `--static-thresh` | 0.5                     | 静止判断阈值（px/帧 @720p） |
| `--shake-thresh`  | 2.5                     | 抖动判断阈值（卡尔曼残差 px/帧 @720p） |
| `--severe-thresh` | 8.0                     | 剧烈抖动标记阈值（px/帧 @720p） |
| `--track-width`   | 640                     | 跟踪用灰度图宽度（0 = 不缩放） |
| `--no-clahe`      | 关                       | 关闭 CLAHE 光照归一化（默认开启） |
| `--kalman-q`      | 1e-3                    | 卡尔曼过程噪声（越大越跟随原始测量） |
| `--kalman-r`      | 0.5                     | 卡尔曼测量噪声（越大越平滑） |
| `--stabilize`     | 关                       | 开启防抖预览窗口            |
| `--no-display`    | 关                       | 无窗口运行               |
| `--max-frames`    | 0                       | 最多处理帧数（0 = 不限制）     |
| `--width`         | 0                       | 等比缩放画面宽度（0 = 原始分辨率） |

### 快捷键

- `q` / `ESC`：退出
- `s`：开关防抖预览窗口

## 桌面 GUI

无需记忆命令行参数，图形界面直接操作：

```bash
python3 gui.py
```

界面布局：
- **左侧面板**：选择视频源（文件或摄像头索引）、调节全部阈值与卡尔曼参数、CLAHE/防抖开关、开始/停止按钮
- **右侧区域**：实时视频预览（含特征点、运动箭头、HUD 叠加）
- **左下结果区**：实时显示各运动状态帧数占比与剧烈抖动片段清单

> GUI 与命令行共用同一套 `MotionAnalyzer` 算法，结果完全一致。

## 打包为可执行文件

将项目打包成**单文件可执行程序**，接收方无需安装 Python 即可运行：

```bash
./build.sh
```

产物位于 `dist/MotionAnalyzer`（Windows 上为 `dist/MotionAnalyzer.exe`），双击即可启动桌面 GUI。

**分发说明**：
- 可执行文件已内嵌 Python 解释器、OpenCV、NumPy、Pillow，体积约 80~150 MB
- **无法交叉编译**：必须在目标操作系统上运行 `build.sh`（Windows 产物需在 Windows 构建，macOS/Linux 同理）
- 打包产物 `dist/`、`build/`、`*.spec` 已在 `.gitignore` 中，不会污染仓库
- 可将 `dist/` 下的可执行文件上传到 GitHub Release 供他人下载

## 输出文件

- **motion\_log.csv**：逐帧运动日志，列为 `时间戳(s), 帧号, 运动类型, 平均位移dx(px), 平均位移dy(px), 平均位移大小(px), 运动波动(px)`（UTF-8 BOM，Excel 可直接打开）。
- **shake\_segments.csv**：剧烈抖动片段列表，列为 `开始时间(s), 结束时间(s), 起始帧, 结束帧, 峰值波动(px/帧)`。
- 运行结束时控制台打印各状态帧数占比与抖动片段汇总。

## 目录结构

```
.
├── video_motion_analyzer.py   # 命令行入口
├── gui.py                     # 桌面 GUI 入口
├── analyzer.py                # 核心算法
├── io_utils.py                # 视频输入
├── viz.py                     # 可视化
├── build.sh                   # 打包脚本
├── requirements.txt
├── README.md
├── tests/                     # 单元测试
└── output/                    # 运行时生成的日志目录（已 gitignore）
```

## 实现要点

- **Shi-Tomasi 角点检测**：`maxCorners=200, qualityLevel=0.01, minDistance=10`，跟踪点不足 10 个时自动重新检测。
- **LK 稀疏光流**：金字塔 3 层、窗口 21x21，仅使用 `status==1` 的有效跟踪点。
- **离群剔除**：位移幅度超过中值 3 倍（且 >2px）的跟踪点视为误匹配剔除，避免个别错配拉偏均值。
- **防抖预览**：累计画面偏移做指数平滑后用 `cv2.warpAffine` 反向平移补偿（仅平移模型）。
- **CSV 使用 UTF-8 BOM 编码**，避免 Excel 打开中文乱码。

