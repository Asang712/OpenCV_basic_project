#!/usr/bin/env bash
# 视频镜头运动分析器 - 桌面应用打包脚本
# 用法: ./build.sh
# 产物: dist/MotionAnalyzer (Linux/macOS) 或 dist/MotionAnalyzer.exe (Windows)
# 跨平台说明: 必须在目标操作系统上执行本脚本, 无法交叉编译
set -euo pipefail
cd "$(dirname "$0")"

APP_NAME="MotionAnalyzer"

# 1. 检查/安装 PyInstaller
if ! python3 -m PyInstaller --version >/dev/null 2>&1; then
  echo "==> 正在安装 PyInstaller ..."
  pip3 install --user pyinstaller
fi

echo "==> 开始打包 ${APP_NAME} (单文件模式) ..."

# 2. 打包
#    --onefile    生成单个可执行文件
#    --windowed   无控制台窗口 (GUI 应用)
#    --hidden-import / --collect-submodules  确保 cv2/numpy/PIL 被完整包含
python3 -m PyInstaller \
  --noconfirm \
  --clean \
  --onefile \
  --windowed \
  --name "${APP_NAME}" \
  --hidden-import=cv2 \
  --hidden-import=numpy \
  --collect-submodules=PIL \
  gui.py

echo ""
echo "==> 打包完成!"
echo "    可执行文件: dist/${APP_NAME}"
echo ""
echo "    分发说明:"
echo "    - 该文件已内嵌 Python 解释器与全部依赖, 接收方无需安装 Python"
echo "    - 体积约 80~150 MB (含 OpenCV)"
echo "    - 跨平台: 需在 Windows/macOS/Linux 各自环境重新运行本脚本"
