#!/usr/bin/env bash
# auto_push.sh — 监测仓库代码变更，自动 commit 并 push 到远端
#
# 用法:
#   ./auto_push.sh                  # 前台运行, 日志打印到终端
#   nohup ./auto_push.sh > auto_push.log 2>&1 &   # 后台运行
#   pkill -f auto_push.sh           # 停止
#   DRY_RUN=1 ./auto_push.sh        # 演练模式: 只检测和提示, 不真正提交/推送
#
# 说明:
#   - 优先使用 inotifywait 实时监听; 未安装 inotify-tools 时降级为轮询 git status
#   - 检测到变更后等待 $DEBOUNCE 秒, 聚合连续修改, 避免每次按键都产生提交
#   - 只提交被 git 跟踪的变更 (遵循 .gitignore, output/ 等不会提交)
#   - push 失败时自动 pull --rebase 后重试一次

set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRANCH="main"
REMOTE="origin"
DEBOUNCE=5          # 变更聚合等待时间(秒)
POLL_INTERVAL=10    # 轮询模式检查间隔(秒)
GIT_NAME="Asang712"
GIT_EMAIL="Asang712@users.noreply.github.com"
DRY_RUN="${DRY_RUN:-0}"

cd "$REPO_DIR" || exit 1

log() { echo "[$(date '+%F %T')] $*"; }

sync_changes() {
    [ -z "$(git status --porcelain)" ] && return 0

    sleep "$DEBOUNCE"   # 等待连续修改落盘
    [ -z "$(git status --porcelain)" ] && return 0

    files=$(git status --porcelain | awk '{print $2}' | paste -sd' ' - | cut -c1-120)
    msg="auto: sync ${files} $(date '+%F %T')"

    if [ "$DRY_RUN" = "1" ]; then
        log "[dry-run] 检测到变更, 将提交: $msg"
        return 0
    fi

    git add -A
    git -c user.name="$GIT_NAME" -c user.email="$GIT_EMAIL" commit -m "$msg" >/dev/null || return 0
    log "已提交: $msg"

    if git push "$REMOTE" "$BRANCH" >/dev/null 2>&1; then
        log "已推送到 $REMOTE/$BRANCH"
    else
        log "推送失败, 尝试 pull --rebase 后重试..."
        git pull --rebase --autostash "$REMOTE" "$BRANCH" >/dev/null 2>&1
        if git push "$REMOTE" "$BRANCH" >/dev/null 2>&1; then
            log "已推送到 $REMOTE/$BRANCH"
        else
            log "[警告] 推送仍失败, 请检查网络或远端状态; 下次变更时会重试"
        fi
    fi
}

log "开始监控: $REPO_DIR (分支 $BRANCH, 远端 $REMOTE, dry-run=$DRY_RUN)"
sync_changes   # 启动时先处理一次未提交的变更

if command -v inotifywait >/dev/null 2>&1; then
    log "模式: inotify 实时监听"
    while true; do
        inotifywait -r -q -e modify,create,delete,move \
            --exclude '/\.git' --exclude '/output/' \
            "$REPO_DIR" >/dev/null 2>&1
        sync_changes
    done
else
    log "模式: 轮询 (每 ${POLL_INTERVAL}s, 安装 inotify-tools 可改为实时监听)"
    while true; do
        sleep "$POLL_INTERVAL"
        sync_changes
    done
fi
