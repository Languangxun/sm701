#!/usr/bin/env bash
# 查看下载/TCN训练进度与CPU温度
# 用法: bash watch_progress.sh [刷新秒数, 默认30]
INTERVAL=${1:-30}
while true; do
  clear
  echo "===== $(date '+%H:%M:%S') ====="
  echo "== 下载 =="
  grep -a "进度" "$HOME/桌面/sm701/extra.log" | tail -1
  echo "== TCN =="
  grep -aE "epoch|超限" "$HOME/桌面/sm701/tcn.log" | tail -6
  echo "== 多周期训练 =="
  grep -aE "h=|完成" "$HOME/桌面/sm701/horizon.log" | tail -5
  echo "== GPU =="
  nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu \
    --format=csv,noheader 2>/dev/null
  echo "== 温度 =="
  sensors 2>/dev/null | grep -E "Package id|Core 0"
  echo ""
  echo "(每 ${INTERVAL}s 刷新, Ctrl+C 退出)"
  sleep "$INTERVAL"
done
