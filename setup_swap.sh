#!/usr/bin/env bash
# 扩容交换分区到 32G (新增 28G swap 文件, 不动现有 /swap.img, 训练中也能安全执行)
# 用法: sudo bash setup_swap.sh
set -euo pipefail

TARGET_TOTAL_GB=32
CURRENT_SWAPFILE=/swap.img
NEW_SWAPFILE=/swap2.img

if [ "$(id -u)" -ne 0 ]; then
  echo "需要 root: sudo bash $0"
  exit 1
fi

# 现有 swap 总量(MB, /proc/meminfo 单位为 kB)
cur_mb=$(( $(awk '/SwapTotal/ {print $2}' /proc/meminfo) / 1024 ))
target_mb=$((TARGET_TOTAL_GB * 1024))
add_mb=$((target_mb - cur_mb))

if [ "$add_mb" -le 0 ]; then
  echo "当前 swap 已是 ${cur_mb}MB, 无需扩容"
  swapon --show
  exit 0
fi
echo "当前 swap ${cur_mb}MB, 新增 $((add_mb / 1024))G -> ${NEW_SWAPFILE}"

# 创建新 swap 文件 (已存在且足够大则复用)
if [ -f "$NEW_SWAPFILE" ]; then
  have_mb=$(du -m "$NEW_SWAPFILE" | cut -f1)
  if [ "$have_mb" -ge "$add_mb" ]; then
    echo "$NEW_SWAPFILE 已存在(${have_mb}MB), 直接启用"
  else
    swapoff "$NEW_SWAPFILE" 2>/dev/null || true
    rm -f "$NEW_SWAPFILE"
    fallocate -l "${add_mb}M" "$NEW_SWAPFILE"
  fi
else
  fallocate -l "${add_mb}M" "$NEW_SWAPFILE"
fi
chmod 600 "$NEW_SWAPFILE"
mkswap "$NEW_SWAPFILE" >/dev/null
swapon "$NEW_SWAPFILE"

# 幂等写入 fstab, 重启自动挂载
if ! grep -q "^${NEW_SWAPFILE}[[:space:]]" /etc/fstab; then
  echo "${NEW_SWAPFILE} none swap sw 0 0" >> /etc/fstab
fi

# 内存吃紧场景: 适度降低换出倾向(需保留一些 swap 缓冲大矩阵)
echo 'vm.swappiness=20' > /etc/sysctl.d/99-swappiness.conf
sysctl -q vm.swappiness=20

echo "--- 完成 ---"
swapon --show
free -h
