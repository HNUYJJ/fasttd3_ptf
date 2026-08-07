#!/usr/bin/env bash
# sync_to_publish.sh 的回归测试（R0.1 B3）。
#
# **核心用例是重复标题**：旧实现用 `git log --format="%H %s" | grep -F -m1 "$LAST_MSG"`
# 定位同步点，两个 commit 标题相同时 grep -m1 会命中**更早**的那个，
# 于是已同步的 commit 被当成待同步、重复重放。
# "继续"、"修正笔误"这类标题在本项目里真实出现过。
#
# 用法：bash tests/test_sync_to_publish.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYNC="$SCRIPT_DIR/scripts/sync_to_publish.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

FAILED=0
ok()   { echo "  PASS  $*"; }
bad()  { echo "  FAIL  $*"; FAILED=$((FAILED+1)); }

git_q() { git -C "$1" -c user.email=t@t -c user.name=t "${@:2}" >/dev/null 2>&1; }

# ── 构造：source 仓库含两个标题完全相同的 commit ────────────────────
SRC="$TMP/src"; PUB="$TMP/pub"
mkdir -p "$SRC" "$PUB"
git_q "$SRC" init -b main
git_q "$PUB" init -b main

echo "a" > "$SRC/a.txt"; git_q "$SRC" add -A; git_q "$SRC" commit -m "继续"
C1=$(git -C "$SRC" rev-parse HEAD)
echo "b" > "$SRC/b.txt"; git_q "$SRC" add -A; git_q "$SRC" commit -m "中间提交"
echo "c" > "$SRC/c.txt"; git_q "$SRC" add -A; git_q "$SRC" commit -m "继续"   # ← 同标题
C3=$(git -C "$SRC" rev-parse HEAD)

# publish 已同步到 C3（带 trailer），内容对齐
for f in a.txt b.txt c.txt; do cp "$SRC/$f" "$PUB/$f"; done
git_q "$PUB" add -A
git -C "$PUB" -c user.email=t@t -c user.name=t \
    commit -q -m "继续

Source-Commit: $C3" >/dev/null 2>&1

# 再加一个 source commit，它是唯一应被同步的
echo "d" > "$SRC/d.txt"; git_q "$SRC" add -A; git_q "$SRC" commit -m "新增 d"
C4=$(git -C "$SRC" rev-parse HEAD)

echo "T1  重复标题下同步点必须由 trailer 精确定位"
OUT=$(SRC="$SRC" PUB="$PUB" MAPFILE="$TMP/map.txt" DRY_RUN=1 bash "$SYNC" 2>&1)
if grep -q "Source-Commit trailer 确定：${C3:0:7}" <<<"$OUT"; then
  ok "定位到 ${C3:0:7}（第二个"继续"），未被第一个同标题 commit 误导"
else
  bad "未按 trailer 定位；输出：$OUT"
fi
if grep -q "待同步 1 个 commit" <<<"$OUT" && grep -q "新增 d" <<<"$OUT"; then
  ok "只有 1 个待同步 commit（新增 d）"
else
  bad "待同步集合不正确；输出：$OUT"
fi

echo "T2  publish 停在**较早**的同名 commit 时，旧的标题匹配会静默漏同步"
# git log 是逆序，grep -m1 命中的是**最新**的同名 commit。
# 故缺陷方向不是"重复重放"而是"漏同步"：publish 实际停在 C1，
# 标题匹配却把 BASE 定到 C3，于是 C2、C3 被整段跳过——
# 这正是 reviewer 看不到某些改动的根源。
PUB4="$TMP/pub4"; mkdir -p "$PUB4"; git_q "$PUB4" init -b main
cp "$SRC/a.txt" "$PUB4/"; git_q "$PUB4" add -A
git -C "$PUB4" -c user.email=t@t -c user.name=t \
    commit -q -m "继续

Source-Commit: $C1" >/dev/null 2>&1

BASE_OLD=$(git -C "$SRC" log --format="%H %s" | grep -F -m1 -- "继续" | cut -d' ' -f1)
if [[ "$BASE_OLD" == "$C1" ]]; then
  bad "标题匹配意外命中 C1，本用例失去意义"
else
  ok "旧行为把 BASE 误定为 ${BASE_OLD:0:7}（应为 ${C1:0:7}）"
  MISSED=$(git -C "$SRC" log --format=%H "${BASE_OLD}..HEAD" | wc -l)
  TRUE_N=$(git -C "$SRC" log --format=%H "${C1}..HEAD" | wc -l)
  ok "旧行为只会同步 $MISSED 个，实际应同步 $TRUE_N 个 —— 漏 $((TRUE_N-MISSED)) 个"
fi

OUT4=$(SRC="$SRC" PUB="$PUB4" MAPFILE="$TMP/map4.txt" DRY_RUN=1 bash "$SYNC" 2>&1)
if grep -q "Source-Commit trailer 确定：${C1:0:7}" <<<"$OUT4" \
   && grep -q "待同步 3 个 commit" <<<"$OUT4"; then
  ok "新实现按 trailer 定位到 ${C1:0:7}，待同步 3 个，一个不漏"
else
  bad "新实现未正确处理该场景；输出：$OUT4"
fi

echo "T3  trailer 指向不存在的 commit 时必须硬失败，不得静默回退"
PUB2="$TMP/pub2"; mkdir -p "$PUB2"; git_q "$PUB2" init -b main
echo x > "$PUB2/x.txt"; git_q "$PUB2" add -A
git -C "$PUB2" -c user.email=t@t -c user.name=t \
    commit -q -m "坏 trailer

Source-Commit: $(printf '0%.0s' {1..40})" >/dev/null 2>&1
OUT2=$(SRC="$SRC" PUB="$PUB2" MAPFILE="$TMP/map2.txt" DRY_RUN=1 bash "$SYNC" 2>&1)
RC2=$?
if [[ $RC2 -ne 0 ]] && grep -q "在主仓库不存在" <<<"$OUT2"; then
  ok "硬失败并说明原因"
else
  bad "未硬失败（rc=$RC2）；输出：$OUT2"
fi

echo "T4  无 trailer 的历史 publish 才允许回退标题匹配，且必须告警"
PUB3="$TMP/pub3"; mkdir -p "$PUB3"; git_q "$PUB3" init -b main
echo a > "$PUB3/a.txt"; git_q "$PUB3" add -A; git_q "$PUB3" commit -m "中间提交"
OUT3=$(SRC="$SRC" PUB="$PUB3" MAPFILE="$TMP/map3.txt" DRY_RUN=1 bash "$SYNC" 2>&1)
if grep -q "无 Source-Commit trailer，回退到标题匹配" <<<"$OUT3"; then
  ok "回退分支触发并告警"
else
  bad "回退分支未按预期告警；输出：$OUT3"
fi

echo
if [[ $FAILED -eq 0 ]]; then
  echo "sync_to_publish 回归测试：全部通过"
  exit 0
fi
echo "sync_to_publish 回归测试：$FAILED 项失败"
exit 1
