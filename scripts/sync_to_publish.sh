#!/usr/bin/env bash
# 把主仓库的新提交同步到干净的审计仓库并推送到 GitHub。
#
# 为什么需要它：主仓库 .git 达 59GB 且早期提交含 2.4GB blob（超 GitHub 100MB 硬限），
# 永久无法直接推送。故维护一份干净仓库 /home/yjj/fasttd3_ptf_publish，
# 逐 commit 重放（保留原始时间戳，使"预注册先于实现"可由 git log 核对）。
#
# **每次在主仓库提交后都应运行本脚本**，否则外部 reviewer 看不到最新改动。
#
#   bash scripts/sync_to_publish.sh
#   DRY_RUN=1 bash scripts/sync_to_publish.sh    # 只看会同步哪些 commit
set -uo pipefail

SRC="${SRC:-/home/yjj/fasttd3_ptf}"
PUB="${PUB:-/home/yjj/fasttd3_ptf_publish}"
DRY_RUN="${DRY_RUN:-0}"
PROXY=(-c http.proxy=socks5h://127.0.0.1:7891 -c https.proxy=socks5h://127.0.0.1:7891)

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -d "$PUB/.git" ]] || die "$PUB 不是 git 仓库"

# 用 commit message 首行定位同步点——重放时完整保留了 message，故可唯一匹配。
LAST_MSG=$(git -C "$PUB" log -1 --format=%s)
BASE=$(git -C "$SRC" log --format="%H %s" | grep -F -m1 -- "$LAST_MSG" | cut -d' ' -f1)
[[ -n "$BASE" ]] || die "无法在主仓库定位 publish 的同步点：'$LAST_MSG'"

PENDING=$(git -C "$SRC" log --format=%H --reverse "${BASE}..HEAD")
if [[ -z "$PENDING" ]]; then
  echo "已是最新，无需同步（同步点 ${BASE:0:7}）"
else
  echo "同步点 ${BASE:0:7}，待同步 $(echo "$PENDING" | wc -l) 个 commit："
  for c in $PENDING; do echo "  $(git -C "$SRC" log -1 --format='%h %s' "$c")"; done
fi

if [[ "$DRY_RUN" == "1" ]]; then echo "(DRY_RUN，未执行)"; exit 0; fi

for c in $PENDING; do
  short=$(git -C "$SRC" log -1 --format=%h "$c")
  msg=$(git -C "$SRC" log -1 --format=%B "$c")
  n=0
  for f in $(git -C "$SRC" diff-tree --no-commit-id --name-only -r "$c"); do
    # publish 的 .gitignore 会挡掉权重等；此处只负责搬运，由 git add 判定收录
    if git -C "$SRC" cat-file -e "$c:$f" 2>/dev/null; then
      mkdir -p "$PUB/$(dirname "$f")"
      git -C "$SRC" show "$c:$f" > "$PUB/$f" 2>/dev/null && n=$((n+1))
    else
      rm -f "$PUB/$f" 2>/dev/null && n=$((n+1))   # 该 commit 删除了此文件
    fi
  done
  git -C "$PUB" add -A >/dev/null 2>&1
  if git -C "$PUB" diff --cached --quiet; then echo "SKIP $short（无收录范围内的变化）"; continue; fi
  GIT_AUTHOR_DATE="$(git -C "$SRC" log -1 --format=%aI "$c")" \
  GIT_COMMITTER_DATE="$(git -C "$SRC" log -1 --format=%cI "$c")" \
    git -C "$PUB" commit -q -m "$msg"
  echo "OK   $short → $(git -C "$PUB" rev-parse --short HEAD) ($n 文件)"
done

# 安全闸：绝不推送权重或超限文件
git -C "$PUB" ls-files | grep -qE '\.(pt|pth|ckpt|safetensors)$' && die "仓库内含权重文件，拒绝推送"
BIG=$(git -C "$PUB" ls-files -z | xargs -0 ls -l 2>/dev/null | awk '$5>100*1024*1024{print $9}')
[[ -z "$BIG" ]] || die "存在 >100MB 文件：$BIG"

echo "推送中..."
GIT_TERMINAL_PROMPT=0 git -C "$PUB" "${PROXY[@]}" push origin main 2>&1 | tail -3 \
  || die "推送失败"

REMOTE=$(git -C "$PUB" "${PROXY[@]}" ls-remote --heads origin main 2>/dev/null | cut -f1)
LOCAL=$(git -C "$PUB" rev-parse HEAD)
if [[ "$REMOTE" == "$LOCAL" ]]; then
  echo "已同步：远端 ${REMOTE:0:7} == 本地 ${LOCAL:0:7}"
  echo "https://github.com/HNUYJJ/fasttd3_ptf"
else
  die "推送后远端($REMOTE) 与本地($LOCAL) 不一致"
fi
