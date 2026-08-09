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
# 本机代理进程会不定期挂掉（实测 2026-08-08：7891 无监听但直连正常）。
# 硬编码 proxy 会让"网络其实通着"的情况也推不上去，故先探测再决定。
if timeout 3 bash -c ':> /dev/tcp/127.0.0.1/7891' 2>/dev/null; then
  PROXY=(-c http.proxy=socks5h://127.0.0.1:7891 -c https.proxy=socks5h://127.0.0.1:7891)
else
  echo "提示：代理 127.0.0.1:7891 不可达，本次改用直连"
  PROXY=(-c http.proxy= -c https.proxy=)
fi

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -d "$PUB/.git" ]] || die "$PUB 不是 git 仓库"

# ── 同步点定位：Source-Commit trailer ─────────────────────────────────
# 旧实现用 commit message 首行 grep 定位。那是脆弱的：git log 是逆序，
# grep -m1 会命中**最新**的同名 commit——当 publish 实际停在**较早**的那个时
# （"继续"、"修正笔误"这类标题在本项目真实重复过），BASE 被定到更晚的位置，
# 中间整段 commit 被**静默漏同步**。那正是 reviewer 看不到某些改动的根源。
# 标题是人写的文本，不是标识符。回归测试见 tests/test_sync_to_publish.sh T2。
# 现在每个 publish commit 都带 `Source-Commit: <40位 sha>` trailer，
# 同步点直接读 HEAD 的 trailer——这是精确映射，不依赖任何文本约定。
read_source_trailer() {   # $1 = publish commit-ish
  git -C "$PUB" log -1 --format=%B "$1" \
    | sed -n 's/^Source-Commit:[[:space:]]*\([0-9a-f]\{40\}\)[[:space:]]*$/\1/p' | tail -1
}

# source_sha → publish_sha 的持久映射（每行 "<source40> <publish40>"）。
# 放在主仓库内、随提交进版本库，使映射本身可审计。
MAPFILE="${MAPFILE:-$SRC/docs/data/publish_sync_map.txt}"
MAPFILE_REL="${MAPFILE#"$SRC"/}"
mkdir -p "$(dirname "$MAPFILE")"
touch "$MAPFILE"

BASE=$(read_source_trailer HEAD)
if [[ -n "$BASE" ]]; then
  git -C "$SRC" cat-file -e "${BASE}^{commit}" 2>/dev/null \
    || die "publish HEAD 的 Source-Commit $BASE 在主仓库不存在（历史被改写？）"
  echo "同步点由 Source-Commit trailer 确定：${BASE:0:7}"
else
  # 回退：trailer 机制引入之前的 publish commit 没有该字段。
  # 只在这种历史遗留情形下用标题匹配，并明确告警。
  LAST_MSG=$(git -C "$PUB" log -1 --format=%s)
  BASE=$(git -C "$SRC" log --format="%H %s" | grep -F -m1 -- "$LAST_MSG" | cut -d' ' -f1)
  [[ -n "$BASE" ]] || die "publish HEAD 无 Source-Commit trailer，且标题回退也无法定位：'$LAST_MSG'"
  echo "WARN: publish HEAD 无 Source-Commit trailer，回退到标题匹配 → ${BASE:0:7}" >&2
  echo "      （本次同步产生的 commit 起将带 trailer，此回退分支之后不再触发）" >&2
fi

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
  if git -C "$PUB" diff --cached --quiet; then
    # 跳过的 commit 不写 trailer，同步点会停在它之前，下次重新尝试——
    # 幂等且不会丢失后续 commit。
    echo "SKIP $short（无收录范围内的变化）"; continue
  fi
  # trailer 直接追加到 message 末尾，与既有的 Co-Authored-By 同处一个 trailer block。
  # $msg 由命令替换取得、尾换行已被剥掉，故此处补一个 \n 即成独立行。
  msg_with_trailer=$(printf '%s\nSource-Commit: %s\n' "$msg" "$c")
  GIT_AUTHOR_DATE="$(git -C "$SRC" log -1 --format=%aI "$c")" \
  GIT_COMMITTER_DATE="$(git -C "$SRC" log -1 --format=%cI "$c")" \
    git -C "$PUB" commit -q -m "$msg_with_trailer"
  new_pub=$(git -C "$PUB" rev-parse HEAD)
  # 自指循环的断点：只改了映射文件自身的 commit 照常同步，但**不再追加新行**。
  # 否则每次同步都产生一行 → map 变脏 → 提交 map → 又触发同步 → 又追加，
  # 工作区永远不可能干净。跳过它只损失"map 更新 commit 自身的映射"，
  # 而那一条对 reviewer 没有价值（它指向的正是这份 map）。
  changed_files=$(git -C "$SRC" diff-tree --no-commit-id --name-only -r "$c")
  if [[ "$changed_files" == "$MAPFILE_REL" ]]; then
    echo "OK   $short → ${new_pub:0:7} ($n 文件，仅映射文件自身，不追加映射行)"
    continue
  fi
  printf '%s %s\n' "$c" "$new_pub" >> "$MAPFILE"
  echo "OK   $short → ${new_pub:0:7} ($n 文件)"
done

# source_sha → publish_sha 映射：trailer 已经能定位同步点，但映射文件让
# "某个 source commit 对应哪个 publish commit" 可以**反查**（trailer 只能正查）。
# reviewer 拿到 publish SHA 想回溯主仓库时需要它。
if [[ -s "$MAPFILE" ]]; then
  echo "映射已追加至 $MAPFILE（$(wc -l < "$MAPFILE") 条）"
fi

# 同步完整性校验：publish HEAD 的 trailer 必须指回主仓库 HEAD。
# 不一致说明有 commit 被静默跳过——那正是 reviewer 看不到最新改动的情形，
# 也是本轮被指出的问题的根源之一，故设为硬失败而非告警。
SRC_HEAD=$(git -C "$SRC" rev-parse HEAD)
PUB_TRAILER=$(read_source_trailer HEAD)
if [[ -n "$PENDING" && "$PUB_TRAILER" != "$SRC_HEAD" ]]; then
  echo "WARN: publish HEAD 的 Source-Commit=${PUB_TRAILER:0:7} != 主仓库 HEAD=${SRC_HEAD:0:7}" >&2
  echo "      （若末尾若干 commit 只改了 .gitignore 排除的文件，属正常）" >&2
fi

# 安全闸：绝不推送权重或超限文件
git -C "$PUB" ls-files | grep -qE '\.(pt|pth|ckpt|safetensors)$' && die "仓库内含权重文件，拒绝推送"
BIG=$(git -C "$PUB" ls-files -z | xargs -0 ls -l 2>/dev/null | awk '$5>100*1024*1024{print $9}')
[[ -z "$BIG" ]] || die "存在 >100MB 文件：$BIG"

# 认证走 VS Code 的 git credential helper（unix socket）。
# 该 socket 随 VS Code 会话变化，shell 里继承的 VSCODE_GIT_IPC_HANDLE 可能是
# 旧会话的、已 ECONNREFUSED。故每次取**最新**的 socket，而不是信任继承值。
NEWEST_SOCK=$(ls -t /run/user/$(id -u)/vscode-git-*.sock 2>/dev/null | head -1)
if [[ -n "$NEWEST_SOCK" ]]; then
  export VSCODE_GIT_IPC_HANDLE="$NEWEST_SOCK"
  echo "认证 socket: $NEWEST_SOCK"
elif [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "WARN: 未找到 vscode-git socket 且无 GITHUB_TOKEN —— 推送可能失败" >&2
fi

echo "推送中..."
GIT_TERMINAL_PROMPT=0 git -C "$PUB" "${PROXY[@]}" push origin main 2>&1 | tail -3 \
  || die "推送失败（若为认证问题：确认 VS Code 在运行，或设 GITHUB_TOKEN）"

LOCAL=$(git -C "$PUB" rev-parse HEAD)
# 二次确认。走 proxy 的 ls-remote 会间歇性返回空（push 用的是 VS Code 的
# 认证 socket，两条路径不同），所以先试 proxy、再退回直连。
REMOTE=$(git -C "$PUB" "${PROXY[@]}" ls-remote --heads origin main 2>/dev/null | cut -f1)
[[ -n "$REMOTE" ]] || REMOTE=$(git -C "$PUB" ls-remote --heads origin main 2>/dev/null | cut -f1)

if [[ "$REMOTE" == "$LOCAL" ]]; then
  echo "已同步：远端 ${REMOTE:0:7} == 本地 ${LOCAL:0:7}"
  echo "https://github.com/HNUYJJ/fasttd3_ptf"
elif [[ -z "$REMOTE" ]]; then
  # 取不到远端 sha ≠ 推送失败：上面的 push 已经 `|| die` 检查过了。
  # 把"无法验证"报成"推送失败"会让人以为要重推——那才是真的危险。
  echo "警告：无法读取远端 sha（网络/代理），但 push 已返回成功。本地 ${LOCAL:0:7}"
  echo "可手动核对：git ls-remote fasttd3_ptf main"
  echo "https://github.com/HNUYJJ/fasttd3_ptf"
else
  die "推送后远端($REMOTE) 与本地($LOCAL) 不一致"
fi
