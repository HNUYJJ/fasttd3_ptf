#!/usr/bin/env bash
# 把干净的审计仓库推送到 GitHub。
#
# 背景：主仓库 .git 达 59GB，且早期提交含 2.4GB 的 replay.pt，
# 超过 GitHub 单文件 100MB 硬限制 → 该历史永久无法推送。
# 故另建干净仓库 /home/yjj/fasttd3_ptf_publish（24MB，含完整 P0/P1/P2 三段式时序）。
#
# 用法（三选一）：
#   1) 已有 personal access token：
#        GITHUB_TOKEN=ghp_xxx bash scripts/push_audit_repo.sh
#      （会自动创建仓库并推送）
#
#   2) 已配好 SSH 密钥：
#        USE_SSH=1 bash scripts/push_audit_repo.sh
#      （需先在 GitHub 网页手工创建空仓库）
#
#   3) 只想看要执行什么：
#        DRY_RUN=1 bash scripts/push_audit_repo.sh
#
# 环境变量：
#   GH_USER   GitHub 用户名，默认 HNUYJJ
#   GH_REPO   仓库名，默认 fasttd3_ptf
#   GH_PRIVATE  1=私有（默认），0=公开
set -uo pipefail

PUB="${PUB:-/home/yjj/fasttd3_ptf_publish}"
GH_USER="${GH_USER:-HNUYJJ}"
GH_REPO="${GH_REPO:-fasttd3_ptf}"
GH_PRIVATE="${GH_PRIVATE:-1}"
DRY_RUN="${DRY_RUN:-0}"
USE_SSH="${USE_SSH:-0}"

# git 全局代理曾指向失效端口 7893；此处显式用已验证可用的 7891。
PROXY_ARGS=(-c http.proxy=socks5h://127.0.0.1:7891 -c https.proxy=socks5h://127.0.0.1:7891)

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -d "$PUB/.git" ]] || die "$PUB 不是 git 仓库；请先确认干净仓库已构建"

echo "=== 待推送仓库 ==="
echo "路径     $PUB"
echo "提交数   $(git -C "$PUB" rev-list --count HEAD)"
echo "文件数   $(git -C "$PUB" ls-files | wc -l)"
echo "体积     $(du -sh --exclude=.git "$PUB" | cut -f1)"
echo "HEAD     $(git -C "$PUB" log -1 --format='%h %s')"
echo

# 安全闸：绝不推送权重文件
if git -C "$PUB" ls-files | grep -qE '\.(pt|pth|ckpt|safetensors)$'; then
  die "仓库内含权重文件，拒绝推送"
fi
BIG=$(git -C "$PUB" ls-files -z | xargs -0 ls -l 2>/dev/null | awk '$5>100*1024*1024{print $9}')
[[ -z "$BIG" ]] || die "存在超过 GitHub 100MB 硬限的文件：$BIG"
echo "安全检查通过：无权重文件、无 >100MB 文件"
echo

if [[ "$USE_SSH" == "1" ]]; then
  REMOTE="git@github.com:${GH_USER}/${GH_REPO}.git"
else
  REMOTE="https://github.com/${GH_USER}/${GH_REPO}.git"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN —— 将要执行："
  echo "  git -C $PUB remote add origin $REMOTE"
  echo "  git -C $PUB branch -M main"
  echo "  git -C $PUB ${PROXY_ARGS[*]} push -u origin main"
  exit 0
fi

# ── 有 token 时先创建仓库 ────────────────────────────────────────────
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  echo "检测到 GITHUB_TOKEN，尝试创建仓库 ${GH_USER}/${GH_REPO} ..."
  PRIV=$([[ "$GH_PRIVATE" == "1" ]] && echo true || echo false)
  CODE=$(curl -s -o /tmp/gh_create_resp.json -w '%{http_code}' \
    -X POST https://api.github.com/user/repos \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -d "{\"name\":\"${GH_REPO}\",\"private\":${PRIV},\"auto_init\":false}")
  case "$CODE" in
    201) echo "  仓库已创建" ;;
    422) echo "  仓库已存在，继续" ;;
    401|403) die "认证失败（HTTP $CODE），检查 token 权限（需 repo scope）" ;;
    *) echo "  创建返回 HTTP $CODE：$(head -c 200 /tmp/gh_create_resp.json)" ;;
  esac
  REMOTE="https://${GH_USER}:${GITHUB_TOKEN}@github.com/${GH_USER}/${GH_REPO}.git"
else
  echo "未设置 GITHUB_TOKEN —— 请确认已在 https://github.com/new 手工创建空仓库 ${GH_REPO}"
fi

git -C "$PUB" remote remove origin 2>/dev/null || true
git -C "$PUB" remote add origin "$REMOTE"
git -C "$PUB" branch -M main

echo
echo "推送中（24MB，通常十几秒）..."
if git -C "$PUB" "${PROXY_ARGS[@]}" push -u origin main; then
  echo
  echo "推送成功：https://github.com/${GH_USER}/${GH_REPO}"
  # 立刻把带 token 的 URL 换成干净 URL，避免凭证留在 .git/config
  git -C "$PUB" remote set-url origin "https://github.com/${GH_USER}/${GH_REPO}.git"
  echo "已将 remote 重置为不含凭证的 URL"
else
  die "推送失败；若为代理问题可试 USE_SSH=1，或检查 token 权限"
fi
