#!/usr/bin/env bash
# Publish this skill to GitHub (run on a machine with git + GitHub access).
#   ./push.sh            -> pushes to https://github.com/ydc97/video-text-removal
# Requires: git, and push access to the ydc97 account (HTTPS credentials or SSH key).
set -euo pipefail
REPO="https://github.com/ydc97/video-text-removal.git"

cd "$(dirname "${BASH_SOURCE[0]}")"
if [ ! -d .git ]; then
  git init
  git branch -M main
fi
git add -A
git -c user.name="${GIT_AUTHOR_NAME:-ydc97}" \
    -c user.email="${GIT_AUTHOR_EMAIL:-ydc97@users.noreply.github.com}" \
  commit -m "video-text-removal skill: ProPainter + LaMa video text/subtitle removal" || echo "nothing to commit"
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO"
git push -u origin main
echo "published -> $REPO"
