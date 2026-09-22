#!/usr/bin/env bash
# Install this skill into an agent's skills directory.
# Usage: ./install.sh [target-dir]
# If no target is given, tries common locations (first match wins):
#   ~/.claude/skills   ~/.zcode/skills   ~/.agents/skills
set -euo pipefail
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

target="${1:-}"
if [ -z "$target" ]; then
  for d in "$HOME/.claude/skills" "$HOME/.zcode/skills" "$HOME/.agents/skills"; do
    if [ -d "$d" ]; then target="$d"; break; fi
  done
fi
if [ -z "$target" ]; then
  echo "No skills directory found. Pass one explicitly:  ./install.sh ~/.your-agent/skills"
  exit 1
fi

mkdir -p "$target"
dest="$target/$(basename "$SKILL_DIR")"
cp -r "$SKILL_DIR" "$dest"
echo "installed -> $dest"
echo "next: run 'bash $dest/setup.sh' once to fetch model weights (~380 MB)"
