#!/usr/bin/env bash
# Explicit, opt-in hook installation; never run from an install lifecycle.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if ! git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  printf '%s\n' 'Not a Git checkout; no hooks installed.'
  exit 0
fi
HOOKS_DIR="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-path hooks)"
mkdir -p "$HOOKS_DIR"
if [[ -e "$HOOKS_DIR/pre-push" ]] && ! cmp -s "$REPO_ROOT/scripts/pre-push" "$HOOKS_DIR/pre-push"; then
  printf '%s\n' 'Refusing to replace an existing pre-push hook; merge it explicitly.' >&2
  exit 1
fi
cp "$REPO_ROOT/scripts/pre-push" "$HOOKS_DIR/pre-push"
chmod +x "$HOOKS_DIR/pre-push"
printf '%s\n' 'Installed git hooks: pre-push'
