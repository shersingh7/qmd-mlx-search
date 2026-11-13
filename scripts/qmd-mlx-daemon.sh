#!/usr/bin/env bash
# Foreground MLX daemon and safe, explicit launchd controls.
# Models are local/cached only; this launcher never downloads weights.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$REPO/scripts/launchd/qmd_mlx_daemon.py" "$@"
