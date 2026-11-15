#!/usr/bin/env bash
# Real local tarball smoke in a disposable HOME/prefix. No global installs,
# containers, model downloads, test-output pipelines, or ignored exit statuses.
# Usage: test/smoke-install.sh [--runtime node|bun] [--package /path/to/file.tgz]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec node "$ROOT/scripts/smoke-package.mjs" "$@"
