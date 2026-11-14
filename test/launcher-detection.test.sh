#!/usr/bin/env bash
# Execute the real bin/qmd with recording runtimes. Never duplicate its branch logic.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/qmd-launcher-test.XXXXXXXX")"
trap 'rm -rf "$WORK"' EXIT
# bin/qmd resolves its own installation directory physically. Model the same
# fixture directory under /var aliases or a caller-supplied symlinked TMPDIR;
# user argv values below remain literal and must never be normalized.
WORK="$(cd "$WORK" && pwd -P)"
mkdir -p "$WORK/runtimes" "$WORK/package with spaces/bin" "$WORK/package with spaces/dist/cli" "$WORK/links"
cp "$ROOT/bin/qmd" "$WORK/package with spaces/bin/qmd"
printf '// fixture entrypoint\n' > "$WORK/package with spaces/dist/cli/qmd.js"
# Shell stubs record each argument separately, including empty values and newlines.
for runtime in node bun; do
  printf '#!/bin/sh\nprintf "%%s\\n" "%s"\nfor arg do printf "<%%s>\\n" "$arg"; done\nexit "${QMD_TEST_EXIT:-0}"\n' "$runtime" > "$WORK/runtimes/$runtime"
  chmod +x "$WORK/runtimes/$runtime"
done
PATH="$WORK/runtimes:$PATH"
export PATH
PACKAGE="$WORK/package with spaces"
PASS=0
assert_runtime() {
  local label="$1" expected="$2" output
  output="$(env -u QMD_RUNTIME BUN_INSTALL=/unrelated/bun "$PACKAGE/bin/qmd" 'space value' '' '--dash' 'wild*card' $'line\nbreak')"
  local want
  want="$(printf '%s\n<%s>\n<space value>\n<>\n<--dash>\n<wild*card>\n<line\nbreak>' "$expected" "$PACKAGE/dist/cli/qmd.js")"
  if [[ "$output" != "$want" ]]; then
    printf 'FAIL: %s runtime/argv\nexpected:\n%s\nactual:\n%s\n' "$label" "$want" "$output" >&2
    exit 1
  fi
  printf 'OK: %s runtime and argv\n' "$label"
  PASS=$((PASS + 1))
}
assert_runtime 'no lockfiles (BUN_INSTALL ignored)' node
touch "$PACKAGE/bun.lock"
assert_runtime 'bun.lock' bun
rm "$PACKAGE/bun.lock"
touch "$PACKAGE/bun.lockb"
assert_runtime 'bun.lockb' bun
touch "$PACKAGE/package-lock.json"
assert_runtime 'npm lock outranks Bun lock' node
rm "$PACKAGE/package-lock.json" "$PACKAGE/bun.lockb"
mkdir "$PACKAGE/node_modules"
touch "$PACKAGE/bun.lock" "$PACKAGE/node_modules/.package-lock.json"
assert_runtime 'actual npm metadata outranks source lock' node
rm "$PACKAGE/node_modules/.package-lock.json"
touch "$PACKAGE/node_modules/.modules.yaml"
assert_runtime 'actual pnpm metadata outranks source lock' node
rm "$PACKAGE/node_modules/.modules.yaml"
ln -s '../package with spaces/bin/qmd' "$WORK/links/qmd"
ln -s "$WORK/links/qmd" "$WORK/links/qmd-absolute"
output="$(QMD_RUNTIME=node "$WORK/links/qmd-absolute" 'symlink argv')"
[[ "$output" == "$(printf 'node\n<%s>\n<symlink argv>' "$PACKAGE/dist/cli/qmd.js")" ]]
printf 'OK: relative/absolute symlinks and runtime override preserve argv\n'
PASS=$((PASS + 1))
# Published packages have no repository lock: inspect the real enclosing install.
INSTALLED_ROOT="$WORK/install-root"
mkdir -p "$INSTALLED_ROOT/node_modules/@tobilu"
cp -R "$PACKAGE" "$INSTALLED_ROOT/node_modules/@tobilu/qmd"
rm "$INSTALLED_ROOT/node_modules/@tobilu/qmd/bun.lock"
ORIGINAL_PACKAGE="$PACKAGE"
PACKAGE="$INSTALLED_ROOT/node_modules/@tobilu/qmd"
touch "$INSTALLED_ROOT/bun.lock"
assert_runtime 'published Bun install root' bun
touch "$INSTALLED_ROOT/node_modules/.package-lock.json"
assert_runtime 'published npm install metadata' node
PACKAGE="$ORIGINAL_PACKAGE"
set +e
QMD_TEST_EXIT=37 QMD_RUNTIME=node "$PACKAGE/bin/qmd" >/dev/null
status=$?
set -e
[[ "$status" -eq 37 ]]
printf 'OK: real launcher preserves child exit status 37\n'
PASS=$((PASS + 1))
if QMD_RUNTIME=invalid "$PACKAGE/bin/qmd" >/dev/null 2>&1; then
  printf 'FAIL: invalid runtime accepted\n' >&2; exit 1
fi
rm "$PACKAGE/dist/cli/qmd.js"
if "$PACKAGE/bin/qmd" >"$WORK/missing.log" 2>&1; then
  printf 'FAIL: missing build accepted\n' >&2; exit 1
fi
[[ "$(<"$WORK/missing.log")" == *'build missing'* ]]
printf 'OK: invalid runtime and missing build refuse\n'
PASS=$((PASS + 1))
printf '%s launcher checks passed\n' "$PASS"
