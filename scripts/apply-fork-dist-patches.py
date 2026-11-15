#!/usr/bin/env python3
"""Retired compatibility entrypoint. Production files must come from source builds."""
import sys


def main() -> int:
    print(
        "error: apply-fork-dist-patches.py is retired; it never modifies installed packages. "
        "Run npm run build and npm pack, then deliberately install the resulting package. "
        "MCP origin protection is versioned in src/mcp/server.ts. "
        "Unversioned home-directory patches are no longer supported.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
