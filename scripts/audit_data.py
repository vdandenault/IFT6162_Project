"""Compatibility entry point; prefer lake-como-audit."""
from lake_como.audit import main

if __name__ == "__main__":
    raise SystemExit(main())
