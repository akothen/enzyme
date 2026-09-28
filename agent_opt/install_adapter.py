#!/usr/bin/env python3
"""Install the versioned Axon-emitted test adapter into an nkilib checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "test_axon_emitted.py"
DESTINATION = Path("test/integration/nkilib/utils/test_axon_emitted.py")


def install(nkilib: Path, *, force: bool = False) -> str:
    root = nkilib.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"nkilib package directory not found: {root}")
    destination = root / DESTINATION
    source_bytes = SOURCE.read_bytes()
    if destination.is_file():
        if destination.read_bytes() == source_bytes:
            return "noop"
        if not force:
            raise SystemExit(
                f"{destination} exists and differs from {SOURCE}; "
                "use --force to replace it"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, destination)
    return "installed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nkilib", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    status = install(Path(args.nkilib), force=args.force)
    print(f"{status}: {Path(args.nkilib).expanduser().resolve() / DESTINATION}")


if __name__ == "__main__":
    main()
