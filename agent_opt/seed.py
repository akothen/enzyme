#!/usr/bin/env python3
"""Seed (or re-seed) a kernel's pinned input_kernel.py from an Axon winner.

optimize.py optimizes the PINNED input_kernel.py in prompts/kernels/<kernel>/ — a
versioned snapshot, not a live read of the (gitignored, ephemeral) out/winners
dir. This copies a winner into that snapshot and stamps its provenance + a content
hash (input_sha) into meta.json.

Refuses to overwrite an existing, DIFFERENT input_kernel.py unless force=True, so a
re-seed can't silently change what an experiment optimized; a byte-identical
re-seed is a no-op.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "prompts" / "kernels"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def seed_kernel(kdir: Path, src: Path, *, force: bool = False) -> str:
    """Copy the winner at `src` into `kdir/input_kernel.py` and stamp meta.json.

    Returns "noop" (identical, nothing written) or "seeded" (written + stamped).
    Raises SystemExit if the target exists, differs, and force is False.
    """
    if not kdir.is_dir():
        raise SystemExit(f"no kernel bundle at {kdir} (create meta.json there first)")
    if not src.is_file():
        raise SystemExit(f"source winner not found: {src}")
    new = src.read_text()
    dest = kdir / "input_kernel.py"
    metap = kdir / "meta.json"

    if dest.is_file():
        if dest.read_text() == new:
            return "noop"
        if not force:
            note = ""
            if metap.is_file():
                prov = json.loads(metap.read_text()).get("provenance", {})
                if prov:
                    note = (
                        f"\n  (existing seeded {prov.get('seeded_utc', '?')} "
                        f"from {prov.get('source', '?')})"
                    )
            raise SystemExit(
                f"{dest} exists and DIFFERS from {src}.\n"
                f"Overwriting changes the pinned input; existing results/ for "
                f"'{kdir.name}' will no longer correspond to it.\n"
                f"Re-run with --force to replace.{note}"
            )

    dest.write_text(new)
    meta = json.loads(metap.read_text()) if metap.is_file() else {}
    meta["input_sha"] = _sha(new)
    meta["provenance"] = {
        "source": str(src),
        # noqa keeps the portable spelling: `_dt.UTC` needs 3.11, and this script
        # runs under whatever interpreter the caller has.
        "seeded_utc": _dt.datetime.now(_dt.timezone.utc).strftime(  # noqa: UP017
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }
    metap.write_text(json.dumps(meta, indent=2) + "\n")
    return "seeded"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kernel", required=True, help="dir under prompts/kernels/")
    ap.add_argument(
        "--from",
        dest="src",
        required=True,
        help="Axon winner NKI file to seed from (e.g. out/winners/<stem>.py)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing, different input_kernel.py",
    )
    args = ap.parse_args()

    kdir = KERNELS / args.kernel
    src = Path(args.src)
    status = seed_kernel(kdir, src, force=args.force)
    if status == "noop":
        print(
            f"input_kernel.py already up to date (sha {_sha(src.read_text())[:12]}); no change."
        )
    else:
        print(
            f"seeded {kdir / 'input_kernel.py'} from {src} "
            f"(sha {_sha(src.read_text())[:12]}); stamped meta.json."
        )


if __name__ == "__main__":
    main()
