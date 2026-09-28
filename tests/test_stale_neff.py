"""Host test: the compile worker must never report a pre-existing NEFF as
this run's compile result.

`_compile_one_nki_kernel` classifies success by "kernel.neff exists on disk"
(deliberately, because the recipe writes the NEFF before post-compile steps
can raise). But NEFF dirs persist across runs, so after the kernel source is
re-emitted, a failed/skipped compile left the PREVIOUS emission's NEFF in
place and the worker returned it as success — the sweep then benched machine
code from stale source (relu_mlp 2026-07-15: 160/256 configs benched July-14
NEFFs against re-emitted refs, max_abs_err up to 183k vs 763 when actually
recompiled). The worker must clear a pre-existing NEFF before compiling so
"NEFF exists" can only mean "this compile produced it".
"""

from __future__ import annotations

import pytest

from axon.bench_runner import _compile_one_nki_kernel


def test_preexisting_neff_is_cleared_before_compile(tmp_path):
    # Plant a stale NEFF where the worker will look for its output (a
    # per-config subdir, mirroring RunPaths.neff's layout).
    neff_dir = tmp_path / "__v0_t0_tiles_1_1_1"
    neff_dir.mkdir()
    neff = neff_dir / "kernel.neff"
    neff.write_bytes(b"stale bytes from a previous emission")

    # A module that fails to load aborts the worker before any compile (the
    # loader raises out of the worker) — the stale NEFF must already be gone
    # by then, so it can never be classified as this run's output.
    missing_module = tmp_path / "does_not_exist.py"
    with pytest.raises(FileNotFoundError):
        _compile_one_nki_kernel(
            (
                str(missing_module),
                "kernel_fn",
                (1, 1, 1),
                (),
                {},
                str(neff),
                None,
                1,
            )
        )

    assert not neff.exists(), (
        "pre-existing kernel.neff survived into the compile attempt — a "
        "failed/skipped compile would report it as success and the sweep "
        "would bench stale machine code"
    )


def test_failed_compile_returns_no_neff_despite_stale_one(tmp_path):
    # The worker's internal error path: the module loads, but it is not a
    # real @nki.jit kernel, so the compile attempt fails INSIDE the worker's
    # try block and it returns (key, None, err) instead of raising. With a
    # stale NEFF planted, the pre-fix worker classified this failure as
    # success (file exists on disk); now the dir is cleared first.
    # Mirror the real layout: the variant module lives in the run dir, the
    # NEFF in a per-config subdir (which the worker clears wholesale).
    neff_dir = tmp_path / "__v0_t0_tiles_1_1_1"
    neff_dir.mkdir()
    neff = neff_dir / "kernel.neff"
    neff.write_bytes(b"stale bytes from a previous emission")

    bogus_module = tmp_path / "bogus_kernel.py"
    bogus_module.write_text("def kernel_fn(x):\n    return x\n")

    key = (1, 1, 1)
    result_key, result_neff, err = _compile_one_nki_kernel(
        (
            str(bogus_module),
            "kernel_fn",
            key,
            (),
            {},
            str(neff),
            None,
            1,
        )
    )

    assert result_key == key
    assert result_neff is None, (
        "failed compile reported the pre-existing stale NEFF as its product"
    )
    assert err
    assert not neff.exists()
