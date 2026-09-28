"""Generate the per-case nkilib test-harness module
(`out/winners/<kernel>_<case_id>_case.py`).

The generated module is what nkilib's `TestAxonEmitted` imports for one
head-to-head case: `torch_ref` / `make_inputs` / `output_tensors` plus a `CASE`
metadata dict that points at the sibling winner kernel by relative path.

It loads the real kernel spec to reuse its refs (single source of truth). The
spec imports `axon`/`nkipy` at module top level and nkilib's test runtime has
neither, so the generated module first installs `sys.modules` stand-ins for
those imports (see `_SHIM_BLOCK`), then loads the spec. The stand-ins are inert
because the harness only reads the spec's numpy/torch refs: `AxonArray` is never
instantiated (`axon_kernel` is never called), the stubbed `nkipy.bfloat16` is an
overridden `make_inputs` default, and the stand-in `KernelSpec` only holds its
constructor kwargs."""

from __future__ import annotations

import json
import os
from pathlib import Path

from axon.kernel_spec import SEED, KernelSpec


def _input_names(spec: KernelSpec) -> list[str]:
    return [name for name, _ in spec.input_specs]


def generate_case_module(
    spec: KernelSpec,
    kernel_path: Path | str,
    dtype: str,
    dest: Path,
    *,
    name: str,
    entry_point: str,
    shape_args: dict[str, int],
    rtol: float,
    atol: float,
) -> None:
    """Write the per-case harness module to `dest`. The spec path is baked
    relative to `dest` and re-resolved at import.

    `name` is the case id (the `<stem>`), `entry_point` the kernel fn name in the
    sibling module, `shape_args` the merged dims + winning tile values, and
    `rtol`/`atol` the case tolerances. These populate the module's `CASE` dict.
    Requires `spec.torch_ref` and `spec.baseline_op` (head-to-head only)."""
    if spec.torch_ref is None or spec.baseline_op is None:
        raise ValueError(
            f"{spec.name}: head-to-head case needs both spec.torch_ref and "
            f"spec.baseline_op; got torch_ref={spec.torch_ref!r}, "
            f"baseline_op={spec.baseline_op!r}"
        )
    kabs = Path(kernel_path).resolve()
    # Directory-layout kernel: the harness loads the axon-free `refs.py`
    # directly (module attrs stand in for the spec), so no import shims are
    # needed. Single-file specs import axon/nkipy at top level and keep them.
    if kabs.is_dir():
        refs = kabs / "refs.py"
        if not refs.is_file():
            raise ValueError(f"{spec.name}: kernel dir {kabs} has no refs.py")
        kabs = refs
        shim_block = (
            "def _install_spec_import_shims():\n"
            "    pass  # refs.py is axon-free; no shims needed"
        )
    else:
        shim_block = _SHIM_BLOCK
    dest = dest.resolve()
    # relpath spans `..` segments (out/winners -> ../../kernels), unlike relative_to.
    krel = os.path.relpath(kabs, dest.parent)
    input_names = _input_names(spec)
    tile_args = list(spec.tile_args)
    dim_vars = list(spec.dim_vars)

    # These names are emitted as bare code tokens (fn params, kwargs keys), so a
    # non-identifier would make the generated module a SyntaxError far from here.
    for label, names in (
        ("input", input_names),
        ("dim", dim_vars),
        ("tile", tile_args),
    ):
        bad = [n for n in names if not n.isidentifier()]
        if bad:
            raise ValueError(
                f"{spec.name}: {label} names are not valid identifiers: {bad}"
            )

    tile_params = ", ".join(tile_args)
    torch_ref_params = ", ".join([*input_names, *tile_args])
    make_inputs_params = ", ".join([*dim_vars, *tile_args])

    coerced_data_pass = ", ".join(f"_to_torch_f32({n})" for n in input_names)
    dim_pass = ", ".join(f"{d}={d}" for d in dim_vars)
    tile_pass_dict = ", ".join(f'"{t}": int({t})' for t in tile_args)
    # `del` the tile kwargs so they're "used" (they are intentionally ignored).
    del_tiles = f"    del {tile_params}\n" if tile_args else ""

    # The winner kernel lives beside this module, named "<stem>.py".
    case = {
        "name": name,
        "entry_point": entry_point,
        "kernel_rel": f"{name}.py",
        "refs_rel": krel,
        "dtype": dtype,
        "input_names": input_names,
        "shape_args": shape_args,
        "rtol": rtol,
        "atol": atol,
    }
    case_literal = json.dumps(case, indent=4)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        _MODULE_TEMPLATE.format(
            kernel_name=spec.name,
            dtype=dtype,
            seed=SEED,
            shim_block=shim_block,
            kernel_rel=krel,
            input_names=repr(input_names),
            torch_ref_params=torch_ref_params,
            del_tiles=del_tiles,
            coerced_data_pass=coerced_data_pass,
            make_inputs_params=make_inputs_params,
            dim_pass=dim_pass,
            tile_pass_dict=tile_pass_dict,
            case_literal=case_literal,
        )
    )


# Emitted verbatim into each case module (see this module's docstring for why).
# Inserted as a pre-formatted value into the template, so it must contain no
# `str.format` fields of its own.
_SHIM_BLOCK = '''\
def _install_spec_import_shims():
    """Stub the spec's axon/nkipy imports so it loads where neither exists."""
    import sys as _sys
    import types as _types

    if "nkipy.core.language" not in _sys.modules:
        _nkipy = _sys.modules.setdefault("nkipy", _types.ModuleType("nkipy"))
        _core = _sys.modules.setdefault("nkipy.core", _types.ModuleType("nkipy.core"))
        _lang = _types.ModuleType("nkipy.core.language")
        _lang.bfloat16 = ml_dtypes.bfloat16
        _nkipy.core = _core
        _core.language = _lang
        _sys.modules["nkipy.core.language"] = _lang
    if "axon" not in _sys.modules:
        _axon = _types.ModuleType("axon")
        _axon.AxonArray = object

        class _KernelSpec:  # stores constructor kwargs; that is all we read
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        _ks = _types.ModuleType("axon.kernel_spec")
        _ks.KernelSpec = _KernelSpec
        _axon.kernel_spec = _ks
        _sys.modules["axon"] = _axon
        _sys.modules["axon.kernel_spec"] = _ks'''


_MODULE_TEMPLATE = '''\
"""Generated by axon.case_gen for kernel {kernel_name!r} (dtype={dtype!r}).
Do not edit."""

import importlib.util as _ilu
import os as _os

import ml_dtypes  # noqa: F401 — registers bf16 so np.dtype("bfloat16") resolves
import numpy as np
import torch

_SEED = {seed}
_DTYPE = "{dtype}"
_KERNEL_PATH = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), {kernel_rel!r})
)
_INPUT_NAMES = {input_names}

CASE = {case_literal}

_torch_dt = getattr(torch, _DTYPE)
_np_dt = np.dtype(_DTYPE)


def _to_torch_f32(a):
    """Normalize an input to an fp32 torch tensor before the spec's torch_ref.

    Order matters for bf16: torch can't ingest numpy's ml_dtypes bfloat16
    directly (TypeError in torch.from_numpy), so widen to fp32 on the numpy
    side first. fp32 is the reference convention regardless: the reference
    must not accumulate bf16 intermediate-rounding error the kernel under
    test does not incur. Centralizing this here keeps every spec's torch_ref
    pure math (already-fp32-torch in, tensor out)."""
    if isinstance(a, torch.Tensor):
        return a.to(torch.float32)
    return torch.from_numpy(np.asarray(a).astype(np.float32))


{shim_block}


def _load_spec():
    _install_spec_import_shims()
    _s = _ilu.spec_from_file_location("_axon_genref_{kernel_name}", _KERNEL_PATH)
    _m = _ilu.module_from_spec(_s)
    _s.loader.exec_module(_m)
    # Single-file specs expose SPEC; a directory kernel's refs.py exposes
    # torch_ref/baseline_op/make_inputs as module attrs and is its own "spec".
    return getattr(_m, "SPEC", _m)


_SPEC = _load_spec()


def torch_ref({torch_ref_params}):  # noqa: N803
{del_tiles}    result = _SPEC.torch_ref({coerced_data_pass})
    if not isinstance(result, tuple):
        result = (result,)
    return {{
        f"output_{{i}}": t.to(_torch_dt) for i, t in enumerate(result)
    }}


def make_inputs({make_inputs_params}):  # noqa: N803
    data = _SPEC.make_inputs({dim_pass}, dtype=_np_dt, rng=np.random.default_rng(_SEED))
    out = {{name: arr for name, arr in zip(_INPUT_NAMES, data)}}
    out.update({{{tile_pass_dict}}})
    return out


def output_tensors(kernel_input):
    ref = _SPEC.baseline_op(*[kernel_input[name] for name in _INPUT_NAMES])
    if not isinstance(ref, tuple):
        ref = (ref,)
    return {{
        f"output_{{i}}": np.zeros(np.asarray(a).shape, dtype=_np_dt)
        for i, a in enumerate(ref)
    }}
'''
