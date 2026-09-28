"""Per-op ISA emitters and the small name/ref helpers they need.

Each emitter turns one hardware-graph ``Node`` into a single NKI call string with
a ``{DST}`` placeholder the body builder substitutes with the destination tile
slice.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from axon.codegen.context import EmitCtx
from axon.ir import Node

OpEmitter = Callable[[EmitCtx, Node, dict[str, str]], str]


class UnsupportedEmission(Exception):
    """A hardware-graph node cannot be faithfully emitted to NKI."""


_NL_OP_MAP: dict[str, str] = {
    "add": "nl.add",
    "copy": "nl.copy",
    "divide": "nl.divide",
    "div": "nl.divide",
    "exp": "nl.exp",
    "gelu": "nl.gelu",
    "maximum": "nl.maximum",
    "minimum": "nl.minimum",
    "mul": "nl.multiply",
    "multiply": "nl.multiply",
    "relu": "nl.relu",
    "sigmoid": "nl.sigmoid",
    "silu": "nl.silu",
    "sqrt": "nl.sqrt",
    "subtract": "nl.subtract",
    "tanh": "nl.tanh",
}


def _nki_op_ref(op: Any) -> str:
    if op is None:
        return "None"
    name: str = ""
    if hasattr(op, "name") and isinstance(op.name, str):
        name = op.name
    elif isinstance(op, str):
        name = op
    else:
        name = str(op)
    return _NL_OP_MAP.get(name, f"nl.{name}")


def nki_safe_var(node_id: str) -> str:
    result = node_id.replace("-", "_").replace(".", "_").replace("/", "_")
    result = result.lstrip("_")
    if not result or result[0].isdigit():
        result = "v" + result
    return result


def _strip_emitted_comments(lines: list[str]) -> list[str]:
    result: list[str] = []
    blank_run = 0
    for line in lines:
        if line.strip().startswith("#"):
            continue
        cleaned = re.sub(r"  #.*$", "", line)
        if cleaned.strip() == "":
            blank_run += 1
            if blank_run <= 1:
                result.append(cleaned)
        else:
            blank_run = 0
            result.append(cleaned)
    return result


def resolve_input(node: Node, id_to_var: dict[str, str], index: int) -> str:
    if index >= len(node.inputs):
        raise ValueError(
            f"NKIEmitter: node {node.id!r} op={node.op!r} has only "
            f"{len(node.inputs)} input(s); requested index {index}"
        )
    inp_id = node.inputs[index]
    if inp_id not in id_to_var:
        raise ValueError(
            f"NKIEmitter: input {inp_id!r} (index {index}) for node "
            f"{node.id!r} op={node.op!r} has not been emitted yet — "
            "ensure the graph is in topological order"
        )
    return id_to_var[inp_id]


def input_is_partition_scalar(ctx: EmitCtx, node: Node, index: int, var: str) -> bool:
    """True when input ``index`` of ``node`` is a per-partition (P, 1) scalar."""
    return index < len(node.inputs) and node.inputs[index] in ctx.partition_scalar_ids


def _emit_nc_matmul(ctx: EmitCtx, node: Node, id_to_var: dict[str, str]) -> str:
    stationary = resolve_input(node, id_to_var, 0)
    moving = resolve_input(node, id_to_var, 1)
    attrs = node.attrs
    extra: list[str] = []
    if attrs.get("is_transpose"):
        extra.append("is_transpose=True")
    if attrs.get("accumulate") is True:
        extra.append("accumulate=True")
    tail = (", " + ", ".join(extra)) if extra else ""
    return f"nisa.nc_matmul({{DST}}, {stationary}, {moving}{tail})"


def _emit_nc_transpose(ctx: EmitCtx, node: Node, id_to_var: dict[str, str]) -> str:
    data = resolve_input(node, id_to_var, 0)
    return f"nisa.nc_transpose({{DST}}, {data})"


def _emit_activation(ctx: EmitCtx, node: Node, id_to_var: dict[str, str]) -> str:
    attrs = node.attrs
    data = resolve_input(node, id_to_var, 0)
    op_ref = _nki_op_ref(attrs.get("op"))
    extra: list[str] = []
    if "bias_input_index" in attrs:
        bias_var = resolve_input(node, id_to_var, attrs["bias_input_index"])
        extra.append(f"bias={bias_var}")
    elif "bias_const" in attrs and attrs["bias_const"] is not None:
        extra.append(f"bias={attrs['bias_const']!r}")
    scale = attrs.get("scale", 1.0)
    if "scale_input_index" in attrs:
        scale_var = resolve_input(node, id_to_var, attrs["scale_input_index"])
        extra.append(f"scale={scale_var}")
    elif scale != 1.0:
        extra.append(f"scale={scale!r}")
    tail = (", " + ", ".join(extra)) if extra else ""
    return f"nisa.activation({{DST}}, {op_ref}, {data}{tail})"


def _activation_reduce_call(
    ctx: EmitCtx,
    node: Node,
    id_to_var: dict[str, str],
    act_dst: str | None,
) -> str:
    """NKI: activation_reduce(dst, op, data, reduce_op, reduce_res, bias, scale).

    The node's e-class value is the (P, 1) reduce result, so {DST} binds to
    ``reduce_res``. ``act_dst`` is where the (P, F) elementwise activation output
    lands: a caller-supplied destination when some consumer wants that value, and
    otherwise a throwaway scratch tile shaped like the input ``data`` (its shape,
    NOT {DST_SHAPE}=(P, 1)), following the scratch-tile idiom of
    _emit_tensor_tensor's divide split.
    """
    attrs = node.attrs
    data = resolve_input(node, id_to_var, 0)
    op_ref = _nki_op_ref(attrs.get("op"))
    reduce_op_ref = _nki_op_ref(attrs.get("reduce_op"))
    extra: list[str] = []
    if "bias_input_index" in attrs:
        bias_var = resolve_input(node, id_to_var, attrs["bias_input_index"])
        extra.append(f"bias={bias_var}")
    elif "bias_const" in attrs and attrs["bias_const"] is not None:
        extra.append(f"bias={attrs['bias_const']!r}")
    scale = attrs.get("scale", 1.0)
    if "scale_input_index" in attrs:
        scale_var = resolve_input(node, id_to_var, attrs["scale_input_index"])
        extra.append(f"scale={scale_var}")
    elif scale != 1.0:
        extra.append(f"scale={scale!r}")
    tail = (", " + ", ".join(extra)) if extra else ""
    if act_dst is not None:
        return (
            f"nisa.activation_reduce({act_dst}, {op_ref}, {data},"
            f" {reduce_op_ref}, reduce_res={{DST}}{tail})"
        )
    act = f"{nki_safe_var(node.id)}_act"
    return (
        f"{act} = nl.ndarray({data}.shape, dtype=nl.float32, buffer=nl.sbuf)\n"
        f"{{IND}}nisa.activation_reduce({act}, {op_ref}, {data},"
        f" {reduce_op_ref}, reduce_res={{DST}}{tail})"
    )


def _emit_activation_reduce(ctx: EmitCtx, node: Node, id_to_var: dict[str, str]) -> str:
    """The default registry entry: the activation output is a throwaway tile."""
    return _activation_reduce_call(ctx, node, id_to_var, act_dst=None)


def _emit_tensor_reduce(ctx: EmitCtx, node: Node, id_to_var: dict[str, str]) -> str:
    attrs = node.attrs
    data = resolve_input(node, id_to_var, 0)
    op_ref = _nki_op_ref(attrs.get("op"))
    axis = attrs.get("axis")
    extra: list[str] = []
    if attrs.get("negate"):
        extra.append("negate=True")
    if attrs.get("keepdims"):
        extra.append("keepdims=True")
    tail = (", " + ", ".join(extra)) if extra else ""
    # NKI signature: nisa.tensor_reduce(dst, op, data, axis, ...).
    return f"nisa.tensor_reduce({{DST}}, {op_ref}, {data}, axis={axis!r}{tail})"


def _emit_tensor_partition_reduce(
    ctx: EmitCtx, node: Node, id_to_var: dict[str, str]
) -> str:
    attrs = node.attrs
    data = resolve_input(node, id_to_var, 0)
    op_ref = _nki_op_ref(attrs.get("op"))
    return f"nisa.tensor_partition_reduce({{DST}}, {op_ref}, {data})"


def _emit_tensor_scalar(ctx: EmitCtx, node: Node, id_to_var: dict[str, str]) -> str:
    attrs = node.attrs
    data = resolve_input(node, id_to_var, 0)
    op0 = attrs.get("op0")
    op0_ref = _nki_op_ref(op0)
    extra: list[str] = []
    reverse0 = bool(attrs.get("reverse0"))
    if "operand0_input_index" in attrs:
        op0_idx = attrs["operand0_input_index"]
        op0_var = resolve_input(node, id_to_var, op0_idx)
        # nisa.tensor_scalar wants `data` as the (P, F) tile and `operand0` as
        # the (P, 1) scalar, so swap when the synthesizer emitted them reversed.
        data_is_scalar = input_is_partition_scalar(ctx, node, 0, data)
        op0_is_scalar = input_is_partition_scalar(ctx, node, op0_idx, op0_var)
        if data_is_scalar and not op0_is_scalar:
            data, op0_var = op0_var, data
            reverse0 = not reverse0
        extra.append(f"operand0={op0_var}")
    elif "operand0_const" in attrs:
        const_val = attrs["operand0_const"]
        extra.append(f"operand0={const_val!r}")
    if reverse0:
        extra.append("reverse0=True")
    op1 = attrs.get("op1")
    if op1 is not None:
        extra.append(f"op1={_nki_op_ref(op1)}")
        if "operand1_input_index" in attrs:
            op1_var = resolve_input(node, id_to_var, attrs["operand1_input_index"])
            extra.append(f"operand1={op1_var}")
        elif "operand1_const" in attrs and attrs["operand1_const"] is not None:
            extra.append(f"operand1={attrs['operand1_const']!r}")
        if attrs.get("reverse1"):
            extra.append("reverse1=True")
    tail = (", " + ", ".join(extra)) if extra else ""
    # NKI signature: nisa.tensor_scalar(dst, data, op0, operand0, ...).
    return f"nisa.tensor_scalar({{DST}}, {data}, {op0_ref}{tail})"


def _emit_scalar_tensor_tensor(
    ctx: EmitCtx, node: Node, id_to_var: dict[str, str]
) -> str:
    attrs = node.attrs
    data = resolve_input(node, id_to_var, 0)
    op0_ref = _nki_op_ref(attrs.get("op0"))
    op1_ref = _nki_op_ref(attrs.get("op1"))
    op0_var: str | None = None
    if "operand0_input_index" in attrs:
        op0_var = resolve_input(node, id_to_var, attrs["operand0_input_index"])
    op1_idx = attrs["operand1_input_index"]
    op1_var = resolve_input(node, id_to_var, op1_idx)
    # NKI contract: `data` and `operand1` are (P,F), `operand0` is the (P,1)
    # scalar. Swap `data` with whichever (P,F) operand when the synthesizer
    # emitted the scalar as `data`.
    op0_idx = attrs.get("operand0_input_index")
    data_is_scalar = input_is_partition_scalar(ctx, node, 0, data)
    op1_is_scalar = input_is_partition_scalar(ctx, node, op1_idx, op1_var)
    op0_is_scalar = (
        op0_var is not None
        and op0_idx is not None
        and input_is_partition_scalar(ctx, node, op0_idx, op0_var)
    )
    if data_is_scalar and not op1_is_scalar:
        data, op1_var = op1_var, data
        data_is_scalar, op1_is_scalar = False, True
    if data_is_scalar and op0_var is not None and not op0_is_scalar:
        data, op0_var = op0_var, data
    parts: list[str] = ["dst={DST}"]
    parts += [f"data={data}", f"op0={op0_ref}"]
    if op0_var is not None:
        parts.append(f"operand0={op0_var}")
    elif "operand0_const" in attrs:
        parts.append(f"operand0={attrs['operand0_const']!r}")
    if attrs.get("reverse0"):
        parts.append("reverse0=True")
    parts.append(f"op1={op1_ref}")
    parts.append(f"operand1={op1_var}")
    if attrs.get("reverse1"):
        parts.append("reverse1=True")
    return f"nisa.scalar_tensor_tensor({', '.join(parts)})"


def _emit_tensor_tensor(ctx: EmitCtx, node: Node, id_to_var: dict[str, str]) -> str:
    attrs = node.attrs
    data1 = resolve_input(node, id_to_var, 0)
    data2 = resolve_input(node, id_to_var, 1)
    op_name = (
        attrs["op"].name if hasattr(attrs.get("op"), "name") else str(attrs.get("op"))
    )
    # nisa.tensor_tensor has no op=divide, so lower as data1 * (1 / data2).
    is_divide = op_name in ("divide", "div") or op_name.endswith(".divide")
    if is_divide:
        # Scratch tile named per node id so chained divides don't collide.
        tmp = f"{nki_safe_var(node.id)}_recip"
        return (
            f"{tmp} = nl.ndarray(({{DST_SHAPE}}), dtype=nl.float32,"
            " buffer=nl.sbuf)\n"
            f"{{IND}}nisa.reciprocal({tmp}[{{DST_SLICE}}], {data2})\n"
            f"{{IND}}nisa.tensor_tensor({{DST}}, {data1},"
            f" {tmp}[{{DST_SLICE}}], nl.multiply)"
        )
    op_ref = _nki_op_ref(attrs.get("op"))
    return f"nisa.tensor_tensor({{DST}}, {data1}, {data2}, {op_ref})"


def _emit_tensor_scalar_cumulative(
    ctx: EmitCtx, node: Node, id_to_var: dict[str, str]
) -> str:
    attrs = node.attrs
    src = resolve_input(node, id_to_var, 0)
    op0_ref = _nki_op_ref(attrs.get("op0"))
    op1_ref = _nki_op_ref(attrs.get("op1"))
    if "imm0_input_index" in attrs:
        imm0 = resolve_input(node, id_to_var, attrs["imm0_input_index"])
    else:
        # imm0 must be FP32 per ISA; Python `0.0` would be fp64.
        imm0_const = attrs.get("imm0_const", 0)
        if isinstance(imm0_const, (int, float)):
            imm0 = f"np.float32({float(imm0_const)!r})"
        else:
            imm0 = repr(imm0_const)
    return (
        "nisa.tensor_scalar_cumulative("
        f"dst={{DST}}, src={src}, op0={op0_ref}, op1={op1_ref}, imm0={imm0})"
    )


def _emit_dma_copy(ctx: EmitCtx, node: Node, id_to_var: dict[str, str]) -> str:
    data = resolve_input(node, id_to_var, 0)
    return f"nisa.dma_copy(dst={{DST}}, src={data})"


def _emit_dma_transpose(ctx: EmitCtx, node: Node, id_to_var: dict[str, str]) -> str:
    data = resolve_input(node, id_to_var, 0)
    attrs = node.attrs
    axes = attrs.get("axes")
    # Unused today (the synthesizer routes transpose to nc_transpose).
    if axes is not None:
        return f"nisa.dma_transpose({{DST}}, {data}, axes={axes!r})"
    return f"nisa.dma_transpose({{DST}}, {data})"


def _emit_tensor_copy(ctx: EmitCtx, node: Node, id_to_var: dict[str, str]) -> str:
    data = resolve_input(node, id_to_var, 0)
    return f"nisa.tensor_copy({{DST}}, {data})"


def _emit_exponential(ctx: EmitCtx, node: Node, id_to_var: dict[str, str]) -> str:
    data = resolve_input(node, id_to_var, 0)
    return f"nisa.activation({{DST}}, nl.exp, {data})"


def _emit_reciprocal(ctx: EmitCtx, node: Node, id_to_var: dict[str, str]) -> str:
    data = resolve_input(node, id_to_var, 0)
    return f"nisa.reciprocal({{DST}}, {data})"


_OP_EMITTERS: dict[str, OpEmitter] = {
    "nc_matmul": _emit_nc_matmul,
    "nc_transpose": _emit_nc_transpose,
    "activation": _emit_activation,
    "activation_reduce": _emit_activation_reduce,
    "exponential": _emit_exponential,
    "reciprocal": _emit_reciprocal,
    "tensor_reduce": _emit_tensor_reduce,
    "tensor_partition_reduce": _emit_tensor_partition_reduce,
    "scalar_tensor_tensor": _emit_scalar_tensor_tensor,
    "tensor_scalar": _emit_tensor_scalar,
    "tensor_scalar_cumulative": _emit_tensor_scalar_cumulative,
    "tensor_tensor": _emit_tensor_tensor,
    "dma_copy": _emit_dma_copy,
    "dma_transpose": _emit_dma_transpose,
    "tensor_copy": _emit_tensor_copy,
}


def emit_node_call(ctx: EmitCtx, node: Node, id_to_var: dict[str, str]) -> str | None:
    if node.op == "input":
        return None
    emitter = _OP_EMITTERS.get(node.op)
    if emitter is not None:
        return emitter(ctx, node, id_to_var)
    raise UnsupportedEmission(
        f"NKIEmitter: no emitter registered for op {node.op!r} (id={node.id})"
    )


def emit_activation_reduce_to(
    ctx: EmitCtx, node: Node, id_to_var: dict[str, str], act_dst: str
) -> str:
    """`activation_reduce` whose (P, F) activation output lands in `act_dst`.

    A caller with a real destination for that value (a staged buffer slice) asks
    for it here, so the emitter writes the final call in one pass."""
    if node.op != "activation_reduce":
        raise UnsupportedEmission(
            f"NKIEmitter: {node.id!r} is {node.op!r}, not activation_reduce"
        )
    try:
        return _activation_reduce_call(ctx, node, id_to_var, act_dst=act_dst)
    except ValueError as e:
        raise UnsupportedEmission(f"cannot emit {node.op} (id={node.id}): {e}") from e


def emit_or_raise(ctx: EmitCtx, node: Node, id_to_var: dict[str, str]) -> str | None:
    """`emit_node_call`, but a `ValueError` from an unlowerable node is
    re-raised as `UnsupportedEmission` so the whole variant is dropped."""
    try:
        return emit_node_call(ctx, node, id_to_var)
    except ValueError as e:
        raise UnsupportedEmission(f"cannot emit {node.op} (id={node.id}): {e}") from e
