"""Saturate small e-graphs and print the graph each admitted rewrite produced.

Every test captures the rewrite events with ``capture_rewrites(echo=True)``, so
``pytest -s`` shows, for each union the e-graph admitted, the matched graph and
the new graph expanded down to the inputs. The assertions then check which
rewrites were admitted and which were not.
"""

from __future__ import annotations

from typing import Any

from axon.egraph.adapter import EGraphAdapter, Snapshot, reachable_classes
from axon.egraph.analysis import (
    analyze_snapshot,
    decode_isa,
    ensure_isa_semantics_registered,
)
from axon.egraph.codec import decode_tensor_enode, encode_isa_enode, encode_isa_input
from axon.egraph.fusion import encode_isa_candidate, is_eligible_isa_op
from axon.egraph.proof import ProofStore
from axon.egraph.propagation import run_propagation_round
from axon.egraph.rewrite_trace import (
    RewriteEvent,
    capture_rewrites,
    enable_from_cli,
    record_term_rewrite,
)
from axon.egraph.tensor import ingest_tensor_graph, saturate_tensor
from axon.ir import build_graph_from_kernel
from axon.isa_semantics import nl

_TIMEOUT = 2500


def _ingest(kernel: Any, *specs: Any, dim_sizes: dict[str, int]):
    G = build_graph_from_kernel(kernel, *specs, dim_sizes=dim_sizes)
    adapter = EGraphAdapter("t")
    return adapter, ingest_tensor_graph(adapter, G)


def _root_ops(snapshot: Snapshot, adapter: EGraphAdapter, handle: Any) -> list[str]:
    ref = adapter.resolve_handle(snapshot, handle)
    ops: list[str] = []
    for row in snapshot.members(ref):
        try:
            ops.append(decode_tensor_enode(snapshot, row).op)
        except Exception:
            continue
    return ops


def _new_graphs(events: list[RewriteEvent]) -> list[str]:
    return [event.after for event in events]


def test_tensor_transpose_of_matmul_prints_new_graph() -> None:
    def kernel(x, y):
        return (x @ y).transpose()

    adapter, ingest = _ingest(
        kernel, ("x", ("m", "k")), ("y", ("k", "n")), dim_sizes={"m": 4, "k": 8, "n": 6}
    )
    with capture_rewrites(echo=True) as events:
        status = saturate_tensor(adapter, timeout=_TIMEOUT)
    assert status.status == "fixed_point"
    snap = adapter.freeze_snapshot()
    (out_handle,) = ingest.output_handles
    assert "matmul" in _root_ops(snap, adapter, out_handle)

    assert events, "an admitted union must produce a rewrite event"
    # The new graph is a matmul of transposes, rendered down to the inputs.
    swapped = [g for g in _new_graphs(events) if g.startswith("matmul")]
    assert swapped
    assert "transpose" in swapped[0]
    assert "input 'x'" in swapped[0] and "input 'y'" in swapped[0]
    assert all(event.stage == "tensor_propagation" for event in events)
    assert [event.sequence for event in events] == sorted(
        event.sequence for event in events
    )


def test_tensor_softmax_never_swaps_with_transpose() -> None:
    def kernel(x):
        return x.softmax(axis=-1).transpose()

    adapter, ingest = _ingest(kernel, ("x", ("n", "n")), dim_sizes={"n": 4})
    with capture_rewrites(echo=True) as events:
        status = saturate_tensor(adapter, timeout=1500)
    assert status.status == "fixed_point"
    snap = adapter.freeze_snapshot()
    (out_handle,) = ingest.output_handles
    assert "softmax" not in _root_ops(snap, adapter, out_handle)
    assert not any(g.startswith("softmax") for g in _new_graphs(events))


def _seed(adapter: EGraphAdapter, name: str, shape: tuple[int, ...]) -> Any:
    return adapter.intern_expr(encode_isa_input(name, shape), provenance=name).handle


def _op(adapter: EGraphAdapter, op: str, attrs: dict, children: list, tag: str) -> Any:
    return adapter.intern_expr(
        encode_isa_enode(op, attrs, children), provenance=tag
    ).handle


def test_isa_propagation_moves_activation_through_transpose() -> None:
    ensure_isa_semantics_registered()
    adapter = EGraphAdapter("isa")
    x = _seed(adapter, "x", (4, 6))
    transposed = _op(adapter, "nc_transpose", {}, [x], "tr")
    out = _op(
        adapter,
        "activation",
        {"op": nl.relu, "scale": 1.0, "reduce_cmd": None, "with_reduce": False},
        [transposed],
        "relu",
    )
    with capture_rewrites(echo=True) as events:
        result = run_propagation_round(
            adapter,
            decode_isa,
            encode_isa_candidate,
            is_eligible_isa_op,
            ProofStore(),
            analyze=lambda s, d: analyze_snapshot(s, d),
            stage="isa_propagation",
            timeout=_TIMEOUT,
        )
    assert result.equalities_added >= 1
    assert events
    event = events[0]
    assert event.stage == "isa_propagation"
    assert event.before.startswith("activation")
    assert event.after.startswith("nc_transpose")
    assert "activation{" in event.after and "input 'x'" in event.after
    snap = adapter.freeze_snapshot()
    ref = adapter.resolve_handle(snap, out)
    assert ref in reachable_classes(snap, [ref])


def test_env_sink_appends_to_file(tmp_path, monkeypatch) -> None:
    path = tmp_path / "rewrites.log"
    monkeypatch.delenv("AXON_TRACE_REWRITES", raising=False)
    enable_from_cli(str(path))

    def kernel(x, y):
        return (x @ y).transpose()

    adapter, _ingest_result = _ingest(
        kernel, ("x", ("m", "k")), ("y", ("k", "n")), dim_sizes={"m": 4, "k": 8, "n": 6}
    )
    try:
        saturate_tensor(adapter, timeout=_TIMEOUT)
    finally:
        monkeypatch.delenv("AXON_TRACE_REWRITES", raising=False)
    text = path.read_text()
    assert "+++ new graph" in text and "matmul" in text


def test_tracing_off_records_nothing(monkeypatch) -> None:
    monkeypatch.delenv("AXON_TRACE_REWRITES", raising=False)

    class Boom:
        @property
        def snapshot(self):  # rendering would touch this
            raise AssertionError("rendered while tracing was off")

    record_term_rewrite("stage", Boom(), None, None, None)  # type: ignore[arg-type]


def test_cli_flag_parses() -> None:
    from axon.cli import _build_parser

    parser = _build_parser()
    base = ["kernels/mul", "--sizes", "4", "4"]
    assert parser.parse_args(base).trace_rewrites is None
    assert parser.parse_args([*base, "--trace-rewrites"]).trace_rewrites == ""
    assert (
        parser.parse_args([*base, "--trace-rewrites", "r.log"]).trace_rewrites
        == "r.log"
    )
