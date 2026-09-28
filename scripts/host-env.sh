#!/usr/bin/env bash
# Build `.venv-host`: an environment that runs the WHOLE test suite without the
# Neuron toolchain and without a Trainium device.
#
# Why this exists. `uv sync` installs the project's hard dependencies, which
# include `neuronx-cc`, `torch-neuronx`, `nkipy`, and `spike` — about 9.5 GB, and
# none of it is reachable from a test. The suite needs `nki` only for
# `nki.simulate` (a CPU simulator) and `torch` only for the kernels' torch
# references. Measured on this tree: 807 of 807 tests pass here, the env is
# ~1.3 GB, and `nki` installs in about 2 s from the public Neuron index.
#
# The Makefile picks this env up automatically once it exists, so `make test`,
# `make ci`, and the hooks all use it.
#
#   ./scripts/host-env.sh          # create or update .venv-host
#
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

VENV=.venv-host
PY="$VENV/bin/python"

uv venv --python 3.11 "$VENV"

# Pure-python runtime deps, plus the test runner and the pinned gate tools.
uv pip install --python "$PY" \
    egglog numpy pandas tqdm ml_dtypes z3-solver \
    pytest pytest-xdist \
    'ruff==0.15.13' 'basedpyright==1.40.1'

# The NKI simulator. Public index; no neuronx-cc, no device.
uv pip install --python "$PY" \
    --extra-index-url https://pip.repos.neuron.amazonaws.com/ \
    --index-strategy unsafe-best-match \
    'nki>=0.3.0'

# CPU-only torch. The default wheel would pull a CUDA stack for nothing.
uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cpu torch

# `axon` itself, without its dependency list: the Neuron wheels above are exactly
# what this env leaves out.
uv pip install --python "$PY" --no-deps -e .

echo
"$PY" -c 'import axon, nki, torch, z3, egglog; print("host env ready:", axon.__file__)'
echo "run the suite with: make test"
