# Data-driven case targets.
#
# Cases live in eval/sizes.json; eval/gen_size_targets.py turns each
# kernel/case into a file target `out/<kernel>_<case_id>.csv` (recipe runs
# `axon`), a phony alias `<kernel>_<case_id>`, a per-kernel `CASES_<kernel>`,
# and the aggregate `CASE_TARGETS`.
# A kernel is either a single spec file kernels/<name>.py or a package
# directory kernels/<name>/ (detected by its __init__.py).
KERNELS := $(patsubst kernels/%.py,%,$(wildcard kernels/*.py)) \
           $(patsubst kernels/%/__init__.py,%,$(wildcard kernels/*/__init__.py))

AXON_ARGS ?=

.DELETE_ON_ERROR:
.PHONY: all lnc1 lnc2 clean format lint typecheck ci test test-fast test-device \
        host-env hooks _hook-ruff $(KERNELS) $(addprefix clean-,$(KERNELS))

# ---------------------------------------------------------------------------
# Interpreters and gate tools
#
# Tests run in the host-only env when it exists (`make host-env`: ~1.3 GB, no
# Neuron wheels, no device — it runs the whole suite), otherwise in the project
# env, which `uv run` syncs to ~9.5 GB because the Neuron wheels are hard
# dependencies. The gate tools run from uv's ephemeral cache at the pinned
# versions, so a fresh clone can lint and commit without syncing either env.
# ---------------------------------------------------------------------------
PY := $(if $(wildcard .venv-host/bin/python),.venv-host/bin/python,uv run python)
RUFF := uv run -q --no-project --with ruff==0.15.13 ruff
PYRIGHT := uv run -q --no-project --with basedpyright==1.40.1 basedpyright

# The commit-hook subset: pure-logic tests with no synthesis, no solver, and no
# simulation. 142 tests in under 2 s serially, which is what makes it usable on
# every commit. The full suite is the pre-push gate.
FAST_TESTS := tests/test_candidate_filter.py tests/test_run_case.py \
              tests/test_cli_sizes.py tests/test_agent_opt.py \
              tests/test_combined_agent_opt.py tests/test_bench_resume.py \
              tests/test_export_winner.py

# Regenerate the case fragment whenever sizes.json or the generator changes.
out/size_targets.mk: eval/sizes.json eval/gen_size_targets.py
	@mkdir -p out
	uv run python eval/gen_size_targets.py > $@

-include out/size_targets.mk

all: $(CASE_TARGETS)

# Run only the single-core (lnc=1) cases or only the sharded (lnc=2) cases.
# The split comes from each case's `lnc` key in eval/sizes.json.
lnc1: $(LNC1_TARGETS)

lnc2: $(LNC2_TARGETS)

format:
	$(RUFF) format .
	$(RUFF) check --fix .

lint:
	$(RUFF) format --check .
	$(RUFF) check .

# Not part of `ci` yet: 8 pre-existing errors in src (Optional narrowing in
# codegen/plan.py, codegen/bodies/matmul_generic.py, candidate_filter.py, and a
# pandas truthiness one in winner.py). Burn those down, then fold this into `ci`.
typecheck:
	$(PYRIGHT) src

# The gate. `ci` is the single definition of "green", so a hook, a human, and a
# future GitHub Actions job all check the same thing.
ci: lint test

test:
	$(PY) -m pytest tests/ -q

test-fast:
	$(PY) -m pytest $(FAST_TESTS) -q -n0

# Trainium only. Empty until device-marked tests exist; the marker is registered
# so a device test can never be picked up by `ci` or a hook by accident.
test-device:
	$(PY) -m pytest tests/ -q -m device

# Build the host-only env that runs the whole suite without the Neuron wheels.
host-env:
	./scripts/host-env.sh

# Internal, called by the commit hook with FILES=<staged paths>. `force-exclude`
# in pyproject keeps the generated trees (winners/, agent_opt/prompts/kernels)
# excluded even when a path is named explicitly.
_hook-ruff:
	@$(RUFF) format --check -- $(FILES)
	@$(RUFF) check -- $(FILES)

# Point git at the tracked hooks. Per clone, once.
hooks:
	git config core.hooksPath scripts/hooks
	@echo "hooks enabled: pre-commit runs lint + $(words $(FAST_TESTS)) fast test files, pre-push runs make ci"

# A bare `make <kernel>` is ambiguous now that a kernel has multiple cases, so
# it prints that kernel's case targets (from CASES_<kernel>) and exits 2.
$(KERNELS): %:
	@echo "Pick a case for '$*'. Available case targets:"
	@for c in $(CASES_$*); do echo "  make $$c"; done
	@exit 2

# `make clean-<kernel>` wipes every per-case artifact for that kernel.
$(addprefix clean-,$(KERNELS)): clean-%:
	@for c in $(CASES_$*); do \
		rm -f "out/$$c.csv"; \
		rm -rf "out/$$c/" "out/baseline_$$c/"; \
		rm -f "out/winners/$$c.py" "out/winners/$${c}_case.py"; \
	done

clean: $(addprefix clean-,$(KERNELS))
	rm -f out/size_targets.mk
