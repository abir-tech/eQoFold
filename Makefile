# eQFold -- WISER 2026 Moderna Challenge -- reproducibility entry point.
#
#   make env         create .venv and install the pinned environment
#   make sequences   regenerate the tier FASTA files from the global seed
#   make reference   fold every sequence -> data/references/vienna_reference.csv
#   make experiments every table on the model-fidelity track (no credentials)
#   make circuits    the GQE / PCE circuit-search track (needs the qms package)
#   make test        run the test suite
#   make verify      assert `make reference` reproduces the committed table
#   make all         everything above except `circuits`
#
# Windows users without GNU make: use the equivalent `./make.ps1 <target>`.

PYTHON ?= python
VENV   ?= .venv

ifeq ($(OS),Windows_NT)
  VPY := $(VENV)/Scripts/python.exe
else
  VPY := $(VENV)/bin/python
endif

REFERENCE := data/references/vienna_reference.csv

.PHONY: all help env install sequences reference experiments circuits figures \
        test lint verify clean distclean

all: sequences reference experiments figures test

help:
	@echo "targets: env sequences reference experiments circuits figures test lint verify clean"

$(VPY):
	$(PYTHON) -m venv $(VENV)
	$(VPY) -m pip install --upgrade pip

env: $(VPY)
	$(VPY) -m pip install -e ".[dev]"

install: env

sequences: | $(VPY)
	$(VPY) scripts/generate_sequences.py

reference: | $(VPY)
	$(VPY) scripts/build_reference.py

# Every results table. Runs on CPU, needs no credentials; ~15 min.
experiments: | $(VPY)
	$(VPY) experiments/ablate_enumeration.py --tiers A,M
	$(VPY) experiments/run_encoding_gap.py --tiers A,M --quiet
	$(VPY) experiments/run_fidelity_ladder.py --tiers A,M --max-stems 45
	$(VPY) experiments/run_dirac3_study.py --tiers A,M --max-stems 35 --seeds 3
	$(VPY) experiments/run_solvers.py --tiers A,M --max-stems 16 --budget 2.0
	$(VPY) experiments/run_advanced.py --max-stems 12

# The generative circuit-search track. Kept separate because it is the one
# part of the repository that needs the companion `qms` package (see README,
# "Installing"); everything under `experiments` above runs without it.
circuits: | $(VPY)
	$(VPY) experiments/scaling_sweep.py
	$(VPY) experiments/scaling_sweep_calibrated.py
	$(VPY) experiments/flagship_deep_dive.py
	$(VPY) experiments/hardware_aware_demo.py
	$(VPY) experiments/noise_robustness.py
	$(VPY) experiments/pseudoknot_illustration.py

# Figures read the committed tables only -- they never recompute science.
figures: | $(VPY)
	$(VPY) experiments/make_ladder_figures.py
	$(VPY) experiments/make_figures.py

test: | $(VPY)
	$(VPY) -m pytest

lint: | $(VPY)
	$(VPY) -m ruff check src tests scripts

# Phase 1 exit criterion: a clean clone must reproduce the reference table
# byte-for-byte. Regenerates into a temporary path and diffs.
verify: | $(VPY)
	$(VPY) scripts/build_reference.py --out /tmp/vienna_reference_check.csv
	diff -u $(REFERENCE) /tmp/vienna_reference_check.csv && \
	  echo "OK: reference table reproduces exactly"

clean:
	rm -rf .pytest_cache **/__pycache__ src/rnaqopt/__pycache__ tests/__pycache__

distclean: clean
	rm -rf $(VENV) *.egg-info src/*.egg-info
