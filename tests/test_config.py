"""Tests for the frozen configuration.

The configuration is the project's single point of failure for comparability:
if two results were produced under different model details their energies are
not comparable, and a table mixing them is invalid.  These tests make that
failure loud.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from rnaqopt.config import (
    GLOBAL_SEED,
    REPO_ROOT,
    STEMS,
    VIENNA,
    VIENNA_STOCK,
    StemConfig,
    ViennaConfig,
)


def test_config_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        VIENNA.temperature_c = 25.0  # type: ignore[misc]


def test_frozen_project_settings_are_the_documented_ones():
    assert VIENNA.temperature_c == 37.0
    assert VIENNA.dangles == 0
    assert VIENNA.no_lonely_pairs is True
    assert VIENNA.allow_gu is True
    assert VIENNA.param_set == "Turner2004"
    assert VIENNA.min_hairpin_loop == 3


def test_stem_defaults_are_the_measured_ones():
    """L_min stays at 3 (L_min=2 measurably worsens the encoding gap); sub-stems
    are ON, following the plan's own rule that they be enabled if the encoding
    gap analysis shows they matter -- it does. See config.StemConfig and
    results/tables/enumeration_ablation.csv."""
    assert STEMS.min_stem_length == 3
    assert STEMS.include_substems is True
    assert STEMS.pseudoknot_mode is False


def test_fingerprint_is_stable_across_calls():
    assert VIENNA.fingerprint() == VIENNA.fingerprint()
    assert len(VIENNA.fingerprint()) == 12


def test_fingerprint_changes_when_any_setting_changes():
    base = VIENNA.fingerprint()
    for field in dataclasses.fields(ViennaConfig):
        current = getattr(VIENNA, field.name)
        if isinstance(current, bool):
            other = not current
        elif isinstance(current, int | float):
            other = current + 1
        else:
            other = "Turner1999"
        if field.name == "param_set":
            other = "Turner1999"
        changed = dataclasses.replace(VIENNA, **{field.name: other})
        assert changed.fingerprint() != base, field.name


def test_primary_and_stock_configs_are_distinguishable():
    assert VIENNA.fingerprint() != VIENNA_STOCK.fingerprint()


def test_header_line_reports_the_settings_that_matter():
    line = VIENNA.header_line()
    for token in ("Turner2004", "T=37C", "dangles=0", "noLP=1", VIENNA.fingerprint()):
        assert token in line


def test_model_details_applies_every_setting():
    md = VIENNA.model_details()
    assert md.temperature == pytest.approx(VIENNA.temperature_c)
    assert md.dangles == VIENNA.dangles
    assert md.noLP == int(VIENNA.no_lonely_pairs)
    assert md.noGU == int(not VIENNA.allow_gu)
    assert md.special_hp == int(VIENNA.special_hairpins)
    assert md.min_loop_size == VIENNA.min_hairpin_loop


def test_unsupported_parameter_set_is_rejected():
    with pytest.raises(ValueError):
        ViennaConfig(param_set="Turner1999").model_details()


def test_stem_config_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        STEMS.min_stem_length = 2  # type: ignore[misc]


def test_stem_config_can_be_varied_for_ablations():
    ablation = StemConfig(min_stem_length=2, include_substems=True)
    assert ablation.min_stem_length == 2
    assert STEMS.min_stem_length == 3  # the global default is untouched


def test_repo_root_looks_like_the_repository():
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert (REPO_ROOT / "src" / "rnaqopt").is_dir()


def test_global_seed_is_an_int():
    assert isinstance(GLOBAL_SEED, int)


def test_requirements_mirror_pyproject():
    """requirements.txt is a pinned mirror of pyproject; drift between them is
    a reproducibility bug."""
    import re

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    reqs = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

    declared = set(re.findall(r'"([A-Za-z][\w.-]*==[\d.]+)"', pyproject))
    mirrored = {
        line.strip()
        for line in reqs.splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert declared == mirrored, (
        f"only in pyproject: {sorted(declared - mirrored)}; "
        f"only in requirements.txt: {sorted(mirrored - declared)}"
    )


def test_generated_directories_are_under_the_repo():
    from rnaqopt.config import REFERENCE_DIR, RESULTS_DIR, SEQUENCE_DIR

    for d in (SEQUENCE_DIR, REFERENCE_DIR, RESULTS_DIR):
        assert Path(REPO_ROOT) in Path(d).parents
