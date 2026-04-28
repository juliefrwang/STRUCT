"""Integration tests for the --wobble path through the target-mode pipeline.

These tests exercise the dispatch logic in:
- ``process_targets.process_df`` (stem finder selection)
- ``structure_target_mode.SS_target`` (mutation + p-value path)
- ``get_pval.target_p_outcome_ext`` and friends (anchor-p convolution)

A small synthetic SPLASH output TSV is generated on-the-fly so the test is
self-contained and doesn't depend on fixture files.
"""

import os
import shutil
import tempfile

import pandas as pd
import pytest
from pandarallel import pandarallel

from splash_structure_py.src.process_targets import process_df
from splash_structure_py.src.get_pval import (
    target_p_outcome,
    target_p_outcome_ext,
    prep_for_conv,
    prep_for_conv_ext,
)


# pandarallel must be initialised before parallel_apply is called.
# Use a single worker so tests are deterministic and fast.
pandarallel.initialize(nb_workers=1, progress_bar=False, verbose=0)


# ---------------------------------------------------------------------
# Synthetic SPLASH-output dataframe
# ---------------------------------------------------------------------

def _build_synthetic_splash_df():
    """Build a minimal dataframe matching the columns used by process_df.

    Two anchors. Each carries 4 most-frequent targets so it survives the
    abundance and num_target > 2 filters.

    Anchor 1 (`AAA...`) — most-freq target has a clean 5-bp WCF stem.
    Anchor 2 (`CCC...`) — most-freq target has a 5-bp stem with one G·U
    wobble; the strict-WCF finder still detects a stem here, but the
    wobble-aware finder will incorporate the wobble.
    """
    # Length-27 targets with stem at positions 0-4 / 22-26 and a long loop.
    # Anchor 1 base: "AGCAA" + 17-char loop + "TTGCT" (rc(AGCAA))
    a1_base = "AGCAA" + ("A" * 17) + "TTGCT"
    assert len(a1_base) == 27

    # Variants: one with a stem mismatch, others identical loop changes.
    # Keep enough Hamming variation so num_target > 2 and weights survive.
    a1_v1 = "AGCAA" + "A" * 16 + "T" + "TTGCT"   # 1 loop mut
    a1_v2 = "AGCAA" + "A" * 15 + "TT" + "TTGCT"  # 2 loop muts
    a1_v3 = "AGCAA" + "A" * 14 + "TTT" + "TTGCT"  # 3 loop muts

    # Anchor 2 base: stem with one wobble at pair 1 (G·T) — left "AGCAA"
    # / right "TTGTT" (so right-stem char 11 is T not C).
    a2_base = "AGCAA" + ("C" * 17) + "TTGTT"
    assert len(a2_base) == 27
    a2_v1 = "AGCAA" + "C" * 16 + "T" + "TTGTT"
    a2_v2 = "AGCAA" + "C" * 15 + "TT" + "TTGTT"
    a2_v3 = "AGCAA" + "C" * 14 + "TTT" + "TTGTT"

    # Build columns the pipeline expects.
    rows = [
        {
            "anchor": "ANCHOR1_AAAAAAAAAAAAAAAAAAAAAAA",
            "M": 1000,
            "most_freq_target_1": a1_base,
            "cnt_most_freq_target_1": 500,
            "most_freq_target_2": a1_v1,
            "cnt_most_freq_target_2": 250,
            "most_freq_target_3": a1_v2,
            "cnt_most_freq_target_3": 150,
            "most_freq_target_4": a1_v3,
            "cnt_most_freq_target_4": 100,
        },
        {
            "anchor": "ANCHOR2_CCCCCCCCCCCCCCCCCCCCCCC",
            "M": 800,
            "most_freq_target_1": a2_base,
            "cnt_most_freq_target_1": 400,
            "most_freq_target_2": a2_v1,
            "cnt_most_freq_target_2": 200,
            "most_freq_target_3": a2_v2,
            "cnt_most_freq_target_3": 120,
            "most_freq_target_4": a2_v3,
            "cnt_most_freq_target_4": 80,
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# process_df dispatch
# ---------------------------------------------------------------------

def test_process_df_default_uses_wcf_stem_finder():
    df = _build_synthetic_splash_df()
    out = process_df(df.copy())
    assert not out.empty
    # Anchor 1 base has a WCF stem at (0, 4, 22, 26); both finders detect it.
    a1 = out[out.base_target.str.startswith("AGCAA")].iloc[0]
    assert a1.stem_start_idx == 0
    assert a1.stem_end_idx == 4


def test_process_df_wobble_finds_wobble_stem():
    df = _build_synthetic_splash_df()
    out = process_df(df.copy(), wobble=True)
    assert not out.empty
    # Both anchors should have stem detected; anchor 2 has 1 wobble pair.
    grouped = out.groupby("anchor").first()
    assert (grouped.stemL > 0).all()


# ---------------------------------------------------------------------
# Outcome enumerators
# ---------------------------------------------------------------------

def test_target_p_outcome_ext_returns_sorted_unique_values():
    b = [("A", "T"), ("G", "C"), ("G", "T"), ("A", "T"), ("C", "G")]
    outcomes = target_p_outcome_ext(k=27, stemL=5, totaMut=3, b=b)
    assert len(outcomes) >= 1
    assert outcomes == sorted(outcomes)
    assert len(outcomes) == len(set(outcomes))
    assert all(0.0 <= p <= 1.0 for p in outcomes)


def test_target_p_outcome_ext_sve_outcome_present_when_stemMut_zero_possible():
    """When totaMut <= k - 2L, stemMut = 0 is possible so the SVE p-value
    must appear in the outcome set."""
    from math import comb
    k, stemL, totaMut = 27, 5, 3
    b = [("A", "T")] * stemL
    sve = comb(k - 2 * stemL, totaMut) / comb(k, totaMut)
    outcomes = target_p_outcome_ext(k, stemL, totaMut, b)
    assert any(abs(o - sve) < 1e-12 for o in outcomes)


def test_prep_for_conv_ext_caps_at_four_targets():
    """prep_for_conv_ext mirrors prep_for_conv: caps targets at 4."""
    b1 = [("A", "T"), ("G", "C")]
    wgt = [0.4, 0.3, 0.2, 0.1, 0.05]
    stemL_list = [2, 2, 2, 2, 2]
    totaMut_list = [1, 2, 1, 2, 1]
    b_list = [b1, b1, b1, b1, b1]
    outcomes, pmf = prep_for_conv_ext(5, wgt, k=10, stemL_list=stemL_list,
                                      totaMut_list=totaMut_list, b_list=b_list)
    assert len(outcomes) == 4
    assert len(pmf) == 4


def test_target_p_outcome_ext_b_vector_changes_outcome_set():
    """Stem composition matters: WCF-only and wobble-only stems give
    different outcome supports."""
    k, stemL, totaMut = 12, 3, 2
    b_wcf = [("A", "T"), ("G", "C"), ("T", "A")]
    b_wob = [("G", "T"), ("T", "G"), ("G", "T")]
    out_wcf = target_p_outcome_ext(k, stemL, totaMut, b_wcf)
    out_wob = target_p_outcome_ext(k, stemL, totaMut, b_wob)
    # Different compositions ⇒ generally different outcome sets. Use SVE
    # cross-check below to be precise; here just assert non-equality.
    assert out_wcf != out_wob


# ---------------------------------------------------------------------
# End-to-end smoke test of the pipeline (target mode, wobble=True)
# ---------------------------------------------------------------------

@pytest.fixture
def synthetic_splash_tsv(tmp_path):
    df = _build_synthetic_splash_df()
    p = tmp_path / "synthetic_splash.tsv"
    df.to_csv(p, sep="\t", index=False)
    return p


def test_SS_target_runs_with_wobble_flag(synthetic_splash_tsv, tmp_path, monkeypatch):
    """Smoke test: SS_target completes without error under --wobble and
    writes an output TSV with the expected columns."""
    from splash_structure_py.structure_target_mode import SS_target

    # Run inside tmp_path so output files don't pollute the repo.
    monkeypatch.chdir(tmp_path)

    SS_target(
        output_prefix="wobble_test",
        splash_output_file=str(synthetic_splash_tsv),
        element_annotation=False,
        wobble=True,
    )

    out_file = tmp_path / "wobble_test_results" / "structure_on_targets.tsv"
    assert out_file.exists()
    out = pd.read_csv(out_file, sep="\t")
    # Wobble path should produce the new columns.
    for col in ["totaMut", "stemMut", "E", "b_vector", "anchor_p", "anchor_p_BH"]:
        assert col in out.columns, f"missing column {col}"


def test_SS_target_runs_without_wobble_flag(synthetic_splash_tsv, tmp_path, monkeypatch):
    """Backward compat: existing pipeline path still works when --wobble
    is omitted."""
    from splash_structure_py.structure_target_mode import SS_target

    monkeypatch.chdir(tmp_path)

    SS_target(
        output_prefix="wcf_test",
        splash_output_file=str(synthetic_splash_tsv),
        element_annotation=False,
        wobble=False,
    )

    out_file = tmp_path / "wcf_test_results" / "structure_on_targets.tsv"
    assert out_file.exists()
    out = pd.read_csv(out_file, sep="\t")
    # Original columns must still be present and "E" must not.
    for col in ["totaMut", "stemMut", "compMut", "anchor_p", "anchor_p_BH"]:
        assert col in out.columns, f"missing column {col}"
    assert "E" not in out.columns, "wobble-only column leaked into WCF path"
