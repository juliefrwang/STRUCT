"""Phase 7 regression tests for the non-default `noncanon` paths
(slices 7.3 stem-finder, 7.4 find_mutation_ext, 7.6 SS_target).

The existing suites only exercise the default G·U path (they confirm
backward compat). These lock in correctness of the *generalized* V(N)
behaviour so a future refactor can't silently break G·A / multi-pair
support.
"""

import os
import tempfile

import pandas as pd
import pytest
from pandarallel import pandarallel

from splash_structure_py.src.non_wcf import build_valid_set, V_EXT
from splash_structure_py.src.process_targets import find_stem_ind_wobble
from splash_structure_py.src.find_comp_mut import find_mutation_ext

pandarallel.initialize(nb_workers=1, progress_bar=False, verbose=0)

V_GU = build_valid_set("GU")
V_GA = build_valid_set("GA")
V_GUGA = build_valid_set("GU,GA")


# ---------------------------------------------------------------------
# 7.3 — stem-finder under V(N)
# ---------------------------------------------------------------------

# 5-pair stem at 0-4 / 8-12 with ONE G·A pair (pair 1), rest WCF;
# C-rich loop prevents accidental alternative stems.
#   pairs: (A,T) (G,A) (C,G) (A,T) (A,T)   -> n_nc = 1
GA_STEM_TARGET = "AGCAA" + "CCC" + "TTGAT"


def test_GA_stem_detected_only_when_N_admits_GA():
    assert len(GA_STEM_TARGET) == 13
    # G·A not in V(GU) -> the only candidate stem is invalid -> no stem.
    assert list(find_stem_ind_wobble(GA_STEM_TARGET, 5, valid=V_GU)) == [0, 0, 0, 0, 0]
    # G·A admitted -> the 5-pair stem is found.
    assert list(find_stem_ind_wobble(GA_STEM_TARGET, 5, valid=V_GA)) == [0, 4, 8, 12, 5]
    assert list(find_stem_ind_wobble(GA_STEM_TARGET, 5, valid=V_GUGA)) == [0, 4, 8, 12, 5]


def test_default_valid_is_VEXT_no_GA():
    """Default (no valid arg) == V_EXT == V(GU): must NOT find the G·A stem."""
    assert list(find_stem_ind_wobble(GA_STEM_TARGET, 5)) == [0, 0, 0, 0, 0]


# Aggregate cap pools ALL non-canonical types. 5-pair stem with 3 G·A
# pairs -> n_nc = 3 > floor(5/2) = 2 -> rejected even though G·A ∈ V(N).
#   pairs: (G,A) (G,A) (G,A) (A,T) (A,T)
GA_HEAVY_TARGET = "GGGAA" + "CCC" + "TTAAA"


def test_aggregate_cap_rejects_GA_heavy_stem():
    assert len(GA_HEAVY_TARGET) == 13
    # 3 non-canonical pairs exceed floor(L/2)=2 -> no stem under V(GU,GA).
    assert list(find_stem_ind_wobble(GA_HEAVY_TARGET, 5, valid=V_GUGA)) == [0, 0, 0, 0, 0]
    assert list(find_stem_ind_wobble(GA_HEAVY_TARGET, 5, valid=V_GA)) == [0, 0, 0, 0, 0]


def test_aggregate_cap_pools_mixed_types():
    """1 G·U + 1 G·A = n_nc 2 (= cap, accepted); the cap counts all
    non-canonical types together, not per-type.

      pairs: (G,T) (G,A) (C,G) (A,T) (A,T)  -> n_nc = 2
    """
    target = "AGCAA" + "CCC" + "TTGAT"  # reuse layout; set pair0 -> (G,T)
    # left "GGCAA": pair0 (G, target[12]) etc. Build explicitly:
    #   left = G G C A A   (idx 0-4)
    #   want (G,T)(G,A)(C,G)(A,T)(A,T): target[12]=T,11=A,10=G,9=T,8=T
    target = "GGCAA" + "CCC" + "TTGAT"
    assert len(target) == 13
    res = find_stem_ind_wobble(target, 5, valid=V_GUGA)
    assert list(res) == [0, 4, 8, 12, 5]   # n_nc=2 == cap -> accepted
    # Under V(GU) the (G,A) pair is invalid -> no stem.
    assert list(find_stem_ind_wobble(target, 5, valid=V_GU)) == [0, 0, 0, 0, 0]


# ---------------------------------------------------------------------
# 7.4 — find_mutation_ext classification under V(N)
# ---------------------------------------------------------------------

# BASE/STEM reused from test_find_mutation_ext.py conventions.
#   BASE pairs (stem 0-2 / 7-9): (A,T) (G,C) (G,T)
_FM_BASE = "AGGAAAATCTAAA"
_FM_STEM = (0, 2, 7, 9)


def test_find_mutation_ext_GA_is_disrupting_under_GU_but_SPC_under_GUGA():
    # Mutate index 8 (pair-1 right side) C -> A: pair 1 becomes (G,A).
    target = _FM_BASE[:8] + "A" + _FM_BASE[9:]
    assert target != _FM_BASE and len(target) == len(_FM_BASE)

    tot_gu, stem_gu, E_gu, *_ = find_mutation_ext(_FM_BASE, target, *_FM_STEM, V_GU)
    tot_ga, stem_ga, E_ga, *_ = find_mutation_ext(_FM_BASE, target, *_FM_STEM, V_GUGA)

    # Same mutation count either way; only the structure-supporting
    # classification of the (G,A) pair differs.
    assert tot_gu == tot_ga
    assert stem_gu == stem_ga
    assert E_gu == 0      # (G,A) not in V(GU) -> disrupting
    assert E_ga == 1      # (G,A) in V(GU,GA) -> SPC, counts toward E


def test_find_mutation_ext_default_matches_explicit_GU():
    target = _FM_BASE[:8] + "A" + _FM_BASE[9:]
    default = find_mutation_ext(_FM_BASE, target, *_FM_STEM)
    explicit = find_mutation_ext(_FM_BASE, target, *_FM_STEM, V_EXT)
    assert default == explicit


# ---------------------------------------------------------------------
# 7.6 — SS_target noncanon gate/threading + backward-compat
# ---------------------------------------------------------------------

def _synthetic_splash_df():
    a1 = "AGCAA" + ("A" * 17) + "TTGCT"          # clean 5-bp WCF stem
    a2 = "AGCAA" + ("C" * 17) + "TTGTT"          # 5-bp stem w/ one G·U
    def row(anchor, base):
        return {
            "anchor": anchor, "M": 1000,
            "most_freq_target_1": base, "cnt_most_freq_target_1": 500,
            "most_freq_target_2": base[:24] + "TTT", "cnt_most_freq_target_2": 250,
            "most_freq_target_3": base[:23] + "TTTT", "cnt_most_freq_target_3": 150,
            "most_freq_target_4": base[:22] + "TTTTT", "cnt_most_freq_target_4": 100,
        }
    return pd.DataFrame([
        row("ANCHOR1_" + "A" * 23, a1),
        row("ANCHOR2_" + "C" * 23, a2),
    ])


def _run(tmp, **kw):
    from splash_structure_py.structure_target_mode import SS_target
    inp = os.path.join(tmp, "in.tsv")
    _synthetic_splash_df().to_csv(inp, sep="\t", index=False)
    prefix = os.path.join(tmp, kw.pop("_tag"))
    SS_target(prefix, inp, False, **kw)
    out = f"{prefix}_results/structure_on_targets.tsv"
    return pd.read_csv(out, sep="\t") if os.path.exists(out) else None


def test_noncanon_none_byte_identical_to_legacy():
    with tempfile.TemporaryDirectory() as tmp:
        legacy = _run(tmp, _tag="legacy", wobble=False)
        none = _run(tmp, _tag="none", noncanon="none")
    assert legacy is not None and none is not None
    pd.testing.assert_frame_equal(
        legacy.reset_index(drop=True), none.reset_index(drop=True)
    )


def test_noncanon_GU_matches_legacy_wobble():
    with tempfile.TemporaryDirectory() as tmp:
        wob = _run(tmp, _tag="wob", wobble=True)
        gu = _run(tmp, _tag="gu", noncanon="GU")
    assert wob is not None and gu is not None
    pd.testing.assert_frame_equal(
        wob.reset_index(drop=True), gu.reset_index(drop=True)
    )


def test_noncanon_GUGA_runs_extended_path():
    with tempfile.TemporaryDirectory() as tmp:
        ext = _run(tmp, _tag="guga", noncanon="GU,GA")
    assert ext is not None
    # Extended path emits the E column (legacy path emits compMut instead).
    assert "E" in ext.columns
    assert "b_vector" in ext.columns
