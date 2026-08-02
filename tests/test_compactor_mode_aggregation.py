"""Phase C4 positive control: hand-crafted compactor row, verify the
per-half decomposition and aggregation reproduce the (k, L, v, s, E, b)
tuple the math requires, and that target_p_ext on the aggregated
parameters falls out cleanly.

Design (cDNA alphabet; the compactor is the virtual concatenation
S = base_S1 + base_S2 of length 54):

base_S1 (27 nt) = "AGCAG" + "A" * 17 + "CTGTT"
  Stem: positions 0-4 (left), 22-26 (right), length 5.
  Pairs (b_L^p, b_R^p) for p = 0..4:
      0: (A, T)  WCF
      1: (G, T)  wobble  ← this is the only wobble in the test
      2: (C, G)  WCF
      3: (A, T)  WCF
      4: (G, C)  WCF

S1 (27 nt) = base_S1 with a single T→C mutation at position 25
  At pair 1 this turns (G, T) wobble into (G, C) WCF: a single-position
  compatible (SPC) event at pair 1, right-side mutation. E_1 = 1.

base_S2 (27 nt) = "AGCTA" + "A" * 17 + "TAGCT"
  Stem: positions 0-4 (left), 22-26 (right), length 5.
  Pairs:
      0: (A, T)
      1: (G, C)
      2: (C, G)
      3: (T, A)
      4: (A, T)
  All WCF.

S2 (27 nt) = base_S2 with mutations at positions 0 and 26
  Pair 0 mutates both positions: (A, T) → (G, C), both transitions, the
  classic base-pair-covariation (BPC) event. E_2 = 1.

Aggregated parameters (the input target_p_ext should see):
  k       = 27 + 27 = 54
  L       = 5 + 5 = 10
  v       = 1 + 2 = 3
  stemMut = 1 + 2 = 3      (> 0, so the indicator switch goes SVP)
  E       = 1 + 1 = 2
  b       = b_1 ⊕ b_2      (concatenation; the DP is order-invariant
                            in b so the convention is arbitrary)
"""

import math

import pytest

from splash_structure_py.src.find_comp_mut import find_mutation_ext
from splash_structure_py.src.get_pval import target_p_ext, target_p_svp
from splash_structure_py.src.non_wcf import V_EXT, V_WCF
from splash_structure_py.src.process_targets import find_stem_ind_wobble


# ---------------------------------------------------------------------
# Half-1 design: wobble at pair 1, SPC event by right-side mutation
# ---------------------------------------------------------------------

BASE_S1 = "AGCAG" + "A" * 17 + "CTGTT"
S1 = "AGCAG" + "A" * 17 + "CTGCT"  # T → C at position 25

EXPECTED_STEM_1 = (0, 4, 22, 26, 5)
EXPECTED_B_1 = [("A", "T"), ("G", "T"), ("C", "G"), ("A", "T"), ("G", "C")]
EXPECTED_NP_1 = [0, 1, 0, 0, 0]


# ---------------------------------------------------------------------
# Half-2 design: all-WCF stem, BPC event at pair 0
# ---------------------------------------------------------------------

BASE_S2 = "AGCTA" + "A" * 17 + "TAGCT"
S2 = "GGCTA" + "A" * 17 + "TAGCC"  # A → G at 0, T → C at 26

EXPECTED_STEM_2 = (0, 4, 22, 26, 5)
EXPECTED_B_2 = [("A", "T"), ("G", "C"), ("C", "G"), ("T", "A"), ("A", "T")]
EXPECTED_NP_2 = [2, 0, 0, 0, 0]


# ---------------------------------------------------------------------
# Per-half: stem detection + mutation classification
# ---------------------------------------------------------------------

def test_half_1_stem_detection():
    assert find_stem_ind_wobble(BASE_S1, 5) == EXPECTED_STEM_1


def test_half_2_stem_detection():
    assert find_stem_ind_wobble(BASE_S2, 5) == EXPECTED_STEM_2


def test_half_1_mutation_classification():
    """SPC event: half-1 has 1 mutation, falling on the right side of
    the wobble pair, converting (G, T) → (G, C) which is in V_EXT."""
    totaMut, stemMut, E, _struc, n_p, b = find_mutation_ext(
        BASE_S1, S1, *EXPECTED_STEM_1[:4]
    )
    assert totaMut == 1
    assert stemMut == 1
    assert E == 1
    assert n_p == EXPECTED_NP_1
    assert b == EXPECTED_B_1


def test_half_2_mutation_classification():
    """BPC event: pair 0 (A, T) → (G, C), both positions mutated, both
    in V_EXT."""
    totaMut, stemMut, E, _struc, n_p, b = find_mutation_ext(
        BASE_S2, S2, *EXPECTED_STEM_2[:4]
    )
    assert totaMut == 2
    assert stemMut == 2
    assert E == 1
    assert n_p == EXPECTED_NP_2
    assert b == EXPECTED_B_2


# ---------------------------------------------------------------------
# Aggregation: (k, L, v, s, E, b) match the math
# ---------------------------------------------------------------------

def _aggregate():
    """Run the per-half pipeline and aggregate exactly as
    structure_compactor_mode.py does, returning the
    (k, L, v, stemMut, E, b) tuple consumed by target_p_ext."""
    _, _, E_1, _, n_p_1, b_1 = find_mutation_ext(BASE_S1, S1, *EXPECTED_STEM_1[:4])
    _, _, E_2, _, n_p_2, b_2 = find_mutation_ext(BASE_S2, S2, *EXPECTED_STEM_2[:4])

    k = len(BASE_S1) + len(BASE_S2)
    L = EXPECTED_STEM_1[4] + EXPECTED_STEM_2[4]
    v = 1 + 2  # totaMut_1 + totaMut_2
    stemMut = 1 + 2
    E = E_1 + E_2
    b = list(b_1) + list(b_2)
    return k, L, v, stemMut, E, b


def test_aggregation_scalars():
    k, L, v, stemMut, E, _b = _aggregate()
    assert k == 54
    assert L == 10
    assert v == 3
    assert stemMut == 3
    assert E == 2


def test_aggregation_b_vector_is_concatenation():
    """b_vector under the pipeline is b_1 ⊕ b_2. The DP is order-
    invariant in b, but this test pins the implementation choice so
    downstream consumers can rely on it."""
    _, _, _, _, _, b = _aggregate()
    assert b == EXPECTED_B_1 + EXPECTED_B_2
    assert len(b) == 10


# ---------------------------------------------------------------------
# target_p_ext on the aggregated tuple: a valid probability, indicator
# switch picks SVP, recovers SVE for the s = 0 case
# ---------------------------------------------------------------------

def test_target_p_ext_on_aggregated_tuple_is_valid_probability():
    k, L, v, stemMut, E, b = _aggregate()
    p = target_p_ext(k, L, v, stemMut, E, b, titv=0.5, valid=V_EXT)
    assert 0.0 <= p <= 1.0
    # stemMut > 0 so indicator switch goes SVP; the value must equal
    # target_p_svp on the same arguments.
    p_svp = target_p_svp(k, v, L, E, b, titv=0.5, valid=V_EXT)
    assert p == pytest.approx(p_svp, rel=1e-12)


def test_target_p_ext_monotone_in_e():
    """Pr(E ≥ e | …) is non-increasing in e by construction of the tail
    sum. On the aggregated stem this must hold across e = 0..L."""
    k, L, v, stemMut, _E, b = _aggregate()
    # Force the SVP branch with stemMut > 0; vary e on top.
    p_prev = 1.0
    for e in range(L + 1):
        p = target_p_ext(k, L, v, stemMut, e, b, titv=0.5, valid=V_EXT)
        assert p <= p_prev + 1e-12
        p_prev = p


def test_sve_branch_aggregates_to_hypergeometric():
    """When stemMut = 0, the indicator switch picks SVE, which is the
    hypergeometric C(k - 2L, v) / C(k, v). The aggregation must
    preserve this exactly — k = 54, L = 10, v = 3 in our design."""
    k, L, v, _stemMut, _E, b = _aggregate()
    # Force the SVE branch by passing stemMut = 0 (e becomes irrelevant
    # in target_p_ext's SVE path).
    p_sve = target_p_ext(k, L, v, stemMut=0, e=0, b=b, titv=0.5, valid=V_EXT)
    expected = math.comb(k - 2 * L, v) / math.comb(k, v)
    assert p_sve == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------
# Sanity: with valid = V_WCF (no wobble admitted), half-1 loses its
# wobble-using stem and the stem-finder either rejects it or finds an
# all-WCF alternative. Pin whichever it is so the downstream behaviour
# is documented.
# ---------------------------------------------------------------------

def test_half_1_under_wcf_only_loses_wobble_stem():
    """With valid restricted to V_WCF, pair 1 of base_S1's stem
    (G, T) is no longer a valid pair, so the wobble-using length-5 stem
    is rejected. There is no all-WCF length ≥ 5 alternative in
    base_S1, so the stem-finder must return the no-stem sentinel."""
    result = find_stem_ind_wobble(BASE_S1, 5, valid=V_WCF)
    assert result == [0, 0, 0, 0, 0]
