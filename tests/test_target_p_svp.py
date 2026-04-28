"""Brute-force vs DP cross-checks for target_p_svp.

For small (k, v, L) the SVP DP from nonWCF_derivation.tex Section 5.4
must equal a direct enumeration of all Hamming-distance-v targets.

Also verifies recovery of the original BPC closed form when restricted
to WCF pairs and counting only n_p = 2 events.
"""

from itertools import combinations, product

import pytest

from splash_structure_py.src.get_pval import (
    target_p1_closed_form,
    target_p_ext,
    target_p_svp,
)
from splash_structure_py.src.non_wcf import BASES, V_EXT, V_WCF


# ---------------------------------------------------------------------
# Brute-force reference implementations
# ---------------------------------------------------------------------

def _build_base_target(k, b):
    """Build a length-k base target T_0 with stem pairs at positions
    (2p, 2p+1) for p = 0..L-1; non-stem positions filled with 'A'."""
    T0 = ["A"] * k
    L = len(b)
    assert 2 * L <= k, "stem must fit"
    for p in range(L):
        T0[2 * p] = b[p][0]
        T0[2 * p + 1] = b[p][1]
    return T0


def _enumerate_distance_v_targets(T0, v):
    """Yield every (positions_tuple, new_bases_tuple) for varied targets at
    Hamming distance exactly v."""
    k = len(T0)
    for positions in combinations(range(k), v):
        # at each position the mutation must differ from T0[pos]
        alts_per_pos = [
            [b for b in BASES if b != T0[pos]] for pos in positions
        ]
        for new_bases in product(*alts_per_pos):
            yield positions, new_bases


def _compute_E(T0, b, positions, new_bases, valid_pairs):
    """Compute the structure-supporting count E for one varied target.

    ``valid_pairs`` is the set used to classify a stem pair as supporting
    (E_p = 1). Pass V_EXT for SVP, V_WCF for BPC-only restriction.
    """
    L = len(b)
    # apply mutations
    Ti = list(T0)
    for pos, nb in zip(positions, new_bases):
        Ti[pos] = nb
    E = 0
    for p in range(L):
        bl, br = Ti[2 * p], Ti[2 * p + 1]
        if bl == b[p][0] and br == b[p][1]:
            continue  # no mutation in pair, E_p = 0
        if (bl, br) in valid_pairs:
            E += 1
    return E


def brute_force_p_svp(k, v, L, e, b):
    """Brute-force Pr(E >= e | k, v, L, b) for SVP under V_EXT."""
    T0 = _build_base_target(k, b)
    total = 0
    hits = 0
    for positions, new_bases in _enumerate_distance_v_targets(T0, v):
        total += 1
        if _compute_E(T0, b, positions, new_bases, V_EXT) >= e:
            hits += 1
    return hits / total if total > 0 else 0.0


def brute_force_p_bpc_wcf(k, v, L, c, b):
    """Brute-force Pr(C >= c | k, v, L, b) where C counts ONLY n_p = 2
    events whose new pair lies in V_WCF. Used to cross-check the original
    target_p1_closed_form."""
    T0 = _build_base_target(k, b)
    total = 0
    hits = 0
    for positions, new_bases in _enumerate_distance_v_targets(T0, v):
        total += 1
        Ti = list(T0)
        for pos, nb in zip(positions, new_bases):
            Ti[pos] = nb
        C = 0
        for p in range(L):
            bl, br = Ti[2 * p], Ti[2 * p + 1]
            l_mut = bl != b[p][0]
            r_mut = br != b[p][1]
            if l_mut and r_mut and (bl, br) in V_WCF:
                C += 1
        if C >= c:
            hits += 1
    return hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------
# DP vs brute force — primary correctness test
# ---------------------------------------------------------------------

# Configurations chosen to exercise WCF-only, wobble-only, and mixed stems
# at various (k, v, e). All small enough for brute-force enumeration.
SVP_CASES = [
    # all-WCF stem
    (8, 2, [("A", "T"), ("G", "C")], 1),
    (8, 2, [("A", "T"), ("G", "C")], 2),
    (8, 3, [("A", "T"), ("G", "C")], 1),
    (8, 3, [("A", "T"), ("G", "C")], 2),
    # all-wobble stem
    (8, 2, [("G", "T"), ("T", "G")], 1),
    (8, 2, [("G", "T"), ("T", "G")], 2),
    (8, 3, [("G", "T"), ("T", "G")], 1),
    # mixed stem
    (10, 3, [("A", "T"), ("G", "T"), ("C", "G")], 1),
    (10, 3, [("A", "T"), ("G", "T"), ("C", "G")], 2),
    (10, 4, [("A", "T"), ("G", "T"), ("C", "G")], 1),
    (10, 4, [("A", "T"), ("G", "T"), ("C", "G")], 2),
    # stem fills the whole target — no non-stem positions
    (4, 2, [("A", "T"), ("G", "C")], 1),
    (4, 2, [("A", "T"), ("G", "C")], 2),
    # larger L
    (12, 3, [("A", "T"), ("G", "C"), ("T", "A"), ("G", "T")], 1),
    (12, 3, [("A", "T"), ("G", "C"), ("T", "A"), ("G", "T")], 2),
    (12, 4, [("A", "T"), ("G", "C"), ("T", "A"), ("G", "T")], 2),
]


@pytest.mark.parametrize("k, v, b, e", SVP_CASES)
def test_target_p_svp_matches_brute_force(k, v, b, e):
    L = len(b)
    dp = target_p_svp(k, v, L, e, b)
    bf = brute_force_p_svp(k, v, L, e, b)
    assert dp == pytest.approx(bf, abs=1e-12)


# ---------------------------------------------------------------------
# Recovery of original BPC closed form (sanity check on the existing code)
# ---------------------------------------------------------------------

BPC_RECOVERY_CASES = [
    # (k, v, b, c) — all-WCF stems so brute_force_p_bpc_wcf matches
    # target_p1_closed_form (which assumes V_WCF and counts only n_p = 2 events).
    (8, 2, [("A", "T"), ("G", "C")], 1),
    (8, 3, [("A", "T"), ("G", "C")], 1),
    (8, 4, [("A", "T"), ("G", "C")], 1),
    (8, 4, [("A", "T"), ("G", "C")], 2),
    (10, 3, [("A", "T"), ("G", "C"), ("T", "A")], 1),
    (10, 4, [("A", "T"), ("G", "C"), ("T", "A")], 1),
    (10, 4, [("A", "T"), ("G", "C"), ("T", "A")], 2),
    (12, 4, [("A", "T"), ("G", "C"), ("T", "A"), ("C", "G")], 1),
    (12, 4, [("A", "T"), ("G", "C"), ("T", "A"), ("C", "G")], 2),
]


@pytest.mark.parametrize("k, v, b, c", BPC_RECOVERY_CASES)
def test_target_p1_closed_form_matches_brute_force_bpc(k, v, b, c):
    """Independent cross-check: existing target_p1_closed_form must match
    a brute-force enumeration restricted to V_WCF + n_p = 2 events."""
    L = len(b)
    cf = target_p1_closed_form(k, v, L, c)
    bf = brute_force_p_bpc_wcf(k, v, L, c, b)
    assert cf == pytest.approx(bf, abs=1e-12)


# ---------------------------------------------------------------------
# Sanity properties: ranges, monotonicity, edge cases
# ---------------------------------------------------------------------

@pytest.mark.parametrize("k, v, b, e", SVP_CASES)
def test_target_p_svp_in_unit_interval(k, v, b, e):
    p = target_p_svp(k, v, len(b), e, b)
    assert 0.0 <= p <= 1.0


def test_target_p_svp_monotone_in_e():
    """Pr(E >= e) must be non-increasing in e."""
    k, v = 12, 4
    b = [("A", "T"), ("G", "C"), ("G", "T"), ("T", "A")]
    L = len(b)
    ps = [target_p_svp(k, v, L, e, b) for e in range(L + 2)]
    for i in range(len(ps) - 1):
        assert ps[i] >= ps[i + 1] - 1e-12


def test_target_p_svp_e_zero_returns_one():
    b = [("A", "T"), ("G", "T")]
    assert target_p_svp(8, 2, 2, 0, b) == 1.0


def test_target_p_svp_e_above_L_returns_zero():
    b = [("A", "T"), ("G", "T")]
    assert target_p_svp(8, 2, 2, 3, b) == 0.0


def test_target_p_svp_v_zero_returns_one_for_e_zero():
    """No mutations means E = 0; Pr(E >= 0) = 1."""
    b = [("A", "T"), ("G", "T")]
    assert target_p_svp(8, 0, 2, 0, b) == 1.0


def test_target_p_svp_v_zero_returns_zero_for_e_one():
    """No mutations means E = 0; Pr(E >= 1) = 0."""
    b = [("A", "T"), ("G", "T")]
    assert target_p_svp(8, 0, 2, 1, b) == 0.0


def test_target_p_svp_b_length_mismatch_raises():
    with pytest.raises(ValueError):
        target_p_svp(8, 2, 3, 1, [("A", "T"), ("G", "C")])  # L=3 but len(b)=2


# ---------------------------------------------------------------------
# target_p_ext — indicator-combined SVE / SVP
# ---------------------------------------------------------------------

def test_target_p_ext_uses_SVE_when_stemMut_zero():
    """When stemMut == 0, target_p_ext returns the SVE hypergeometric."""
    from math import comb
    k, stemL, totaMut = 8, 2, 2
    b = [("A", "T"), ("G", "C")]
    expected = comb(k - 2 * stemL, totaMut) / comb(k, totaMut)
    assert target_p_ext(k, stemL, totaMut, 0, 0, b) == pytest.approx(expected)


def test_target_p_ext_uses_SVP_when_stemMut_positive():
    """When stemMut > 0, target_p_ext delegates to target_p_svp."""
    k, stemL, totaMut, stemMut, e = 10, 3, 4, 2, 1
    b = [("A", "T"), ("G", "T"), ("C", "G")]
    expected = target_p_svp(k, totaMut, stemL, e, b)
    assert target_p_ext(k, stemL, totaMut, stemMut, e, b) == pytest.approx(expected)
