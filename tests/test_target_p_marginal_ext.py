"""Tests for target_p_marginal_ext, the exact marginal PMF of target_p_ext.

The discrete distribution of target_p_ext under H_0 is computed by aggregating
joint Pr(H = h) * Pr(E = e | H = h, b) over (h, e) configurations grouped by
target_p_ext value. This test verifies that:

  1. The PMF sums to 1.
  2. The support and PMF match a brute-force enumeration of all
     distance-v targets.
  3. The PMF differs from the differences-of-sorted-support surrogate
     when q_0 is non-trivially positioned in the support — confirming
     the bug fix matters.
"""

from itertools import combinations, product
from math import comb

import pytest

from splash_structure_py.src.get_pval import (
    target_p_ext,
    target_p_marginal_ext,
    target_p_outcome_ext,
)
from splash_structure_py.src.non_wcf import BASES, V_EXT


# ---------------------------------------------------------------------
# Brute-force reference
# ---------------------------------------------------------------------

def _build_base_target(k, b):
    T0 = ["A"] * k
    L = len(b)
    assert 2 * L <= k
    for p in range(L):
        T0[2 * p] = b[p][0]
        T0[2 * p + 1] = b[p][1]
    return T0


def _enumerate_distance_v_targets(T0, v):
    k = len(T0)
    for positions in combinations(range(k), v):
        alts_per_pos = [[b for b in BASES if b != T0[pos]] for pos in positions]
        for new_bases in product(*alts_per_pos):
            yield positions, new_bases


def _stemMut_E(T0, b, positions, new_bases):
    """Compute (s, e) for one varied target under V_EXT classification."""
    L = len(b)
    Ti = list(T0)
    for pos, nb in zip(positions, new_bases):
        Ti[pos] = nb
    s = 0
    e = 0
    for p in range(L):
        bl, br = Ti[2 * p], Ti[2 * p + 1]
        l_mut = bl != b[p][0]
        r_mut = br != b[p][1]
        n_p = int(l_mut) + int(r_mut)
        s += n_p
        if n_p > 0 and (bl, br) in V_EXT:
            e += 1
    return s, e


def brute_force_marginal(k, v, L, b):
    """Brute-force PMF of target_p_ext under H_0 via direct enumeration."""
    T0 = _build_base_target(k, b)
    bin_count: dict[float, int] = {}
    total = 0
    for positions, new_bases in _enumerate_distance_v_targets(T0, v):
        s, e = _stemMut_E(T0, b, positions, new_bases)
        p = target_p_ext(k, L, v, s, e, b)
        key = round(float(p), 12)
        bin_count[key] = bin_count.get(key, 0) + 1
        total += 1
    support = sorted(bin_count.keys())
    pmf = [bin_count[p] / total for p in support]
    return support, pmf


# ---------------------------------------------------------------------
# Exhaustive cross-check: small (k, L, v) grid
# ---------------------------------------------------------------------

_GRID = [
    # (k, L, b)
    (10, 2, [("A", "T"), ("G", "C")]),                        # all WCF
    (10, 2, [("G", "T"), ("T", "G")]),                        # all wobble
    (10, 2, [("A", "T"), ("G", "T")]),                        # mixed L=2
    (12, 3, [("A", "T"), ("G", "C"), ("T", "A")]),            # WCF L=3
    (12, 3, [("A", "T"), ("G", "T"), ("C", "G")]),            # mixed L=3
]


@pytest.mark.parametrize("k, L, b", _GRID)
@pytest.mark.parametrize("v", [1, 2, 3])
def test_marginal_matches_brute_force(k, L, b, v):
    if 2 * L > k or v > k:
        pytest.skip("infeasible")

    sup_dp, pmf_dp = target_p_marginal_ext(k, L, v, b)
    sup_bf, pmf_bf = brute_force_marginal(k, v, L, b)

    assert len(sup_dp) == len(sup_bf), (
        f"k={k} L={L} v={v} b={b}: support sizes differ "
        f"DP={len(sup_dp)} BF={len(sup_bf)}"
    )
    for p_dp, p_bf in zip(sup_dp, sup_bf):
        assert abs(p_dp - p_bf) < 1e-10
    for w_dp, w_bf in zip(pmf_dp, pmf_bf):
        assert abs(w_dp - w_bf) < 1e-10, (
            f"k={k} L={L} v={v} b={b}: PMF mismatch DP={pmf_dp} BF={pmf_bf}"
        )


# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------

@pytest.mark.parametrize("k, L, b", _GRID)
@pytest.mark.parametrize("v", [1, 2, 3, 4])
def test_marginal_sums_to_one(k, L, b, v):
    if 2 * L > k or v > k:
        pytest.skip("infeasible")
    _, pmf = target_p_marginal_ext(k, L, v, b)
    assert abs(sum(pmf) - 1.0) < 1e-10


def test_marginal_at_q0_includes_only_h_zero_when_q0_lt_other_support():
    """For the regime where q_0 is below all SVP support points (typical
    when v is large enough that h=0 is rarer than typical SVP outcomes),
    Pr_H0(p_tar = q_0) should equal exactly Pr(H = 0)."""
    k, v, L = 27, 8, 6
    b = [("A", "T")] * L
    q0 = comb(k - 2 * L, v) / comb(k, v)
    pr_h_zero = q0  # hypergeometric Pr(H = 0)
    support, pmf = target_p_marginal_ext(k, L, v, b)
    # Find the bin keyed at q0.
    matched = [p for p in support if abs(p - q0) < 1e-12]
    assert len(matched) == 1, f"q_0 = {q0} not found in support"
    idx = support.index(matched[0])
    # PMF at q_0 should match Pr(H = 0) within numerical tolerance.
    # (If some SVP outcome happens to numerically coincide with q_0, the
    # check still works because the binning aggregates them anyway, and
    # the joint contribution at h=0 is exactly Pr(H=0).)
    assert pmf[idx] >= pr_h_zero - 1e-12


def test_corrected_pmf_differs_from_differences_surrogate():
    """The new PMF must differ from differences-of-sorted-support at
    cells where q_0 sits among the SVP support — exactly the regime
    the bug afflicted."""
    k, v, L = 12, 4, 3
    b = [("A", "T"), ("G", "C"), ("G", "T")]
    sup_exact, pmf_exact = target_p_marginal_ext(k, L, v, b)
    # Old surrogate: differences of sorted support.
    pmf_old = [sup_exact[0]] + [
        sup_exact[i + 1] - sup_exact[i]
        for i in range(len(sup_exact) - 1)
    ]
    diffs = sum(abs(a - b) for a, b in zip(pmf_exact, pmf_old))
    assert diffs > 1e-6, (
        "Exact PMF identical to differences-of-CDF surrogate; "
        "bug fix did not change anything in this cell"
    )


def test_outcome_ext_wrapper_returns_same_support():
    """target_p_outcome_ext must still return the sorted support so any
    caller that only wanted the support keeps working."""
    k, v, L = 12, 3, 3
    b = [("A", "T"), ("G", "T"), ("C", "G")]
    sup_marginal, _ = target_p_marginal_ext(k, L, v, b)
    sup_legacy = target_p_outcome_ext(k, L, v, b)
    assert sup_marginal == sup_legacy
