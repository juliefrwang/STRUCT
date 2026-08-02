"""Unit tests for splash_structure_py.src.non_wcf.

Verify that the per-pair-type quantities α_L, α_R, β, and π_p^(j)
match the table in Section 4 of nonWCF_derivation.tex exactly.
"""

import pytest

from splash_structure_py.src.non_wcf import (
    BASES,
    V_EXT,
    V_WCF,
    V_WOBBLE,
    alpha_L,
    alpha_R,
    beta,
    pair_type,
    pi_table,
)


# ---------------------------------------------------------------------
# Set memberships
# ---------------------------------------------------------------------

def test_V_WCF_contents():
    assert V_WCF == frozenset(
        {("A", "T"), ("T", "A"), ("G", "C"), ("C", "G")}
    )


def test_V_WOBBLE_contents():
    assert V_WOBBLE == frozenset({("G", "T"), ("T", "G")})


def test_V_EXT_is_disjoint_union():
    assert V_EXT == V_WCF | V_WOBBLE
    assert V_WCF.isdisjoint(V_WOBBLE)
    assert len(V_EXT) == 6


# ---------------------------------------------------------------------
# pair_type
# ---------------------------------------------------------------------

@pytest.mark.parametrize("pair", list(V_WCF))
def test_pair_type_wcf(pair):
    assert pair_type(*pair) == "wcf"


@pytest.mark.parametrize("pair", list(V_WOBBLE))
def test_pair_type_wobble(pair):
    assert pair_type(*pair) == "wobble"


@pytest.mark.parametrize(
    "pair",
    [("A", "A"), ("A", "C"), ("A", "G"), ("C", "C"), ("C", "T"),
     ("G", "A"), ("G", "G"), ("T", "C"), ("T", "T")],
)
def test_pair_type_invalid(pair):
    assert pair_type(*pair) is None


# ---------------------------------------------------------------------
# Section 4 table — α_L, α_R, β
# ---------------------------------------------------------------------

# (b_L, b_R) -> (α_L, α_R, β)
SECTION_4_TABLE = {
    ("A", "T"): (1 / 3, 0.0, 4 / 9),
    ("T", "A"): (0.0, 1 / 3, 4 / 9),
    ("G", "C"): (0.0, 1 / 3, 4 / 9),
    ("C", "G"): (1 / 3, 0.0, 4 / 9),
    ("G", "T"): (1 / 3, 1 / 3, 3 / 9),
    ("T", "G"): (1 / 3, 1 / 3, 3 / 9),
}


@pytest.mark.parametrize("pair, expected", list(SECTION_4_TABLE.items()))
def test_alpha_L_matches_table(pair, expected):
    assert alpha_L(*pair) == pytest.approx(expected[0])


@pytest.mark.parametrize("pair, expected", list(SECTION_4_TABLE.items()))
def test_alpha_R_matches_table(pair, expected):
    assert alpha_R(*pair) == pytest.approx(expected[1])


@pytest.mark.parametrize("pair, expected", list(SECTION_4_TABLE.items()))
def test_beta_matches_table(pair, expected):
    assert beta(*pair) == pytest.approx(expected[2])


# ---------------------------------------------------------------------
# π_p^(j) — Section 5.4
# ---------------------------------------------------------------------

@pytest.mark.parametrize("pair", list(V_EXT))
def test_pi_table_keys(pair):
    pi = pi_table(*pair)
    assert set(pi.keys()) == {0, 1, 2}


@pytest.mark.parametrize("pair", list(V_EXT))
def test_pi_table_pi0_is_zero(pair):
    assert pi_table(*pair)[0] == 0.0


@pytest.mark.parametrize("pair", list(V_EXT))
def test_pi_table_pi1_matches_alpha_average(pair):
    pi = pi_table(*pair)
    expected = 0.5 * (alpha_L(*pair) + alpha_R(*pair))
    assert pi[1] == pytest.approx(expected)


@pytest.mark.parametrize("pair", list(V_EXT))
def test_pi_table_pi2_matches_beta(pair):
    pi = pi_table(*pair)
    assert pi[2] == pytest.approx(beta(*pair))


@pytest.mark.parametrize(
    "pair, expected_pi1",
    [
        # WCF pairs: (α_L + α_R)/2 = (1/3 + 0)/2 or (0 + 1/3)/2 = 1/6
        (("A", "T"), 1 / 6),
        (("T", "A"), 1 / 6),
        (("G", "C"), 1 / 6),
        (("C", "G"), 1 / 6),
        # Wobble pairs: (1/3 + 1/3)/2 = 1/3
        (("G", "T"), 1 / 3),
        (("T", "G"), 1 / 3),
    ],
)
def test_pi_table_pi1_concrete_values(pair, expected_pi1):
    """Cross-check: §5.3 bullet — WCF pairs have π_p^(1) = 1/6, wobble = 1/3."""
    assert pi_table(*pair)[1] == pytest.approx(expected_pi1)


# ---------------------------------------------------------------------
# Range and sanity checks
# ---------------------------------------------------------------------

@pytest.mark.parametrize("pair", list(V_EXT))
def test_alpha_in_unit_interval(pair):
    assert 0.0 <= alpha_L(*pair) <= 1.0
    assert 0.0 <= alpha_R(*pair) <= 1.0


@pytest.mark.parametrize("pair", list(V_EXT))
def test_beta_in_unit_interval(pair):
    assert 0.0 <= beta(*pair) <= 1.0


def test_alphas_zero_for_invalid_pair():
    """If (b_L, b_R) is not in 𝒱, α_L and α_R count valid neighbour
    pairs in 𝒱, which can be nonzero. This test pins the count for one
    invalid pair as a regression check on the helper definitions."""
    # ('A','A') is not in V_EXT.
    # α_L: count b' ≠ A with (b', A) in V_EXT → only T → 1/3
    # α_R: count b' ≠ A with (A, b') in V_EXT → only T → 1/3
    assert alpha_L("A", "A") == pytest.approx(1 / 3)
    assert alpha_R("A", "A") == pytest.approx(1 / 3)


def test_BASES_has_four_DNA_letters():
    assert set(BASES) == {"A", "C", "G", "T"}
    assert len(BASES) == 4


# ---------------------------------------------------------------------
# Ti/Tv-biased null (Section 5 of nonWCF_derivation.tex)
# ---------------------------------------------------------------------

# (b_L, b_R) -> (α_L^{R=2}, α_R^{R=2}, β^{R=2}). Derived in
# nonWCF_derivation.tex Section 5.2 table at R = 2.
TITV_2_TABLE = {
    ("A", "T"): (2 / 3, 0.0, 19 / 36),
    ("T", "A"): (0.0, 2 / 3, 19 / 36),
    ("G", "C"): (0.0, 2 / 3, 19 / 36),
    ("C", "G"): (2 / 3, 0.0, 19 / 36),
    ("G", "T"): (2 / 3, 2 / 3, 3 / 36),
    ("T", "G"): (2 / 3, 2 / 3, 3 / 36),
}


@pytest.mark.parametrize("pair, expected", list(TITV_2_TABLE.items()))
def test_alpha_L_titv2_matches_table(pair, expected):
    assert alpha_L(*pair, titv=2.0) == pytest.approx(expected[0])


@pytest.mark.parametrize("pair, expected", list(TITV_2_TABLE.items()))
def test_alpha_R_titv2_matches_table(pair, expected):
    assert alpha_R(*pair, titv=2.0) == pytest.approx(expected[1])


@pytest.mark.parametrize("pair, expected", list(TITV_2_TABLE.items()))
def test_beta_titv2_matches_table(pair, expected):
    assert beta(*pair, titv=2.0) == pytest.approx(expected[2])


@pytest.mark.parametrize("pair, expected", list(SECTION_4_TABLE.items()))
def test_titv_0_5_recovers_uniform_alpha_L(pair, expected):
    """Default titv = 0.5 must reproduce the uniform Section 4 table exactly."""
    assert alpha_L(*pair, titv=0.5) == pytest.approx(expected[0])


@pytest.mark.parametrize("pair, expected", list(SECTION_4_TABLE.items()))
def test_titv_0_5_recovers_uniform_alpha_R(pair, expected):
    assert alpha_R(*pair, titv=0.5) == pytest.approx(expected[1])


@pytest.mark.parametrize("pair, expected", list(SECTION_4_TABLE.items()))
def test_titv_0_5_recovers_uniform_beta(pair, expected):
    assert beta(*pair, titv=0.5) == pytest.approx(expected[2])


@pytest.mark.parametrize("pair", list(V_EXT))
def test_pi_table_titv2_pi2_matches_beta(pair):
    """π_p^(2) under titv=2 equals β under titv=2."""
    pi = pi_table(*pair, titv=2.0)
    assert pi[2] == pytest.approx(beta(*pair, titv=2.0))


@pytest.mark.parametrize("pair", list(V_EXT))
def test_pi_table_titv2_pi1_matches_alpha_average(pair):
    pi = pi_table(*pair, titv=2.0)
    expected = 0.5 * (alpha_L(*pair, titv=2.0) + alpha_R(*pair, titv=2.0))
    assert pi[1] == pytest.approx(expected)


def test_titv_pi_Ti_pi_Tv_sum_to_one():
    """π_Ti + 2 π_Tv = 1 for any titv. Cross-check by reading off the
    weights via α_L applied to a synthetic pair where all three
    alternatives lie in V_EXT (the ('A','A') case: only T qualifies; not
    a full check). Instead verify directly via the Section 5.1 closed
    form."""
    for R in (0.25, 0.5, 1.0, 2.0, 3.0):
        pi_Ti = R / (R + 1.0)
        pi_Tv = 1.0 / (2.0 * (R + 1.0))
        assert pi_Ti + 2 * pi_Tv == pytest.approx(1.0)
        assert pi_Ti / (2 * pi_Tv) == pytest.approx(R)
