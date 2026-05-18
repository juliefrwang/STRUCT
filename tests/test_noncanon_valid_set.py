"""Phase 7 (revision-plan E1) tests: generalized non-canonical pair set.

Covers:
- normalize_base / parse_noncanon / build_valid_set
- backward-compat anchor: build_valid_set("GU") == V_EXT
- alpha/beta parameterized by the valid set, matching nonWCF §6 worked
  example exactly at R=1/2 and R=2
- the core invariant: a WCF pair's alpha/beta changes when N grows
  (alpha/beta are NOT per-pair-type constants)
"""

import pytest

from splash_structure_py.src.non_wcf import (
    V_EXT,
    V_WCF,
    V_WOBBLE,
    alpha_L,
    alpha_R,
    beta,
    build_valid_set,
    normalize_base,
    parse_noncanon,
    pi_table,
)


# ---------------------------------------------------------------------
# normalize_base
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [("U", "T"), ("u", "T"), ("a", "A"), ("G", "G"), ("c", "C"), ("T", "T")],
)
def test_normalize_base(raw, expected):
    assert normalize_base(raw) == expected


# ---------------------------------------------------------------------
# parse_noncanon — happy paths
# ---------------------------------------------------------------------

def test_parse_gu_symmetrized():
    assert parse_noncanon("GU") == frozenset({("G", "T"), ("T", "G")})


def test_parse_gt_equals_gu():
    # U is normalized to T, so "GT" and "GU" are the same pair.
    assert parse_noncanon("GT") == parse_noncanon("GU")


def test_parse_case_insensitive():
    assert parse_noncanon("gu") == parse_noncanon("GU")


def test_parse_separators_ignored():
    for tok in ("G-U", "G.U", "G:U", "G·U"):
        assert parse_noncanon(tok) == parse_noncanon("GU")


def test_parse_one_orientation_symmetrizes():
    # User gives only U·G; both orientations must come back (D1).
    assert parse_noncanon("UG") == frozenset({("T", "G"), ("G", "T")})


def test_parse_multi_pair():
    assert parse_noncanon("GU,GA") == frozenset(
        {("G", "T"), ("T", "G"), ("G", "A"), ("A", "G")}
    )


@pytest.mark.parametrize("spec", ["none", "NONE", "", "  ", None])
def test_parse_none_is_empty(spec):
    assert parse_noncanon(spec) == frozenset()


def test_parse_homo_pair_allowed():
    # G·G is a non-canonical pair; symmetrization is a no-op.
    assert parse_noncanon("GG") == frozenset({("G", "G")})


# ---------------------------------------------------------------------
# parse_noncanon — error paths
# ---------------------------------------------------------------------

@pytest.mark.parametrize("spec", ["AU", "UA", "GC", "CG", "AT", "TA"])
def test_parse_rejects_wcf(spec):
    # These normalize to WCF pairs; passing one as non-canonical is an error.
    with pytest.raises(ValueError, match="Watson-Crick"):
        parse_noncanon(spec)


@pytest.mark.parametrize("spec", ["XY", "GX", "1U"])
def test_parse_rejects_invalid_base(spec):
    with pytest.raises(ValueError, match="invalid base"):
        parse_noncanon(spec)


@pytest.mark.parametrize("spec", ["G", "GUU", "GU,G", "GUGA"])
def test_parse_rejects_malformed_token(spec):
    with pytest.raises(ValueError, match="exactly two bases"):
        parse_noncanon(spec)


# ---------------------------------------------------------------------
# build_valid_set + backward-compat anchor
# ---------------------------------------------------------------------

def test_build_gu_equals_V_EXT():
    """The core backward-compat anchor: GU spec reproduces V_EXT exactly."""
    assert build_valid_set("GU") == V_EXT
    assert build_valid_set("GU") == V_WCF | V_WOBBLE


def test_build_none_is_wcf_only():
    assert build_valid_set("none") == V_WCF
    assert build_valid_set(None) == V_WCF
    assert build_valid_set("") == V_WCF


def test_build_default_is_gu():
    assert build_valid_set() == V_EXT


def test_build_multi_pair():
    assert build_valid_set("GU,GA") == V_WCF | frozenset(
        {("G", "T"), ("T", "G"), ("G", "A"), ("A", "G")}
    )


# ---------------------------------------------------------------------
# alpha parameterized by valid set — nonWCF §6 worked example (G,C)
# row: alpha_R for C→b' keeping (G, b') valid
# ---------------------------------------------------------------------

V_GU = build_valid_set("GU")
V_GA = build_valid_set("GA")
V_GUGA = build_valid_set("GU,GA")


@pytest.mark.parametrize(
    "valid, titv, expected",
    [
        (V_GU,   0.5, 1 / 3),   # C→T (Ti), w=1/3 at R=1/2
        (V_GA,   0.5, 1 / 3),   # C→A (Tv), w=1/3 at R=1/2
        (V_GUGA, 0.5, 2 / 3),   # both routes
        (V_GU,   2.0, 2 / 3),   # C→T (Ti), w=2/3 at R=2
        (V_GA,   2.0, 1 / 6),   # C→A (Tv), w=1/6 at R=2
        (V_GUGA, 2.0, 2 / 3 + 1 / 6),  # = 5/6
    ],
)
def test_alpha_R_GC_worked_example(valid, titv, expected):
    """Matches nonWCF_derivation.tex §6 worked-example table exactly."""
    assert alpha_R("G", "C", titv, valid) == pytest.approx(expected)


# ---------------------------------------------------------------------
# Core invariant: alpha/beta of a WCF pair change when N grows.
# alpha/beta are NOT per-pair-type constants.
# ---------------------------------------------------------------------

def test_wcf_pair_alpha_changes_with_N():
    """(G,C) is WCF, yet its alpha_R differs between N={GU} and
    N={GU,GA} — the central correctness invariant of Phase 7."""
    a_gu = alpha_R("G", "C", 0.5, V_GU)
    a_guga = alpha_R("G", "C", 0.5, V_GUGA)
    assert a_gu == pytest.approx(1 / 3)
    assert a_guga == pytest.approx(2 / 3)
    assert a_guga > a_gu  # monotonicity in N


def test_wcf_pair_beta_changes_with_N():
    """(G,C) beta also grows with N (4/9 → 5/9 at uniform R)."""
    b_gu = beta("G", "C", 0.5, V_GU)
    b_guga = beta("G", "C", 0.5, V_GUGA)
    assert b_gu == pytest.approx(4 / 9)
    assert b_guga == pytest.approx(5 / 9)
    assert b_guga > b_gu


def test_pi_table_threads_valid_set():
    """pi_table must reflect the valid set, not a frozen G·U table."""
    pi_gu = pi_table("G", "C", 0.5, V_GU)
    pi_guga = pi_table("G", "C", 0.5, V_GUGA)
    assert pi_gu[2] == pytest.approx(4 / 9)
    assert pi_guga[2] == pytest.approx(5 / 9)
    assert pi_guga[1] > pi_gu[1]


# ---------------------------------------------------------------------
# Backward compatibility: default path unchanged
# ---------------------------------------------------------------------

@pytest.mark.parametrize("pair", list(V_EXT))
def test_default_valid_matches_explicit_VEXT(pair):
    """Calling without `valid` (default V_EXT) equals passing
    build_valid_set('GU') explicitly — at uniform and biased R."""
    for titv in (0.5, 2.0):
        assert alpha_L(*pair, titv) == pytest.approx(
            alpha_L(*pair, titv, V_GU)
        )
        assert alpha_R(*pair, titv) == pytest.approx(
            alpha_R(*pair, titv, V_GU)
        )
        assert beta(*pair, titv) == pytest.approx(
            beta(*pair, titv, V_GU)
        )
