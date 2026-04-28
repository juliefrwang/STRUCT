"""Constants and helpers for the non-WCF (G·U wobble) extension.

Implements the per-pair-type quantities defined in Section 4 of
``nonWCF_derivation.tex``: the extended valid pair set 𝒱, the
single-position compatibility probabilities α_L/α_R, the two-position
compatibility probability β, and the per-mutation-count Bernoulli
parameters π_p^(j) used by the Section 5.4 dynamic-programming
algorithm.

Bases use the cDNA alphabet (T represents U in the RNA molecule).

Phase 0 of feature/non-wcf — no behaviour change to existing pipelines.
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# Valid pair sets
# ---------------------------------------------------------------------

#: Strict Watson-Crick-Franklin pairs (cDNA alphabet).
V_WCF: frozenset[tuple[str, str]] = frozenset(
    {("A", "T"), ("T", "A"), ("G", "C"), ("C", "G")}
)

#: G·U wobble pairs (cDNA alphabet).
V_WOBBLE: frozenset[tuple[str, str]] = frozenset({("G", "T"), ("T", "G")})

#: Extended valid pair set 𝒱 = V_WCF ∪ V_WOBBLE.
V_EXT: frozenset[tuple[str, str]] = V_WCF | V_WOBBLE

#: Alphabet used for null-model mutation enumeration.
BASES: tuple[str, str, str, str] = ("A", "C", "G", "T")


# ---------------------------------------------------------------------
# Pair classification
# ---------------------------------------------------------------------

def pair_type(b_L: str, b_R: str) -> str | None:
    """Return ``'wcf'``, ``'wobble'``, or ``None`` for the pair (b_L, b_R)."""
    pair = (b_L, b_R)
    if pair in V_WCF:
        return "wcf"
    if pair in V_WOBBLE:
        return "wobble"
    return None


# ---------------------------------------------------------------------
# Single-position compatibility probabilities (Section 4)
# ---------------------------------------------------------------------

def alpha_L(b_L: str, b_R: str) -> float:
    """Probability that a random mutation at the left position of pair
    (b_L, b_R) yields a pair in V_EXT.

    α_L(b_L, b_R) = (1/3) · |{b' ≠ b_L : (b', b_R) ∈ V_EXT}|.
    """
    count = sum(1 for b in BASES if b != b_L and (b, b_R) in V_EXT)
    return count / 3.0


def alpha_R(b_L: str, b_R: str) -> float:
    """Probability that a random mutation at the right position of pair
    (b_L, b_R) yields a pair in V_EXT.

    α_R(b_L, b_R) = (1/3) · |{b' ≠ b_R : (b_L, b') ∈ V_EXT}|.
    """
    count = sum(1 for b in BASES if b != b_R and (b_L, b) in V_EXT)
    return count / 3.0


# ---------------------------------------------------------------------
# Two-position compatibility probability (Section 4, eq:beta)
# ---------------------------------------------------------------------

def beta(b_L: str, b_R: str) -> float:
    """Probability that simultaneous random mutations at both positions of
    pair (b_L, b_R) yield a pair in V_EXT.

    β(b_L, b_R) = (1/9) · |{(b'_L, b'_R) : b'_L ≠ b_L, b'_R ≠ b_R,
                            (b'_L, b'_R) ∈ V_EXT}|.
    """
    count = sum(
        1
        for x in BASES
        if x != b_L
        for y in BASES
        if y != b_R and (x, y) in V_EXT
    )
    return count / 9.0


# ---------------------------------------------------------------------
# Per-mutation-count Bernoulli parameters (Section 5.4)
# ---------------------------------------------------------------------

def pi_table(b_L: str, b_R: str) -> dict[int, float]:
    """Return the per-mutation-count success probabilities π_p^(j)
    for j ∈ {0, 1, 2}.

    π_p^(j) = Pr(E_p = 1 | n_p = j, b_L, b_R) where:
      - π_p^(0) = 0
      - π_p^(1) = (α_L + α_R) / 2
      - π_p^(2) = β
    """
    return {
        0: 0.0,
        1: 0.5 * (alpha_L(b_L, b_R) + alpha_R(b_L, b_R)),
        2: beta(b_L, b_R),
    }
