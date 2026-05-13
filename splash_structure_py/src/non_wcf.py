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

#: Transition partner for each base (Pu↔Pu, Py↔Py). All other directed
#: swaps are transversions.
_TRANSITION_PARTNER: dict[str, str] = {"A": "G", "G": "A", "C": "T", "T": "C"}


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
# Per-alternative identity weights under Ti/Tv-biased null
# ---------------------------------------------------------------------

def _w(b: str, b_prime: str, titv: float) -> float:
    """Per-alternative identity probability for the swap b → b_prime under
    an aggregate Ti/Tv event ratio ``titv`` = #Ti / #Tv.

    Returns π_Ti = titv / (titv + 1) if b ↔ b_prime is a transition
    (A↔G or C↔T), and π_Tv = 1 / (2 (titv + 1)) otherwise. At titv = 0.5
    both equal 1/3, recovering the uniform null of Section 4.
    """
    if _TRANSITION_PARTNER[b] == b_prime:
        return titv / (titv + 1.0)
    return 1.0 / (2.0 * (titv + 1.0))


# ---------------------------------------------------------------------
# Single-position compatibility probabilities (Section 4)
# ---------------------------------------------------------------------

def alpha_L(b_L: str, b_R: str, titv: float = 0.5) -> float:
    """Probability that a random mutation at the left position of pair
    (b_L, b_R) yields a pair in V_EXT under a Ti/Tv-biased null.

    Generalises Section 4: at ``titv = 0.5`` (uniform), this returns
    (1/3) · |{b' ≠ b_L : (b', b_R) ∈ V_EXT}|.
    """
    return sum(
        _w(b_L, b, titv)
        for b in BASES
        if b != b_L and (b, b_R) in V_EXT
    )


def alpha_R(b_L: str, b_R: str, titv: float = 0.5) -> float:
    """Probability that a random mutation at the right position of pair
    (b_L, b_R) yields a pair in V_EXT under a Ti/Tv-biased null.
    """
    return sum(
        _w(b_R, b, titv)
        for b in BASES
        if b != b_R and (b_L, b) in V_EXT
    )


# ---------------------------------------------------------------------
# Two-position compatibility probability (Section 4, eq:beta)
# ---------------------------------------------------------------------

def beta(b_L: str, b_R: str, titv: float = 0.5) -> float:
    """Probability that simultaneous random mutations at both positions of
    pair (b_L, b_R) yield a pair in V_EXT under a Ti/Tv-biased null.

    Generalises Section 4: at ``titv = 0.5`` (uniform), this returns
    (1/9) · |{(b'_L, b'_R) : b'_L ≠ b_L, b'_R ≠ b_R, (b'_L, b'_R) ∈ V_EXT}|.
    """
    return sum(
        _w(b_L, x, titv) * _w(b_R, y, titv)
        for x in BASES
        if x != b_L
        for y in BASES
        if y != b_R and (x, y) in V_EXT
    )


# ---------------------------------------------------------------------
# Per-mutation-count Bernoulli parameters (Section 5.4)
# ---------------------------------------------------------------------

def pi_table(b_L: str, b_R: str, titv: float = 0.5) -> dict[int, float]:
    """Return the per-mutation-count success probabilities π_p^(j)
    for j ∈ {0, 1, 2}, under a Ti/Tv-biased null.

    π_p^(j) = Pr(E_p = 1 | n_p = j, b_L, b_R) where:
      - π_p^(0) = 0
      - π_p^(1) = (α_L + α_R) / 2
      - π_p^(2) = β

    At ``titv = 0.5`` (default) this recovers the uniform-null table.
    """
    return {
        0: 0.0,
        1: 0.5 * (alpha_L(b_L, b_R, titv) + alpha_R(b_L, b_R, titv)),
        2: beta(b_L, b_R, titv),
    }
