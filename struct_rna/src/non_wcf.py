"""Constants and helpers for the non-WCF (G·U wobble) extension.

Implements the per-pair-type quantities defined in Section 4 of
``nonWCF_derivation.tex``: the extended valid pair set 𝒱, the
single-position compatibility probabilities α_L/α_R, the two-position
compatibility probability β, and the per-mutation-count Bernoulli
parameters π_p^(j) used by the Section 5.4 dynamic-programming
algorithm.

Bases use the cDNA alphabet (T represents U in the RNA molecule).

Phase 7 (revision-plan E1): the valid pair set is generalized from a
hard-coded G·U set to a user-specified V(N) = V_WCF ∪ N, via
``build_valid_set`` / the ``valid`` parameter on α/β/π. Defaults are
unchanged (``valid = V_EXT`` = V_WCF ∪ G·U), so existing pipelines are
byte-identical.
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
# Generalized valid pair set V(N) = V_WCF ∪ N  (nonWCF_derivation §6)
# ---------------------------------------------------------------------

_BASES_SET: frozenset[str] = frozenset("ACGT")
_PAIR_SEPARATORS: tuple[str, ...] = ("-", ".", ":", "·")


def normalize_base(c: str) -> str:
    """Uppercase and map U→T (the working alphabet is cDNA ``ACGT``)."""
    c = c.upper()
    return "T" if c == "U" else c


def parse_noncanon(spec: str | None) -> frozenset[tuple[str, str]]:
    """Parse a non-canonical pair specification into a swap-closed
    frozenset of ordered ``(b_L, b_R)`` base pairs.

    Accepts comma-separated two-base tokens, e.g. ``"GU"``, ``"GU,GA"``,
    ``"G-U,G-A"`` (an optional ``-``, ``.``, ``:`` or ``·`` separator
    inside a token is ignored). Bases are uppercased and ``U`` is
    normalized to ``T``. The result is **symmetrized** (decision D1):
    for every ``(x, y)`` both ``(x, y)`` and ``(y, x)`` are included.

    ``"none"``, ``""`` and ``None`` denote the empty set (WCF-only /
    legacy path).

    Raises ``ValueError`` on a malformed token, an unknown base, or a
    token that normalizes to a Watson-Crick-Franklin pair (WCF pairs are
    always valid; passing one as ``non-canonical`` is almost certainly a
    mistake — e.g. ``"AU"`` → ``(A,T)``).
    """
    if spec is None:
        return frozenset()
    spec = spec.strip()
    if spec == "" or spec.lower() == "none":
        return frozenset()

    pairs: set[tuple[str, str]] = set()
    for raw in spec.split(","):
        token = raw.strip()
        for sep in _PAIR_SEPARATORS:
            token = token.replace(sep, "")
        if len(token) != 2:
            raise ValueError(
                f"non-canonical token {raw!r} must be exactly two bases "
                f"(optionally separated by -, ., : or ·)"
            )
        b_L, b_R = normalize_base(token[0]), normalize_base(token[1])
        for b in (b_L, b_R):
            if b not in _BASES_SET:
                raise ValueError(
                    f"non-canonical token {raw!r} contains an invalid base "
                    f"{b!r}; allowed bases are A C G T (U is read as T)"
                )
        if (b_L, b_R) in V_WCF:
            raise ValueError(
                f"non-canonical token {raw!r} normalizes to the "
                f"Watson-Crick-Franklin pair {(b_L, b_R)}; WCF pairs are "
                f"always valid and must not be given as non-canonical"
            )
        pairs.add((b_L, b_R))
        pairs.add((b_R, b_L))  # symmetrize (D1)
    return frozenset(pairs)


def build_valid_set(spec: str | None = "GU") -> frozenset[tuple[str, str]]:
    """Return the valid pair set ``V(N) = V_WCF ∪ parse_noncanon(spec)``.

    ``spec="GU"`` reproduces the G·U-extended set (equal to the module
    constant ``V_EXT``); ``spec="none"`` (or ``""`` / ``None``) gives the
    legacy WCF-only set (equal to ``V_WCF``).
    """
    return V_WCF | parse_noncanon(spec)


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

def alpha_L(
    b_L: str, b_R: str, titv: float = 0.5,
    valid: frozenset[tuple[str, str]] = V_EXT,
) -> float:
    """Probability that a random mutation at the left position of pair
    (b_L, b_R) yields a pair in the valid set under a Ti/Tv-biased null.

    ``valid`` is V(N) (default ``V_EXT`` = V_WCF ∪ G·U, preserving prior
    behaviour). At ``titv = 0.5`` (uniform) this returns
    (1/3) · |{b' ≠ b_L : (b', b_R) ∈ valid}|.
    """
    return sum(
        _w(b_L, b, titv)
        for b in BASES
        if b != b_L and (b, b_R) in valid
    )


def alpha_R(
    b_L: str, b_R: str, titv: float = 0.5,
    valid: frozenset[tuple[str, str]] = V_EXT,
) -> float:
    """Probability that a random mutation at the right position of pair
    (b_L, b_R) yields a pair in the valid set under a Ti/Tv-biased null.
    """
    return sum(
        _w(b_R, b, titv)
        for b in BASES
        if b != b_R and (b_L, b) in valid
    )


# ---------------------------------------------------------------------
# Two-position compatibility probability (Section 4, eq:beta)
# ---------------------------------------------------------------------

def beta(
    b_L: str, b_R: str, titv: float = 0.5,
    valid: frozenset[tuple[str, str]] = V_EXT,
) -> float:
    """Probability that simultaneous random mutations at both positions of
    pair (b_L, b_R) yield a pair in the valid set under a Ti/Tv-biased null.

    ``valid`` is V(N) (default ``V_EXT``). At ``titv = 0.5`` (uniform)
    this returns
    (1/9) · |{(b'_L, b'_R) : b'_L ≠ b_L, b'_R ≠ b_R, (b'_L, b'_R) ∈ valid}|.
    """
    return sum(
        _w(b_L, x, titv) * _w(b_R, y, titv)
        for x in BASES
        if x != b_L
        for y in BASES
        if y != b_R and (x, y) in valid
    )


# ---------------------------------------------------------------------
# Per-mutation-count Bernoulli parameters (Section 5.4)
# ---------------------------------------------------------------------

def pi_table(
    b_L: str, b_R: str, titv: float = 0.5,
    valid: frozenset[tuple[str, str]] = V_EXT,
) -> dict[int, float]:
    """Return the per-mutation-count success probabilities π_p^(j)
    for j ∈ {0, 1, 2}, under a Ti/Tv-biased null and valid set V(N).

    π_p^(j) = Pr(E_p = 1 | n_p = j, b_L, b_R) where:
      - π_p^(0) = 0
      - π_p^(1) = (α_L + α_R) / 2
      - π_p^(2) = β

    At ``titv = 0.5`` and ``valid = V_EXT`` (defaults) this recovers the
    uniform-null G·U table.
    """
    return {
        0: 0.0,
        1: 0.5 * (alpha_L(b_L, b_R, titv, valid) + alpha_R(b_L, b_R, titv, valid)),
        2: beta(b_L, b_R, titv, valid),
    }
