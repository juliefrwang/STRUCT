"""Unit tests for find_mutation_ext.

The base target ``"AGGAAAATCTAAA"`` (length 13) has the following stem:
- Left stem at indices [0, 2] inclusive: ``A G G``
- Loop at indices [3, 6]: ``AAAA``
- Right stem at indices [7, 9] inclusive: ``T C T``
- Trailing region [10, 12]: ``AAA``

Pairs (0-indexed from the outside in, using reverse-complement pairing):

- Pair 0: positions (0, 9) → ``(A, T)`` WCF
- Pair 1: positions (1, 8) → ``(G, C)`` WCF
- Pair 2: positions (2, 7) → ``(G, T)`` wobble
"""

import math

import pytest

from splash_structure_py.src.find_comp_mut import find_mutation, find_mutation_ext

BASE = "AGGAAAATCTAAA"
STEM = (0, 2, 7, 9)  # (stem_start_idx, stem_end_idx, rc_start_idx, rc_end_idx)
EXPECTED_B = [("A", "T"), ("G", "C"), ("G", "T")]


def call_ext(target):
    """Convenience wrapper."""
    return find_mutation_ext(BASE, target, *STEM)


# ---------------------------------------------------------------------
# Trivial / structural sanity
# ---------------------------------------------------------------------

def test_no_mutations():
    totaMut, stemMut, E, struc, n_p, b = call_ext(BASE)
    assert totaMut == 0
    assert stemMut == 0
    assert E == 0
    assert n_p == [0, 0, 0]
    assert b == EXPECTED_B
    assert struc == "{---(----)---}---"


def test_b_vector_is_base_target_composition():
    _, _, _, _, _, b = call_ext(BASE)
    assert b == EXPECTED_B


def test_n_p_sums_to_stemMut():
    """Self-consistency: stemMut must equal sum(n_p_vector)."""
    target = "GGAAAAATCAAAA"  # arbitrary
    _, stemMut, _, _, n_p, _ = call_ext(target)
    assert stemMut == sum(n_p)


# ---------------------------------------------------------------------
# SPC events (single mutation, pair stays in V_EXT)
# ---------------------------------------------------------------------

def test_spc_left_wcf_to_wobble():
    """Pair 0: (A, T) → mutate pos 0 A→G yields (G, T) wobble: SPC-left."""
    target = "GGGAAAATCTAAA"
    totaMut, stemMut, E, struc, n_p, b = call_ext(target)
    assert totaMut == 1
    assert stemMut == 1
    assert E == 1
    assert n_p == [1, 0, 0]
    assert struc[1] == "G"  # uppercase, pos 0 in struc -> after '{', char index 1


def test_spc_right_wcf_to_wobble():
    """Pair 1: (G, C) → mutate pos 8 C→T yields (G, T) wobble: SPC-right."""
    target = "AGGAAAATTTAAA"
    totaMut, stemMut, E, struc, n_p, b = call_ext(target)
    assert totaMut == 1
    assert stemMut == 1
    assert E == 1
    assert n_p == [0, 1, 0]


def test_spc_wobble_to_wcf():
    """Pair 2 starts as wobble (G, T). Mutate pos 2 G→A yields (A, T) WCF.
    This is the case the original WCF-only code misses entirely."""
    target = "AGAAAAATCTAAA"
    totaMut, stemMut, E, struc, n_p, b = call_ext(target)
    assert totaMut == 1
    assert stemMut == 1
    assert E == 1
    assert n_p == [0, 0, 1]


# ---------------------------------------------------------------------
# BPC events (both positions mutated, new pair in V_EXT)
# ---------------------------------------------------------------------

def test_bpc_wcf_to_wcf():
    """Pair 0: (A, T) → (G, C) — both positions mutate, new pair WCF."""
    target = "GGGAAAATCCAAA"
    totaMut, stemMut, E, struc, n_p, b = call_ext(target)
    assert totaMut == 2
    assert stemMut == 2
    assert E == 1
    assert n_p == [2, 0, 0]


def test_bpc_to_wobble():
    """Pair 1: (G, C) → (T, G) — both mutate, new pair wobble."""
    # Base:   A G G A A A A T C T A A A   (indices 0..12)
    # Target: A T G A A A A T G T A A A   (pos 1 G→T, pos 8 C→G)
    target = "ATGAAAATGTAAA"
    totaMut, stemMut, E, struc, n_p, b = call_ext(target)
    assert totaMut == 2
    assert stemMut == 2
    assert E == 1
    assert n_p == [0, 2, 0]


# ---------------------------------------------------------------------
# Structure-disrupting events
# ---------------------------------------------------------------------

def test_disrupting_left_only():
    """Pair 1: (G, C) → mutate pos 1 G→A. New pair (A, C) ∉ V_EXT."""
    target = "AAGAAAATCTAAA"
    totaMut, stemMut, E, struc, n_p, b = call_ext(target)
    assert totaMut == 1
    assert stemMut == 1
    assert E == 0
    assert n_p == [0, 1, 0]


def test_disrupting_right_only():
    """Pair 0: (A, T) → mutate pos 9 T→A. New pair (A, A) ∉ V_EXT."""
    target = "AGGAAAATCAAAA"
    totaMut, stemMut, E, struc, n_p, b = call_ext(target)
    assert totaMut == 1
    assert stemMut == 1
    assert E == 0
    assert n_p == [1, 0, 0]


def test_disrupting_both():
    """Pair 0: (A, T) → both mutate to (C, A) ∉ V_EXT."""
    target = "CGGAAAATCAAAA"
    totaMut, stemMut, E, struc, n_p, b = call_ext(target)
    assert totaMut == 2
    assert stemMut == 2
    assert E == 0
    assert n_p == [2, 0, 0]


# ---------------------------------------------------------------------
# Mutations outside the stem
# ---------------------------------------------------------------------

def test_mutations_outside_stem():
    """Mutations at loop and trailing positions don't change E or n_p."""
    target = "AGGTAAATCTACA"
    # pos 3 A→T (loop), pos 11 A→C (trailing)
    totaMut, stemMut, E, struc, n_p, b = call_ext(target)
    assert totaMut == 2
    assert stemMut == 0
    assert E == 0
    assert n_p == [0, 0, 0]


# ---------------------------------------------------------------------
# Combined events
# ---------------------------------------------------------------------

def test_three_events_combined():
    """Pair 0 BPC + Pair 1 SPC + Pair 2 disrupting + outside-stem mutation.

    Construct:
    - Pair 0: (A, T) → (G, C): pos 0 A→G, pos 9 T→C  (BPC)
    - Pair 1: (G, C) → (G, T): pos 8 C→T              (SPC-right)
    - Pair 2: (G, T) → (C, T): pos 2 G→C              (disrupting, (C, T) ∉ V_EXT)
    - Outside: pos 3 A→T (loop)
    """
    target_chars = list(BASE)
    target_chars[0] = "G"   # pair 0 left
    target_chars[9] = "C"   # pair 0 right
    target_chars[8] = "T"   # pair 1 right
    target_chars[2] = "C"   # pair 2 left, disrupting
    target_chars[3] = "T"   # loop mutation
    target = "".join(target_chars)

    totaMut, stemMut, E, struc, n_p, b = call_ext(target)
    assert totaMut == 5
    assert stemMut == 4
    assert E == 2  # pair 0 BPC + pair 1 SPC
    assert n_p == [2, 1, 1]
    assert b == EXPECTED_B


# ---------------------------------------------------------------------
# Stem at non-zero start (verifies index handling)
# ---------------------------------------------------------------------

def test_stem_with_offset_start():
    """Same pair structure but stem starts at index 2 of a length-15 target."""
    base = "TT" + "AGG" + "AAAA" + "TCT" + "AAA"  # length 15
    stem = (2, 4, 9, 11)
    target_chars = list(base)
    target_chars[2] = "G"  # was 'A', now (G, T) wobble — SPC-left at pair 0
    target = "".join(target_chars)

    totaMut, stemMut, E, struc, n_p, b = find_mutation_ext(base, target, *stem)
    assert totaMut == 1
    assert stemMut == 1
    assert E == 1
    assert n_p == [1, 0, 0]
    assert b == [("A", "T"), ("G", "C"), ("G", "T")]


# ---------------------------------------------------------------------
# Notation conventions
# ---------------------------------------------------------------------

def test_struc_uppercase_for_spc():
    target = "GGGAAAATCTAAA"  # SPC-left at pair 0
    *_, struc, _, _ = call_ext(target)
    # struc layout: pos 0 in BASE is at struc index 1 (after '{')
    assert struc[1] == "G"  # uppercase


def test_struc_uppercase_for_bpc():
    target = "GGGAAAATCCAAA"  # BPC at pair 0
    *_, struc, _, _ = call_ext(target)
    # pos 0 → struc[1]; pos 9 → struc[12] (after { at 0, ( at 4, ) at 9, } at 13)
    # base length 13, struc length 13 + 4 = 17
    # Index map: positions 0-2 → struc 1-3, positions 7-9 → struc 10-12
    assert struc[1] == "G"  # left, uppercase
    assert struc[12] == "C"  # right, uppercase


def test_struc_lowercase_for_disrupting():
    target = "AAGAAAATCTAAA"  # pair 1 disrupting at pos 1
    *_, struc, _, _ = call_ext(target)
    # pos 1 → struc[2]
    assert struc[2] == "a"  # lowercase


def test_struc_dash_for_unaffected_position_in_spc():
    """For SPC at pair 0 (pos 0 mutated), the unmutated partner (pos 9)
    must show '-', not lowercase, since the pair is structure-supporting."""
    target = "GGGAAAATCTAAA"
    *_, struc, _, _ = call_ext(target)
    # pos 9 → struc[12]
    assert struc[12] == "-"


# ---------------------------------------------------------------------
# Backward compatibility check: old find_mutation still works
# ---------------------------------------------------------------------

def test_original_find_mutation_unchanged():
    """The original WCF-only find_mutation is preserved untouched."""
    target = "GGGAAAATCCAAA"  # BPC at pair 0
    totaMut, stemMut, compMut, struc = find_mutation(BASE, target, *STEM)
    assert totaMut == 2
    assert stemMut == 2
    assert compMut == 1  # one compensatory pair


# ---------------------------------------------------------------------
# Compactor-mode no-stem-half sentinel (Phase C2)
# ---------------------------------------------------------------------

def test_no_stem_sentinel_returns_empty_vectors():
    """Compactor mode calls find_mutation_ext on each of the two halves,
    including halves where the stem-finder returned the no-stem sentinel
    (0, 0, 0, 0, 0). The function must return empty per-pair vectors and
    NaN struc rather than fabricating a phantom L=1 pair at index 0.
    """
    import math

    base = "ACGTACGTACGT"
    target = "ACGTACGAACGA"  # differs at indices 7 and 11; index 0 unchanged
    totaMut, stemMut, E, struc, n_p, b = find_mutation_ext(
        base, target, 0, 0, 0, 0
    )
    assert totaMut == 2  # outside-stem Hamming distance over the whole half
    assert stemMut == 0
    assert E == 0
    assert n_p == []  # no phantom pair
    assert b == []  # no phantom composition entry
    assert isinstance(struc, float) and math.isnan(struc)


def test_no_stem_sentinel_with_mutation_at_index_zero():
    """Regression for the previous L = stem_end_idx - stem_start_idx + 1
    bug: when both indices are 0, naive arithmetic would set L = 1 and
    treat position 0 as a phantom stem pair, sometimes counting an
    index-0 mutation toward stemMut. The early return must include any
    index-0 difference in totaMut (outside-stem walk) and keep stemMut
    at 0.
    """
    base = "ACGT"
    target = "GCGT"  # mutation at index 0
    totaMut, stemMut, E, struc, n_p, b = find_mutation_ext(
        base, target, 0, 0, 0, 0
    )
    assert totaMut == 1
    assert stemMut == 0
    assert E == 0
    assert n_p == []
    assert b == []
