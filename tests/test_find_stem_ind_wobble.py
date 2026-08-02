"""Unit tests for find_stem_ind_wobble.

Verifies:
- Backward compatibility with find_stem_ind on WCF-only stems
  (when no longer cap-satisfying alternative exists)
- Detection of wobble-using stems that the original would miss
- Selection order (longest first, then fewest wobble, then leftmost)
- Wobble cap of floor(L/2): all-wobble stems are rejected
- Threshold behaviour (stem_L parameter)
"""

import pytest

from splash_structure_py.src.process_targets import (
    find_stem_ind,
    find_stem_ind_wobble,
)


# Helper for tolerant comparison since the function may return a list
# (for the "no stem" sentinel) or a tuple.
def as_list(result):
    return list(result)


# ---------------------------------------------------------------------
# No-stem cases
# ---------------------------------------------------------------------

def test_no_stem_returns_zeros():
    """Sequence too short for any stem."""
    assert as_list(find_stem_ind_wobble("AAAA", stem_L=5)) == [0, 0, 0, 0, 0]


def test_no_valid_pairing_returns_zeros():
    """Sequence with no possible WCF or wobble pairing."""
    # All A's: only valid pairs would need 'T's, none present.
    assert as_list(find_stem_ind_wobble("AAAAAAAAAAAA", stem_L=5)) == [0, 0, 0, 0, 0]


def test_max_size_below_stem_L_returns_zeros():
    """Target whose half-length is below stem_L."""
    # length 8 means max_size = 4; with stem_L=5 nothing qualifies.
    assert as_list(find_stem_ind_wobble("AGCATGCT", stem_L=5)) == [0, 0, 0, 0, 0]


# ---------------------------------------------------------------------
# Backward compatibility: pure-WCF stems
# ---------------------------------------------------------------------

WCF_TARGETS = [
    # (target, expected (stem_start, stem_end, rc_start, rc_end, stemL))
    # 5-bp stem with TTT loop. left "AGCAA" / right "TTGCT" (rc of AGCAA).
    ("AGCAATTTTTGCT",
     (0, 4, 8, 12, 5)),
    # 6-bp stem with AAAA loop. left "AGCATC" / right "GATGCT" (rc of AGCATC).
    ("AGCATCAAAAGATGCT",
     (0, 5, 10, 15, 6)),
]


@pytest.mark.parametrize("target, expected", WCF_TARGETS)
def test_wcf_only_target_matches_original(target, expected):
    """When a WCF stem exists, find_stem_ind_wobble returns the same indices
    as find_stem_ind (the WCF stem has 0 wobble, dominating tie-breaking)."""
    wcf = find_stem_ind(target, stem_L=5)
    wobble = find_stem_ind_wobble(target, stem_L=5)
    assert as_list(wobble) == as_list(wcf)
    assert as_list(wobble) == list(expected)


# ---------------------------------------------------------------------
# Wobble-using stems
# ---------------------------------------------------------------------

def test_single_wobble_in_stem():
    """5-bp stem with one wobble at pair 1 (G·U). Strict WCF cannot detect
    this whole stem, but find_stem_ind_wobble can."""
    # Left stem 0-4 = "AGCAA"; right stem 8-12 = "TTGTT".
    # Pairs: (A,T) WCF, (G,T) wobble, (C,G) WCF, (A,T) WCF, (A,T) WCF.
    target = "AGCAATTTTTGTT"
    result = find_stem_ind_wobble(target, stem_L=5)
    assert as_list(result) == [0, 4, 8, 12, 5]


def test_all_wobble_stem_rejected_by_cap():
    """All-wobble stems violate the floor(L/2) wobble cap and are rejected.

    A length-5 all-wobble stem has n_wobble = 5, which exceeds the cap
    floor(5/2) = 2. No alternative exists in this target, so the result
    is the no-stem sentinel.
    """
    target = "GGGGGTTTTT"  # length 10, 5G + 5T; only valid pairing is all-wobble.
    result = find_stem_ind_wobble(target, stem_L=5)
    assert as_list(result) == [0, 0, 0, 0, 0]


def test_wobble_only_target_no_cap_satisfying_stem():
    """When the only stems available are wobble-piles, the strict WCF
    finder returns no stem AND the wobble-aware finder (with cap) also
    returns no stem. The two finders agree by construction here, since
    the cap excludes the degenerate all-wobble case."""
    target = "GGGGGTTTTT"
    wcf = find_stem_ind(target, stem_L=5)
    wobble = find_stem_ind_wobble(target, stem_L=5)
    assert as_list(wcf) == [0, 0, 0, 0, 0]
    assert as_list(wobble) == [0, 0, 0, 0, 0]


# ---------------------------------------------------------------------
# Selection order: longest first, fewest wobble as tiebreaker
# ---------------------------------------------------------------------

def test_longer_wobble_stem_beats_shorter_wcf():
    """Length-greedy selection: a 5-pair stem with 1 wobble (within the
    floor(5/2) = 2 cap) beats a shorter 3-pair WCF stem at a different
    position in the same target.

    Target construction (stem_L=3 for compactness, with C-rich linkers to
    prevent unintended A·T accidental stems):
      - 3-pair WCF stem at 0-2 / 5-7 ("AGC" / "GCT").
      - 5-pair stem with 1 wobble at 12-16 / 19-23 ("AGCAT" / "ATGTT");
        pair 1 is (G, T) wobble, all others WCF.

    Under length-first priority the 5-pair stem wins despite having a
    wobble pair.
    """
    # Layout: "AGC"(0-2) + "CC"(3-4) + "GCT"(5-7) + "CCCC"(8-11)
    #       + "AGCAT"(12-16) + "CC"(17-18) + "ATGTT"(19-23)
    target = "AGCCCGCTCCCCAGCATCCATGTT"
    assert len(target) == 24
    result = find_stem_ind_wobble(target, stem_L=3)
    assert as_list(result) == [12, 16, 19, 23, 5]


def test_longer_stem_preferred_when_wobble_count_equal():
    """Two WCF stems available; the longer one wins."""
    # 6-pair WCF stem in a 16-character target.
    target = "AGCATCAAAAGATGCT"
    result = find_stem_ind_wobble(target, stem_L=5)
    # The 6-pair WCF stem wins over any 5-pair sub-stem under length-first.
    assert as_list(result) == [0, 5, 10, 15, 6]


# ---------------------------------------------------------------------
# Wobble cap (floor(L/2))
# ---------------------------------------------------------------------

def test_wobble_cap_allows_at_boundary():
    """A 5-pair stem with exactly 2 wobble pairs (= floor(5/2)) is accepted."""
    # Left "GAGAA" / right "TTTTC". Pairs: (G,C)WCF, (A,T)WCF, (G,T)wobble,
    # (A,T)WCF, (A,T)WCF. n_wobble = 1 — too few. Let me use a different one:
    # Left "GGAAA" / right "TTTTC". Pairs:
    #   (G,C) WCF, (G,T) wobble, (A,T) WCF, (A,T) WCF, (A,T) WCF → n_wobble=1.
    # Need exactly 2 wobble pairs in 5. Use:
    # Left "GGGAA" / right "TTTCC". Pairs:
    #   (G,C) WCF, (G,C) WCF, (G,T) wobble, (A,T) WCF, (A,T) WCF? Need check.
    # Easier: construct from scratch.
    # 5 pairs with 2 wobble: e.g., (A,T)(G,T)(G,T)(C,G)(A,T) — left = AGGCA,
    # right (read right-to-left) = TTTGT → right segment left-to-right = TGTTT.
    target = "AGGCA" + "TTT" + "TGTTT"  # 5 + 3 + 5 = 13
    assert len(target) == 13
    # Stem at 0-4 / 8-12. Pair p uses left[p], right[k+(i-1-p)] with k=8, i=5:
    #  p=0: A vs T (right[12])  → (A,T) WCF
    #  p=1: G vs T (right[11])  → (G,T) wobble
    #  p=2: G vs T (right[10])  → (G,T) wobble
    #  p=3: C vs G (right[9])   → (C,G) WCF
    #  p=4: A vs T (right[8])   → (A,T) WCF
    # n_wobble = 2, cap floor(5/2) = 2 → accepted at boundary.
    result = find_stem_ind_wobble(target, stem_L=5)
    assert as_list(result) == [0, 4, 8, 12, 5]


def test_wobble_cap_rejects_just_above():
    """A 5-pair stem with 3 wobble pairs (> floor(5/2) = 2) is rejected
    when no cap-satisfying alternative exists in the target."""
    # 3 wobble pairs in 5: e.g., (A,T)(G,T)(G,T)(G,T)(A,T) — left "AGGGA",
    # right (left-to-right) "TGTTT" — actually that's (A,T)(G,T)(G,T)(G,T)(A,T):
    # pair p uses right[k+(i-1-p)]:
    # left = AGGGA, right segment = TGTTT (so right[0..4] reading along k):
    # pairs: (A, right[4]=T) (G, right[3]=T) (G, right[2]=T) (G, right[1]=G)
    # (A, right[0]=T). Wait that pairs G with G which isn't valid.
    # Try right segment = TTTGT instead, left = AGGGA:
    #  p=0: A vs T (right[4]=T) WCF
    #  p=1: G vs G (right[3]=G) → invalid pair. Doesn't work.
    # Try left = AGGGA, right segment = TTTTT:
    #  p=0: A vs T WCF
    #  p=1: G vs T wobble
    #  p=2: G vs T wobble
    #  p=3: G vs T wobble
    #  p=4: A vs T WCF
    # 3 wobble pairs. n_wobble = 3 > cap 2. Rejected.
    target = "AGGGA" + "TTT" + "TTTTT"
    assert len(target) == 13
    result = find_stem_ind_wobble(target, stem_L=5)
    assert as_list(result) == [0, 0, 0, 0, 0]


def test_wobble_cap_scales_with_L():
    """Cap is floor(L/2): a 7-pair stem accepts up to 3 wobble pairs
    (since floor(7/2) = 3), while a 5-pair stem caps at 2.

    Constructed target with a 7-pair stem of pairs
      (G,T) (G,T) (G,T) (C,G) (A,T) (A,T) (A,T)
    Left segment  = "GGGCAAA" (positions 0-6).
    Right segment = "TTTGTTT" (positions 10-16, read left-to-right).
    n_wobble = 3 = floor(7/2) — at boundary; accepted.
    """
    target = "GGGCAAA" + "AAA" + "TTTGTTT"
    assert len(target) == 17
    result = find_stem_ind_wobble(target, stem_L=5)
    assert as_list(result) == [0, 6, 10, 16, 7]


# ---------------------------------------------------------------------
# Threshold (stem_L parameter)
# ---------------------------------------------------------------------

def test_stem_below_threshold_returns_zeros():
    """3-bp stem present but stem_L=5 → no detection."""
    # 3-bp WCF stem: AGC / GCT (rc of AGC).
    target = "AGCAAAGCT"
    assert as_list(find_stem_ind_wobble(target, stem_L=5)) == [0, 0, 0, 0, 0]


def test_stem_at_threshold_detected():
    """5-bp stem with stem_L=5 is detected."""
    target = "AGCAATTTTTGCT"  # length-5 stem
    result = find_stem_ind_wobble(target, stem_L=5)
    assert result[4] == 5


def test_lower_threshold_finds_smaller_stem():
    """Lowering stem_L exposes shorter stems."""
    target = "AGCAAAGCT"  # 3-bp WCF stem
    result = find_stem_ind_wobble(target, stem_L=3)
    assert as_list(result) == [0, 2, 6, 8, 3]
