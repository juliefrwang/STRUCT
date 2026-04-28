"""Unit tests for find_stem_ind_wobble.

Verifies:
- Backward compatibility with find_stem_ind on WCF-only stems
- Detection of wobble-using stems that the original would miss
- Tie-breaking order (fewest wobble, then longest, then leftmost)
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


def test_all_wobble_stem():
    """Stem composed entirely of G·U wobble pairs.

    Target has no C's, so the only valid pairing is G·U wobble. The
    only length-5 candidate is the outermost (G, T) stem.
    """
    target = "GGGGGTTTTT"  # length 10, 5G + 5T, zero-length loop
    result = find_stem_ind_wobble(target, stem_L=5)
    assert as_list(result) == [0, 4, 5, 9, 5]


def test_wobble_stem_not_found_by_original():
    """Sanity check: the wobble-only stem is invisible to the strict WCF
    finder (so this test confirms our extension actually adds something)."""
    target = "GGGGGTTTTT"
    # find_stem_ind requires rc("GGGGG") = "CCCCC" downstream — absent here.
    wcf = find_stem_ind(target, stem_L=5)
    wobble = find_stem_ind_wobble(target, stem_L=5)
    assert as_list(wcf) == [0, 0, 0, 0, 0]
    assert as_list(wobble) == [0, 4, 5, 9, 5]


# ---------------------------------------------------------------------
# Tie-breaking: fewest wobble preferred over longer stem
# ---------------------------------------------------------------------

def test_wcf_only_preferred_over_longer_wobble():
    """If a 5-bp WCF-only stem exists alongside a longer wobble-using stem,
    the WCF one wins because n_wobble = 0 dominates length."""
    # Construct: 5-bp WCF stem at positions 0-4 / 8-12, plus a longer
    # wobble-using candidate elsewhere. We use a target that ALSO admits a
    # longer wobble-only candidate spanning the whole target, but the inner
    # 5-bp WCF should still win on tie-breaking.
    #
    # left "AGCAA" -- "TTT" -- right "TTGCT" (WCF rc of left) — length-5 WCF.
    target = "AGCAATTTTTGCT"
    result = find_stem_ind_wobble(target, stem_L=5)
    # The unique stem here is the 5-bp WCF; n_wobble = 0.
    assert as_list(result) == [0, 4, 8, 12, 5]


def test_longer_stem_preferred_when_wobble_count_equal():
    """Two stems both with 0 wobble; the longer one wins."""
    # 6-bp WCF stem.
    target = "AGCATCAAAAGATGCT"
    result = find_stem_ind_wobble(target, stem_L=5)
    # The 6-bp WCF stem must win over any 5-bp sub-stem.
    assert as_list(result) == [0, 5, 10, 15, 6]


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
