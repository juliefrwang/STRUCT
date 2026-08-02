"""
This script is the first step of the pipeline. It takes in a dataframe with significant
anchors from SPLASH and returns a dataframe that contains hairpin structure information
and target weight information.
"""
import pandas as pd
import numpy as np
from pandarallel import pandarallel

from struct_rna.src.non_wcf import V_EXT, V_WCF


def rc(seq):
    """
    Take in sequence and return the reverse complement of the given sequence.
    """
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(complement.get(base, base) for base in reversed(seq))

def find_stem_ind(target, stem_L=5):
    """
    This fucntion find a hairpin structure in `target` sequence and returns stem indices.

    Input: 
    A target sequence and the minimum stem length (default value is 5).
    
    Output: 
    A list of 5 quantities, [stem_start_idx, stem_end_idx, rc_start_idx, rc_end_idx, stemL] if 
    a hairpin is found. Else, return [0,0,0,0,0]. 
    1. stem_start_idx: the start index of the stem in the target sequence.
    2. stem_end_idx: the end index of the stem in the target sequence.
    3. rc_start_idx: the start index of the reverse complement of the stem in the target sequence.
    4. rc_end_idx: the end index of the reverse complement of the stem in the target sequence.
    5. stemL: the length of the stem.

    Note: 
    Index starts from 0. The base at stem_start_idx and stem_end_idx are both included in the stem.
    Same for rc_start_idx and rc_end_idx. 
    """
    max_size = len(target) // 2
    for i in reversed(range(stem_L,max_size+1)):
        for j in range(len(target)-2*i+1):
            loc = target[j+i:].find(rc(target[j:j+i]))
            if loc > -1:
                stem_start_idx = j
                stem_end_idx = i + j-1
                rc_start_idx = loc + i + j
                rc_end_idx = loc + i + j + i-1
                return stem_start_idx, stem_end_idx, rc_start_idx, rc_end_idx, stem_end_idx-stem_start_idx+1
    return [0,0,0,0,0]


def find_stem_ind_wobble(target, stem_L=5, valid=V_EXT):
    """Non-canonical-aware stem detection.

    Searches for a stem-loop in ``target`` where each pair lies in the
    valid set ``valid`` = V(N) = V_WCF ∪ N (default ``V_EXT`` = V_WCF ∪
    G·U, preserving prior behaviour). Replaces the strict
    reverse-complement substring match in ``find_stem_ind`` with a
    position-by-position membership check.

    Selection order:

    1. Longest stem (length-greedy).
    2. Fewest non-canonical positions (tiebreaker among same-length
       stems; an all-WCF stem beats any N-using stem at the same length).
    3. Leftmost left-stem start.
    4. Leftmost right-stem start.

    Non-canonical cap: candidates with more than ``floor(L / 2)`` pairs
    drawn from ``N`` (= ``valid`` minus the WCF pairs), counted in
    aggregate over all non-canonical types, are rejected (decision D2).
    Bounds sequence-chance non-canonical-pile stems while allowing
    genuine extensions (e.g., G·U in HIV TAR's lower stem). If every
    candidate of length >= ``stem_L`` violates the cap, the no-stem
    sentinel is returned.

    Returns
    -------
    list or tuple
        ``(stem_start_idx, stem_end_idx, rc_start_idx, rc_end_idx, stemL)``
        if a stem of length >= ``stem_L`` and ``n_nc <= L // 2`` is
        found, else ``[0, 0, 0, 0, 0]``. Index conventions match
        ``find_stem_ind``.

    Notes
    -----
    With the default ``valid = V_EXT`` this is byte-identical to the
    prior G·U wobble finder (N = {G·U, U·G}, so ``n_nc`` counts exactly
    the wobble pairs). Backward compatibility with the strict-WCF finder
    holds only when no longer cap-satisfying N-using stem exists.
    """
    n = len(target)
    max_size = n // 2
    if max_size < stem_L:
        return [0, 0, 0, 0, 0]

    best = None  # (-length, n_nc, left_start, right_start, length)

    for i in range(stem_L, max_size + 1):
        # i = candidate stem length. Aggregate non-canonical cap.
        nc_cap = i // 2
        for j in range(n - 2 * i + 1):
            # j = left stem start
            for k in range(j + i, n - i + 1):
                # k = right stem start; pair p uses left j+p, right k+(i-1-p)
                n_nc = 0
                ok = True
                for p in range(i):
                    bL = target[j + p]
                    bR = target[k + (i - 1 - p)]
                    pair = (bL, bR)
                    if pair not in valid:
                        ok = False
                        break
                    if pair not in V_WCF:
                        # pair ∈ valid and ∉ V_WCF ⇒ a non-canonical (N) pair
                        n_nc += 1
                        if n_nc > nc_cap:
                            ok = False
                            break
                if ok:
                    candidate = (-i, n_nc, j, k, i)
                    if best is None or candidate < best:
                        best = candidate

    if best is None:
        return [0, 0, 0, 0, 0]

    neg_i, n_nc, j, k, i = best
    stem_start_idx = j
    stem_end_idx = j + i - 1
    rc_start_idx = k
    rc_end_idx = k + i - 1
    return stem_start_idx, stem_end_idx, rc_start_idx, rc_end_idx, i


def process_row(row):
    """
    This function takes in a dataframe row with found hairpin in the 
    most-frequent target and returns a dataframe with 11 columns.

    For each row (each significant anchor from SPLASH)):
    1. take rank-1 target as the 'base_target' and the rest as 'target'. 
    2. Calculate the weight of each target.

    Return 11 quantities: 
    1. anchor (same for all rows)
    2. M (number of occurences in data, same for all rows)
    3. stem_start_idx (same for all rows)
    4. stem_end_idx (same for all rows)
    5. rc_start_idx (same for all rows)
    6. rc_end_idx (same for all rows)
    7. stemL (same for all rows)
    8. base_target (same for all rows)
    9. target (different for each row)
    10. target_count (different for each row)
    11. target_wgt (different for each row): target_count / sum(up to top 10 target_counts)
    12. tar_wgt_filtered (different for each row): after abundance filtering, reweight targets, excluding base target
"""
    
    # obtain the target list and its counts. 
    # Filter index names by regex and drop '-' target.
    target_list = [row[tar] for tar in row.filter(regex=("^most_freq_target_")).index.to_list()[1:] if row[tar] != '-' ]
    cnt_list = np.array([row[cnt] for cnt in row.filter(regex=("^cnt_most_freq_target_")).index.to_list()[1:] if row[cnt] != 0])
    
    new_df = pd.DataFrame({"anchor": row["anchor"], 
                           "M": row["M"],
                           "stem_start_idx": row["stem_start_idx"], 
                           "stem_end_idx": row["stem_end_idx"], 
                           "rc_start_idx": row["rc_start_idx"], 
                           "rc_end_idx": row["rc_end_idx"],
                           "stemL": row["stemL"],
                           "base_target": row["most_freq_target_1"], 
                           "target": target_list, 
                           "target_count": cnt_list,
                           "target_wgt": np.array(cnt_list)/(sum(cnt_list) + row["cnt_most_freq_target_1"])})
    return new_df

def process_df(df, wgt_thres=0.05, stemL=5, wobble=False, valid=V_EXT):
    """
    This function takes in a dataframe with significant anchors from SPLASH
    and returns a dataframe that hairpin structure is found in the base target.
    Additionally, targets with weight w.r.t total occurences (M) < 0.05 are
    dropped and the weight is recalculated. Both filtered and unfiltered target
    weights are returned.

    Parameters
    ----------
    wobble : bool
        When True, use the non-canonical-aware stem finder
        ``find_stem_ind_wobble``; otherwise the strict-WCF
        ``find_stem_ind`` (default).
    valid : frozenset
        The valid pair set V(N) used by the non-canonical-aware finder.
        Default ``V_EXT`` (V_WCF ∪ G·U) ⇒ byte-identical to the prior
        wobble path. Ignored when ``wobble`` is False.
    """

    # find stems and store the index (both included)
    if wobble:
        df[["stem_start_idx", "stem_end_idx", "rc_start_idx", "rc_end_idx", "stemL"]] = pd.DataFrame(df.parallel_apply(lambda x: find_stem_ind_wobble(x.most_freq_target_1, stemL, valid), axis=1).tolist())
    else:
        df[["stem_start_idx", "stem_end_idx", "rc_start_idx", "rc_end_idx", "stemL"]] = pd.DataFrame(df.parallel_apply(lambda x: find_stem_ind(x.most_freq_target_1, stemL), axis=1).tolist())

    # drop anchors without stem using condition stem_start_idx == stem_end_idx
    df = df[df.stemL != 0]
    
    # exit program if no stem is found
    if len(df) == 0:
        # print("No structure is found in any target. Exiting..")
        return pd.DataFrame()

    # for each anchor, find base targets and targets
    df_processed = df.parallel_apply(process_row , axis=1)
    df = pd.concat(df_processed.to_list(), ignore_index=True)
    
    # filter target abundance >.05
    df = df.loc[df['target_wgt'] > .05].reset_index(drop=True)
    
    # recalculate target_weight (exclude cnts of base target)
    df["tar_wgt_filtered"] = df["target_count"] / df.groupby("anchor")["target_count"].transform("sum")
    
    return df

    
