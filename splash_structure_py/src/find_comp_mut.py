"""
This script is to find compensatory mutations in the target sequence, along with
all parameters that are crucial for calculating structure p-value.
The script also provides three ways for strucutre notation.
"""
import numpy as np
import pandas as pd

from splash_structure_py.src.non_wcf import V_EXT  # default valid set


def rc(seq):
    """
    Take in sequence and return the reverse complement of the given sequence.
    """
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(complement.get(base, base) for base in reversed(seq))


def find_mutation(base, target, stem_start_idx, stem_end_idx, rc_start_idx, rc_end_idx):
    """
    This functions takes in base target, target and hairpin structure indices,
    and return quanties that are crucial for calculating structure p-value, which include:
    1. totaMut: total number of mutations
    2. stemMut: number of mutations in the stem
    3. compMut: number of compensatory mutations
    4. struc: structure notation
    """
    struc = ['-'] * len(base)

    totaMut = 0
    stemMut = 0
    compMut = 0                      # pair of compensatory mutations
    sumInd = stem_start_idx + rc_end_idx # sum of compensatory stem indices 
    right_checked = []
    for i in range(len(target)):
        if base[i] != target[i]:
            totaMut += 1
            if i in range(stem_start_idx, stem_end_idx+1):          # i in left stem
                comInd = sumInd - i                           # calculate the corresponding index in the right stem 
                if base[comInd] != target[comInd]:            # if the corresponding index in the right stem is mutated
                    stemMut += 1
                    if rc(target[i]) == target[comInd]:       # if the right mutation is compensatory to the left mutation. 
                        compMut += 1
                        struc[i] = target[i]
                        struc[comInd] = target[comInd]
                        right_checked.append(comInd)
                    else:
                        struc[i] = target[i].lower()
                else:                                         # if the corresponding index in the right stem is not mutated
                    stemMut += 1
                    struc[i] = target[i].lower()
            elif i in range(rc_start_idx, rc_end_idx+1):        # i in right stem, already checked right-stem mutations
                stemMut += 1
                if i not in right_checked:
                    struc[i] = target[i].lower()
            else:                                             # i outside stem                   
                stemMut += 0
                struc[i] = target[i].lower()

    struc = (struc[:stem_start_idx] + 
             ['{'] + struc[stem_start_idx: stem_end_idx+1] + 
             ['('] + struc[stem_end_idx + 1: rc_start_idx] + [')'] + 
             struc[rc_start_idx: rc_end_idx+1] + ['}'] + 
             struc[rc_end_idx+1:])

    struc = "".join(struc)
    
    # add a patch for compactor segments: if stemL == 0 (stem_end_idx == 0, indeed)
    if stem_end_idx == 0:
        struc = np.nan

    return (totaMut, stemMut, compMut, struc)


def find_mutation_ext(base, target,
                      stem_start_idx, stem_end_idx,
                      rc_start_idx, rc_end_idx,
                      valid=V_EXT):
    """Extended version of ``find_mutation`` for the non-canonical
    extension. Recognises pairs in the valid set ``valid`` = V(N) as
    structure-supporting and counts both single-position-compatible
    (SPC) and base-pair-covariation (BPC) events. Default
    ``valid = V_EXT`` (V_WCF ∪ G·U) ⇒ byte-identical to the prior
    G·U-only behaviour.

    Parameters
    ----------
    base, target : str
        Equal-length sequences (cDNA alphabet).
    stem_start_idx, stem_end_idx : int
        Inclusive indices of the left stem in ``base``.
    rc_start_idx, rc_end_idx : int
        Inclusive indices of the right stem.

    Returns
    -------
    tuple
        ``(totaMut, stemMut, E, struc, n_p_vector, b_vector)``

        * ``totaMut`` — total Hamming distance v.
        * ``stemMut`` — mismatches falling in stem positions s = sum(n_p).
        * ``E`` — structure-supporting count: number of stem pairs that
          are SPC or BPC under V_EXT in ``target``.
        * ``struc`` — structure notation string. Conventions:
          ``-`` = no mutation OR untouched side of an SPC event;
          UPPERCASE = position is part of an SPC or BPC event
          (structure-supporting); lowercase = mutated, structure-
          disrupting, including positions outside the stem; braces
          ``{ ( ) }`` mark the stem-loop boundaries.
        * ``n_p_vector`` — list of length L with the per-pair mismatch
          count n_p ∈ {0, 1, 2}.
        * ``b_vector`` — list of length L with the base-target stem
          composition (b_L^p, b_R^p) tuples.

    Notes
    -----
    The original ``find_mutation`` is left untouched; this function is
    introduced behind the Phase 4 ``--wobble`` CLI flag.

    No-stem sentinel: compactor mode calls this function on each of the
    two recombined halves, including halves where the stem-finder
    returned ``(0, 0, 0, 0, 0)`` (no stem detected). In that case
    ``stem_end_idx == stem_start_idx == 0`` and a naive
    ``L = stem_end_idx - stem_start_idx + 1`` would equal 1, producing a
    phantom pair and contaminating ``b_vector`` / ``n_p_vector``. We
    short-circuit that case to return the outside-stem Hamming distance
    with empty per-pair vectors, matching the ``struc = np.nan``
    convention from the legacy ``find_mutation``.
    """
    if stem_end_idx == 0 and stem_start_idx == 0:
        totaMut = sum(1 for i in range(len(base)) if base[i] != target[i])
        return (totaMut, 0, 0, np.nan, [], [])

    L = stem_end_idx - stem_start_idx + 1
    struc = ['-'] * len(base)

    n_p_vector = []
    b_vector = []
    E = 0
    stemMut = 0

    # Walk the stem pair-by-pair: pair p (0-indexed) uses left index
    # stem_start_idx + p and right index rc_end_idx - p (reverse-
    # complement pairing).
    for p in range(L):
        left_idx = stem_start_idx + p
        right_idx = rc_end_idx - p

        b_L_p = base[left_idx]
        b_R_p = base[right_idx]
        b_vector.append((b_L_p, b_R_p))

        new_b_L = target[left_idx]
        new_b_R = target[right_idx]

        l_mut = new_b_L != b_L_p
        r_mut = new_b_R != b_R_p
        n_p = int(l_mut) + int(r_mut)
        n_p_vector.append(n_p)
        stemMut += n_p

        if n_p > 0 and (new_b_L, new_b_R) in valid:
            E_p = 1
        else:
            E_p = 0
        E += E_p

        if n_p == 1:
            mut_idx = left_idx if l_mut else right_idx
            mut_base = new_b_L if l_mut else new_b_R
            struc[mut_idx] = mut_base if E_p else mut_base.lower()
        elif n_p == 2:
            if E_p:
                struc[left_idx] = new_b_L
                struc[right_idx] = new_b_R
            else:
                struc[left_idx] = new_b_L.lower()
                struc[right_idx] = new_b_R.lower()
        # n_p == 0: leave both '-'

    # Mutations outside the stem.
    totaMut = stemMut
    for i in range(len(base)):
        if base[i] != target[i]:
            in_left_stem = stem_start_idx <= i <= stem_end_idx
            in_right_stem = rc_start_idx <= i <= rc_end_idx
            if not (in_left_stem or in_right_stem):
                totaMut += 1
                struc[i] = target[i].lower()

    # Insert stem-boundary braces.
    struc = (struc[:stem_start_idx] +
             ['{'] + struc[stem_start_idx:stem_end_idx + 1] +
             ['('] + struc[stem_end_idx + 1:rc_start_idx] + [')'] +
             struc[rc_start_idx:rc_end_idx + 1] + ['}'] +
             struc[rc_end_idx + 1:])
    struc = "".join(struc)

    # Match the original "no stem" patch (zero-length stem from compactor
    # mode signals nothing to score).
    if stem_end_idx == 0:
        struc = np.nan

    return (totaMut, stemMut, E, struc, n_p_vector, b_vector)


def db_notation_from_idx(stem_start_idx, stem_end_idx, rc_start_idx, rc_end_idx, target_length):
    db_notation=''
    for i in range(target_length):
        if (i >= stem_start_idx) & (i <= stem_end_idx):
            db_notation += '('
        elif (i >= rc_start_idx) & (i <= rc_end_idx):
            db_notation += ')'
        else:
            db_notation += '.'
    return db_notation

def db_notation_from_old_notaion(old_notation):
    if pd.isna(old_notation):
        return np.nan
    stem_start_idx = old_notation.find('{')
    stem_end_idx = old_notation.find('(')-2
    rc_start_idx = old_notation.find(')')-2
    rc_end_idx =  old_notation.find('}')-4
    return db_notation_from_idx(stem_start_idx, stem_end_idx, rc_start_idx, rc_end_idx, len(old_notation)-4)

def symbol_notation_from_old_notaion(old_notation, db_notation=None):
    if pd.isna(old_notation):
        return np.nan
    if db_notation is None:
        db_notation = db_notation_from_old_notaion(old_notation)
    rc_start_idx = db_notation.find(')')
    symbol_nototation = list(db_notation)
    old_notation = [item for item in old_notation if item not in ['{','}','(',')']]
    for i, item in enumerate(old_notation):
        if item.islower():
            symbol_nototation[i] = '*'
        elif item.isupper():
            if i < rc_start_idx:
                symbol_nototation[i] = '<'
            else:  
                symbol_nototation[i] = '>'
    return "".join(symbol_nototation)