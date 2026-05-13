import functools
import numpy as np
import pandas as pd
from math import comb
import itertools
import sys
from pandarallel import pandarallel

from splash_structure_py.src.non_wcf import pi_table

### 1. Target p computation ###
def target_p1_closed_form(k, v, L, c):
    """
    k: k-mer length
    v: total mutations
    L: stem length
    c: pair of compensatory mutations
    """
    p_c = 0
    for h in range(2*c, min(v, 2*L)+1):
        # print(f'h: {h}')
        p_h  = comb(2*L, h) * comb(k-2*L, v-h) / comb(k, v)
        p_c_h = 0
        for g in range(c, h//2 +1):
            p_g = 0
            for l in range(g, min(h+1, L+1)):
                l = max(l, h-l)
                if l > L:
                    continue
                sum_m = 0
                for m in range(0, min(l-g+1, h-l-g+1)):
                    sum_m += comb(l-g, m) * 2**m * comb(L-l, h-l-g-m) * 3**(h-l-g-m)
                p_g += comb(L, l) * 3**l * comb(l, g)* sum_m / comb(2*L, h) / 3**h
            p_c_h += p_g
        p_c += p_h * p_c_h
    return p_c

def target_p(k, stemL, totaMut, stemMut, compMut):
    """
    Return taregt p-value:
    p_1: exact p-val found using lookup table `dt` or approximate p for longer stem
    p_2: no stem mutations
    combine multiple p: (stemMut > 0) * p_1 + (stemMut == 0) * p_2
    """

    p_1 = target_p1_closed_form(k, totaMut, stemL, compMut)
    p_2 = comb(k - 2 * stemL, totaMut) / comb(k, totaMut)
    p = (stemMut > 0) * p_1 + (stemMut == 0) * p_2
    return p

### 1b. Extended (SVP) target p computation — non-WCF wobble extension ###

def target_p_svp(k, v, L, e, b, titv=0.5):
    """
    SVP (Stem-Variation-Presence) target p-value via the dynamic-programming
    algorithm in nonWCF_derivation.tex Section 5.4.

    Parameters
    ----------
    k : int
        Target length.
    v : int
        Total Hamming distance from base target.
    L : int
        Stem length (number of stem pairs).
    e : int
        Observed structure-supporting count (E = sum of E_p over all pairs).
    b : sequence of (str, str)
        Stem composition: list of (b_L^p, b_R^p) tuples, length L.
    titv : float, default 0.5
        Aggregate Ti/Tv event ratio assumed by the null (Section 5 of
        nonWCF_derivation.tex). ``0.5`` reproduces the uniform identity
        null; biological data typically sits near ``2``.

    Returns
    -------
    float
        Pr(E >= e | v, k, L, b) under the null model. Computed by
        marginalising the conditional Pr(E >= e | H = h, b) over the
        hypergeometric distribution of stem mismatches H.
    """
    if e <= 0:
        return 1.0
    if e > L:
        return 0.0
    if len(b) != L:
        raise ValueError(f"len(b) = {len(b)} but L = {L}")

    # Per-pair Bernoulli parameters pi_p^(j) for j in {1, 2}.
    pis = [pi_table(b_L, b_R, titv) for (b_L, b_R) in b]

    total = 0.0
    h_max = min(v, 2 * L)
    for h in range(h_max + 1):
        # Pr(H = h | v, k, L) — hypergeometric. comb returns 0 when args invalid.
        pr_h = comb(2 * L, h) * comb(k - 2 * L, v - h) / comb(k, v)
        if pr_h == 0:
            continue

        # Inner DP: g[s][m] over current p-slice.
        prev = [[0.0] * (L + 1) for _ in range(2 * L + 1)]
        prev[0][0] = 1.0

        for p in range(1, L + 1):
            curr = [[0.0] * (L + 1) for _ in range(2 * L + 1)]
            pi1 = pis[p - 1][1]
            pi2 = pis[p - 1][2]
            s_upper = min(2 * p, h)
            for s in range(s_upper + 1):
                m_upper = min(p, s + (p - 0))  # m <= p anyway
                for m in range(min(p, L) + 1):
                    val = 0.0
                    # j_p = 0: weight C(2,0) = 1, E_p = 0 deterministically
                    val += prev[s][m]
                    # j_p = 1: weight C(2,1) = 2; E_p = 1 with prob pi1
                    if s - 1 >= 0:
                        if m - 1 >= 0:
                            val += 2.0 * pi1 * prev[s - 1][m - 1]
                        val += 2.0 * (1.0 - pi1) * prev[s - 1][m]
                    # j_p = 2: weight C(2,2) = 1; E_p = 1 with prob pi2
                    if s - 2 >= 0:
                        if m - 1 >= 0:
                            val += pi2 * prev[s - 2][m - 1]
                        val += (1.0 - pi2) * prev[s - 2][m]
                    curr[s][m] = val
            prev = curr

        # Pr(E >= e | H = h, b) = sum_{m=e}^L g_L[h][m] / C(2L, h)
        denom = comb(2 * L, h)
        if denom == 0:
            continue
        inner = sum(prev[h][m] for m in range(e, L + 1)) / denom
        total += pr_h * inner

    return total


def target_p_ext(k, stemL, totaMut, stemMut, e, b, titv=0.5):
    """
    Extended target p-value combining SVE (s = 0) and SVP (s > 0)
    via an indicator on stemMut, per Section 6 of nonWCF_derivation.tex.

    SVE branch is the (corrected) hypergeometric C(k-2L, v) / C(k, v),
    identity-free and therefore independent of ``titv``. SVP branch calls
    ``target_p_svp`` with the supplied ``titv``.

    Parameters
    ----------
    k : int
        Target length.
    stemL : int
        Stem length L.
    totaMut : int
        Total Hamming distance v.
    stemMut : int
        Number of mismatches falling in stem positions (drives the indicator).
    e : int
        Observed structure-supporting count E (only used when stemMut > 0).
    b : sequence of (str, str)
        Stem composition; length stemL.
    titv : float, default 0.5
        Aggregate Ti/Tv event ratio assumed by the null.

    Returns
    -------
    float
        Indicator-combined extended target p-value.
    """
    if stemMut == 0:
        # SVE: Pr(all v mismatches outside stem). Identity-free, so titv
        # does not enter.
        return comb(k - 2 * stemL, totaMut) / comb(k, totaMut)
    return target_p_svp(k, totaMut, stemL, e, b, titv)


def _g_L_table(k, v, L, b, titv=0.5):
    """Run the L-pair DP from nonWCF_derivation.tex section 5.4 and
    return the final unnormalised g_L[h][m] table.

    Pr(E = m | H = h, b) = g_L[h][m] / C(2L, h).
    """
    pis = [pi_table(b_L, b_R, titv) for (b_L, b_R) in b]
    h_max = min(v, 2 * L)

    prev = [[0.0] * (L + 1) for _ in range(2 * L + 1)]
    prev[0][0] = 1.0
    for p in range(1, L + 1):
        curr = [[0.0] * (L + 1) for _ in range(2 * L + 1)]
        pi1 = pis[p - 1][1]
        pi2 = pis[p - 1][2]
        s_upper = min(2 * p, h_max)
        for s in range(s_upper + 1):
            for m in range(min(p, L) + 1):
                val = 0.0
                # j_p = 0: weight C(2,0) = 1, E_p = 0 deterministically
                val += prev[s][m]
                # j_p = 1: weight C(2,1) = 2; E_p = 1 with prob pi1
                if s - 1 >= 0:
                    if m - 1 >= 0:
                        val += 2.0 * pi1 * prev[s - 1][m - 1]
                    val += 2.0 * (1.0 - pi1) * prev[s - 1][m]
                # j_p = 2: weight C(2,2) = 1; E_p = 1 with prob pi2
                if s - 2 >= 0:
                    if m - 1 >= 0:
                        val += pi2 * prev[s - 2][m - 1]
                    val += (1.0 - pi2) * prev[s - 2][m]
                curr[s][m] = val
        prev = curr
    return prev


def target_p_marginal_ext(k, stemL, totaMut, b, titv=0.5):
    """Exact marginal PMF of target_p_ext under H_0.

    Returns ``(support, pmf)`` where ``support`` is the sorted (ascending)
    list of distinct target_p_ext values and ``pmf[i] = Pr_H0(p_tar = support[i])``.

    Computed by aggregating the joint H_0 probabilities of (H, E) over the
    indicator-switch rule:
      - h = 0: target_p_ext = q_0 = C(k - 2L, v) / C(k, v).
      - h >= 1: target_p_ext = target_p_svp(k, v, L, e, b), depending on the
        observed e.
    For each (h, e) configuration with positive mass, the joint probability
    Pr(H = h | v, k, L) * Pr(E = e | H = h, b) is added to the bin keyed by
    target_p_ext(h, e).

    Replaces the differences-of-sorted-support construction in the previous
    target_p_outcome_ext + prep_for_conv_ext path, which treated the
    indicator-switch composite as if it were discrete-uniform on its CDF
    and was anti-conservative by up to q_0 at the SVE/SVP boundary
    (nonWCF_derivation.tex section 6.A).
    """
    if len(b) != stemL:
        raise ValueError(f"len(b) = {len(b)} but stemL = {stemL}")

    L = stemL
    v = totaMut
    h_max = min(v, 2 * L)

    # Run the DP once to get the joint Pr(H = h, E = m | b) up to the
    # multivariate-hypergeometric normalisation.
    g_L = _g_L_table(k, v, L, b, titv)

    # Pre-compute target_p_svp(e) for each e in 0..L. Calls cache the
    # marginal-over-h SVP p-value used by the indicator switch when h >= 1.
    p_svp_for_e = [target_p_svp(k, v, L, e, b, titv) for e in range(L + 1)]
    q0 = comb(k - 2 * L, v) / comb(k, v) if comb(k, v) > 0 else 0.0

    pmf_dict: dict[float, float] = {}
    denom_kv = comb(k, v)
    for h in range(h_max + 1):
        c2Lh = comb(2 * L, h)
        if c2Lh == 0:
            continue
        c_outside = comb(k - 2 * L, v - h)
        if c_outside == 0:
            continue
        pr_h = c2Lh * c_outside / denom_kv  # hypergeometric Pr(H = h | v)
        if pr_h == 0.0:
            continue

        for e in range(L + 1):
            weight = g_L[h][e]
            if weight == 0.0:
                continue
            pr_e_given_h = weight / c2Lh
            joint = pr_h * pr_e_given_h

            if h == 0:
                p_value = q0
            else:
                p_value = p_svp_for_e[e]

            # Round to bin same numerical p-values together. 12 digits is
            # more than enough for the discrete support set.
            key = round(float(p_value), 12)
            pmf_dict[key] = pmf_dict.get(key, 0.0) + joint

    support = sorted(pmf_dict.keys())
    pmf = [pmf_dict[p] for p in support]
    return support, pmf


def target_p_outcome_ext(k, stemL, totaMut, b, titv=0.5):
    """Backward-compatible wrapper returning only the sorted support of
    target_p_ext under H_0. Prefer ``target_p_marginal_ext`` when the
    paired PMF is needed."""
    support, _ = target_p_marginal_ext(k, stemL, totaMut, b, titv)
    return support


def prep_for_conv_ext(num_target, wgt_all, k, stemL_list, totaMut_list, b_list, titv=0.5):
    """SVP analogue of ``prep_for_conv``.

    Per-target marginal PMF is computed exactly via target_p_marginal_ext
    (aggregation over joint H_0 probabilities of (H, E)), not by the
    differences-of-sorted-support surrogate. Caps targets at 4 to match
    the original.
    """
    if num_target > 4:
        wgt_all = [w / sum(wgt_all[0:4]) for w in wgt_all[0:4]]
        stemL_list = stemL_list[0:4]
        totaMut_list = totaMut_list[0:4]
        b_list = b_list[0:4]
        num_target = 4

    wgted_target_outcomes = []
    target_pmf = []

    for i in range(num_target):
        support, pmf = target_p_marginal_ext(
            k, stemL_list[i], totaMut_list[i], b_list[i], titv
        )
        target_pmf.append(pmf)
        wgted_target_outcomes.append([wgt_all[i] * p for p in support])
    return wgted_target_outcomes, target_pmf

### 2. Anchor p computation ###
def target_p_outcome(k, stemL, totaMut):
    """
    Step 1: calculate each outcome of target_p
    target_p is computed on condition of k, stemL, totaMut

    Output: 
    all_possible_outcome: list of all possible outcomes of target_p
    """
    all_possible_outcome = set()
    stemMut_start = 0 if totaMut - (k - 2 * stemL) < 0 else totaMut - (k - 2 * stemL)
    for stemMut in range(stemMut_start, min(totaMut, 2*stemL)+1):
        for compMut in range((stemMut+2)//2):
            all_possible_outcome.add(target_p(k, stemL, totaMut, stemMut, compMut))
    all_possible_outcome = list(all_possible_outcome)
    all_possible_outcome.sort()
    return all_possible_outcome

def prep_for_conv(num_target, wgt_all, k, stemL_list, totaMut_list):
    """
    Step 2: prep for convolution: calculate PMF of each outcome of target_p
    
    Output: 
    wgted_target_outcomes: nested lists of (weighted) target_p for all targets of an anchor
    target_pmf: 
    """
    if num_target > 4: # cap number of targets for convolution to 4
        wgt_all = [i / sum(wgt_all[0:4]) for i in wgt_all[0:4]]
        stemL_list = stemL_list[0:4]
        totaMut_list = totaMut_list[0:4]
        num_target = 4
        
    wgted_target_outcomes = []
    target_pmf = [] 
    
    for i in range(num_target):
        targetp = target_p_outcome(k, stemL_list[i], totaMut_list[i])
        pmf = [targetp[0]]+[targetp[i+1] - targetp[i] for i in range(len(targetp)-1)]

        target_pmf.append(pmf)
        wgted_target_outcomes.append([wgt_all[i] * j for j in targetp])
    return wgted_target_outcomes, target_pmf

def pmf_anchor_score(wgted_target_outcomes, target_pmf):
    """
    Step 3: calculate the probability of each outcome of 
    
    Output: 
    anchor_p = weighted_average(target_p, anchor_p)
    """
    # all outcomes
    all_anchor_outcomes = [sum(x) for x in itertools.product(*wgted_target_outcomes)] 
    # all pmf
    anchor_pmf = [np.prod(x) for x in itertools.product(*target_pmf)] 
    return all_anchor_outcomes, anchor_pmf

def anchor_p(all_anchor_outcomes, anchor_pmf, anchor_p):
    """
    Step 4: calculate the CDF of anchor_p and find p-value of anchor_p
    """
    df = pd.DataFrame({'outcome':all_anchor_outcomes, 'prob':anchor_pmf})
    df=df.sort_values(by=['outcome']).reset_index(drop=True)
    df['cdf'] = df['prob'].cumsum()
    p_val = df[df.outcome <= anchor_p + 1e-6]['cdf'].max()
    if p_val is np.nan:
        p_val = df.iloc[0]['cdf']
    return p_val
    
def anchor_p_target_subdf(sub_df):
    """
    Step 5 (1): wrap all functions for one anchor and apply to sub-dataframe for stucture-target
    """
    p_val = sub_df['anchor_score'].iloc[0]

    if len(sub_df) > 1:
        all_anchor_outcomes, anchor_pmf = pmf_anchor_score(*prep_for_conv(len(sub_df),\
                                                      list(sub_df['tar_wgt_filtered']), \
                                                      len(sub_df['base_target'].iloc[0]), \
                                                      list(sub_df['stemL']), \
                                                      list(sub_df['totaMut'])))

        p_val = anchor_p(all_anchor_outcomes, anchor_pmf, p_val)

    return p_val


def anchor_p_target_subdf_ext(sub_df, titv=0.5):
    """SVP analogue of ``anchor_p_target_subdf``.

    Same shape, but uses the b_vector column (stem composition per target)
    and ``prep_for_conv_ext``. All targets within an anchor share the same
    base target, so b_vector is identical across rows; we still pass the
    full list to keep the API symmetric.
    """
    p_val = sub_df['anchor_score'].iloc[0]

    if len(sub_df) > 1:
        all_anchor_outcomes, anchor_pmf = pmf_anchor_score(*prep_for_conv_ext(
            len(sub_df),
            list(sub_df['tar_wgt_filtered']),
            len(sub_df['base_target'].iloc[0]),
            list(sub_df['stemL']),
            list(sub_df['totaMut']),
            list(sub_df['b_vector']),
            titv,
        ))

        p_val = anchor_p(all_anchor_outcomes, anchor_pmf, p_val)

    return p_val

def anchor_p_compactor_subdf(sub_df):
    """
    Step 5 (2): wrap all functions for one anchor-split and apply to each anchor for stucture-compactor
    """
    # for compactors, we compute anchor-score for each split
    p_val = sub_df['anchor_score_per_split'].iloc[0]
    
    if len(sub_df) > 1:
        # structure evaluation length for compactor is 80 (HARDCODED)
        all_anchor_outcomes, anchor_pmf = pmf_anchor_score(*prep_for_conv(len(sub_df),\
                                                      list(sub_df['compactor_weight']), \
                                                      80, \
                                                      list(sub_df['stemL']), \
                                                      list(sub_df['totaMut'])))
        p_val = anchor_p(all_anchor_outcomes, anchor_pmf, p_val)
        
    return p_val

def wrap_anchor_p_target(df):
    """
    Step 6 (1): wrap all functions for one anchor and apply to the whole dataframe for stucture-target
    """
    grouped = df.groupby('anchor') # create a groupby object based on 'anchor'
    p_val_results = grouped.parallel_apply(anchor_p_target_subdf) # Apply function to each group
    # The result is a Series where the index is the group keys ('anchor' values)
    # We can now assign this back to your DataFrame, but you'll need to align the indices
    df = df.merge(p_val_results.rename('anchor_p'), left_on='anchor', right_index=True)
    return df


def wrap_anchor_p_target_ext(df, titv=0.5):
    """SVP analogue of ``wrap_anchor_p_target``.

    Requires the input dataframe to carry a ``b_vector`` column (added by
    ``find_mutation_ext`` in Phase 2). ``titv`` is bound into the per-group
    callable via ``functools.partial`` so it travels through pandarallel's
    serialisation unchanged.
    """
    grouped = df.groupby('anchor')
    p_val_results = grouped.parallel_apply(
        functools.partial(anchor_p_target_subdf_ext, titv=titv)
    )
    df = df.merge(p_val_results.rename('anchor_p'), left_on='anchor', right_index=True)
    return df

def wrap_anchor_p_compactor(df):
    """
    Step 6 (1): wrap all functions for one anchor and apply to the whole dataframe for stucture-compactor
    """
    grouped = df.groupby('anchor_split') # create a groupby object based on 'anchor_split'
    p_val_results = grouped.parallel_apply(anchor_p_compactor_subdf) # Apply function to each group
    # The result is a Series where the index is the group keys ('anchor_split' values)
    # We can now assign this back to your DataFrame, but you'll need to align the indices
    df = df.merge(p_val_results.rename('anchor_p'), left_on='anchor_split', right_index=True)
    return df