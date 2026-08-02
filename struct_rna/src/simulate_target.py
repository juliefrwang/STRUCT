import random
import sys
import gc
sys.path.append('/oak/stanford/groups/horence/juliew/structure/src')
import find_comp_mut
import get_pval

def random_mutate(base_target, totaMul):
    """
    Given the base_target and total mutations (hamming distance),
    randomly select totaMul positions and mutate with 3 possibilities.
    """
    mutations = {'A':['C', 'G', 'T'],
                  'C':['A', 'G', 'T'],
                  'G':['A', 'C', 'T'],
                  'T':['A', 'C', 'G']} # possible mutations for each base
    base_list = list(base_target)
    mutate_ind = random.sample(range(len(base_target)), k=totaMul) # select totaMul random locations to mutate
    for i in mutate_ind:
        cur_base = base_list[i]
        base_list[i] = random.choice(mutations[cur_base])

    return  "".join(base_list) # back to string

def get_simulated_p(anchor, df, n_iter,):
    sub_df = df.loc[df.anchor == anchor].reset_index(drop=True)
    base_target = sub_df.loc[0, 'base_target']
    stem_start_idx, stem_end_idx, rc_start_idx, rc_end_idx, stemL = sub_df.loc[0, ['stem_start_idx', 'stem_end_idx', 'rc_start_idx', 'rc_end_idx', 'stemL']]
    
    anchor_score=0
    anchor_p_multi_simu = []
    
    for n in range(n_iter):
        anchor_score = 0
        for i in range(len(sub_df)):
            simu_target = random_mutate(base_target, sub_df.totaMut.iloc[i])
            totaMut, stemMut, compMut, _ = find_comp_mut.find_mutation(base_target, simu_target, stem_start_idx, stem_end_idx, rc_start_idx, rc_end_idx)
            target_p = get_pval.target_p(len(base_target), stemL, totaMut, stemMut, compMut)
            anchor_score += target_p * sub_df.tar_wgt_filtered.iloc[i]

        p_val_one_simu = anchor_score
        if len(sub_df) > 1:
            all_anchor_outcomes, anchor_pmf = get_pval.pmf_anchor_score(*get_pval.prep_for_conv(len(sub_df), \
                                                list(sub_df['tar_wgt_filtered']), \
                                                len(sub_df['base_target'].iloc[0]), \
                                                list(sub_df['stemL']), \
                                                list(sub_df['totaMut'])))
            p_val_one_simu = get_pval.anchor_p(all_anchor_outcomes, anchor_pmf, p_val_one_simu) 

        anchor_p_multi_simu.append(p_val_one_simu)
#    f = open("/oak/stanford/groups/horence/juliew/structure/0204_results/Batson_results/simulation_p.txt", "a")
#    f.write(f"{anchor}\n{anchor_p_multi_simu}\n")
#    f.close()

    # delete sub_df from memory
    del sub_df
    gc.collect()
    return tuple(anchor_p_multi_simu)


def get_simulated_p_compactor(anchor_split, df, n_iter):
    sub_df = df.loc[df.anchor_split == anchor_split].reset_index(drop=True) ###TODO: check if this is correct, should be anchor-split

    base_target_1 = sub_df.loc[0, 'base_S1']
    stem_start_idx_1, stem_end_idx_1, rc_start_idx_1, rc_end_idx_1, stemL_1 = sub_df.loc[0, ['stem_start_idx_1', 'stem_end_idx_1', 'rc_start_idx_1', 'rc_end_idx_1', 'stemL_1']]
    
    base_target_2 = sub_df.loc[0, 'base_S2']
    stem_start_idx_2, stem_end_idx_2, rc_start_idx_2, rc_end_idx_2, stemL_2 = sub_df.loc[0, ['stem_start_idx_2', 'stem_end_idx_2', 'rc_start_idx_2', 'rc_end_idx_2', 'stemL_2']]
   
    
    anchor_p_multi_simu = 0
    
    for n in range(n_iter):
        anchor_score_per_split=0
        anchor_p_one_simu = 0   
        for i in range(len(sub_df)):
            simu_target_1 = random_mutate(base_target_1, sub_df.totaMut_1.iloc[i])
            totaMut_1, stemMut_1, compMut_1, _ = find_comp_mut.find_mutation(base_target_1, simu_target_1, stem_start_idx_1, stem_end_idx_1, rc_start_idx_1, rc_end_idx_1)
            simu_target_1 = random_mutate(base_target_1, sub_df.totaMut_1.iloc[i])
            totaMut_2, stemMut_2, compMut_2, _ = find_comp_mut.find_mutation(base_target_2, simu_target_2, stem_start_idx_2, stem_end_idx_2, rc_start_idx_2, rc_end_idx_2)
            simu_target_2 = random_mutate(base_target_2, sub_df.totaMut_2.iloc[i])

            target_p = get_pval.target_p(80, stemL_1+stemL_2, totaMut_1+totaMut_2, stemMut_1+stemMut_2, compMut_1+compMut_2)

            anchor_score_per_split += target_p * sub_df.target_weight.iloc[i]

        p_val_one_simu = anchor_score_per_split
        if len(sub_df) > 1:
            all_anchor_outcomes, anchor_pmf = get_pval.pmf_anchor_score(*get_pval.prep_for_conv(len(sub_df), \
                                                list(sub_df['target_weight']), \
                                                80, \
                                                list(sub_df['stemL']), \
                                                list(sub_df['totaMut'])))
            p_val_one_simu = get_pval.anchor_p(all_anchor_outcomes, anchor_pmf, p_val_one_simu)

        anchor_p_multi_simu += p_val_one_simu/n_iter

    # delete sub_df from memory
    del sub_df
    gc.collect()
    return anchor_p_multi_simu
