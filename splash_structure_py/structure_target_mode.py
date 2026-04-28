"""
This script is the main pipeline that takes in SPLASH significant anchors output, process
targets, calculate structure anchor-p, and save results.
Two arguments are required:
    1. SPLASH significant anchors output
    2. output folder name (does not need to exist)
    3. optional: -a to run element annotation on extendors
"""
import sys
import os
import argparse
import pandas as pd
from statsmodels.stats.multitest import multipletests
from pandarallel import pandarallel

from splash_structure_py.src.parse_args import *
from splash_structure_py.src.process_targets import *
import splash_structure_py.src.find_comp_mut as find_comp_mut
import splash_structure_py.src.get_pval as get_pval
import splash_structure_py.src.elem_annas as elem_annas

def SS_target(output_prefix, splash_output_file, element_annotation, wobble=False):

    """ Step 0: Preparation """
    # Initialize parallelization. Create folder to save results
    pandarallel.initialize()
    outfolder = f'{output_prefix}_results'
    os.makedirs(outfolder, exist_ok=True)

    """ Step 1: Read in the input file and process dataframe to get base targets and targets """
    df = pd.read_csv(splash_output_file, sep = '\t')
    df = process_df(df, wobble=wobble)
    # exit program if no structure is found in any target
    if len(df) == 0:
        print("No structure is found for any anchor. Exiting..")
        return

    """ Step 2: Find parameters that are to be used in anchor-p computation, along with three types of notations"""
    if wobble:
        # Extended path: SPC + BPC events under V_EXT, returns extra
        # n_p_vector and b_vector columns consumed by target_p_ext.
        df[["totaMut", "stemMut", "E", "strucNotation", "n_p_vector", "b_vector"]] = pd.DataFrame(
            df.parallel_apply(lambda x: find_comp_mut.find_mutation_ext(
                x.base_target, x.target,
                x.stem_start_idx, x.stem_end_idx,
                x.rc_start_idx, x.rc_end_idx), axis=1).tolist()
        )
    else:
        df[["totaMut", "stemMut", "compMut", "strucNotation"]] = pd.DataFrame(df.parallel_apply(lambda x: find_comp_mut.find_mutation(x.base_target, \
                                                    x.target, x.stem_start_idx, x.stem_end_idx, x.rc_start_idx, x.rc_end_idx), axis=1).tolist())
    # dot bracket notation
    df["db_strucNotation"] = df.parallel_apply(lambda x: find_comp_mut.db_notation_from_old_notaion(x.strucNotation), axis=1)
    # symbol notation
    df["symbol_strucNotation"]= df.parallel_apply(lambda x: find_comp_mut.symbol_notation_from_old_notaion(x.strucNotation, x.db_strucNotation), axis=1)

    """ Step 3: Calculate structure target-p """
    # filter out anchors with number of target <= 2
    df['num_target'] = df.groupby('anchor')['target'].transform('count')
    df = df.loc[df.num_target > 2].reset_index(drop=True)
    # target_p
    if wobble:
        df["target_p"] = df.parallel_apply(lambda x: get_pval.target_p_ext(
            len(x['base_target']), x['stemL'], x['totaMut'], x['stemMut'],
            x['E'], x['b_vector']), axis=1)
    else:
        df["target_p"] = df.parallel_apply(lambda x: get_pval.target_p(len(x['base_target']), \
                                           x['stemL'], x['totaMut'], x['stemMut'], x['compMut']), axis=1)

    """ Step 4: Calculate anchor_score """
    df["anchor_score"] = df["tar_wgt_filtered"] * df["target_p"]
    df["anchor_score"] = df.groupby(["anchor"])["anchor_score"].transform("sum")

    """ Step 5: Calculate anchor_p """
    if wobble:
        df = get_pval.wrap_anchor_p_target_ext(df)
    else:
        df = get_pval.wrap_anchor_p_target(df)

    """ Step 6: BH correction on anchors with number of target > 2 """
    df_temp = df[['anchor', 'anchor_p']].drop_duplicates()
    correction = multipletests(df_temp['anchor_p'], alpha=0.05, method='fdr_bh')
    df_temp['anchor_p_BH'] = correction[1]
    df_temp = df_temp.drop(columns=['anchor_p']) 
    df = df.merge(df_temp, on='anchor', how = 'left') # Merge back
    df = df.sort_values(by=['anchor_p_BH', 'anchor'], ascending=True).reset_index(drop=True) #sort

    """ Step 7: Save """
    df.to_csv(f'{outfolder}/structure_on_targets.tsv', index=False, sep='\t')

    if element_annotation:
        """ Step 8: elememt annotations (optional, toggle on by -a) """
        # run element annotations
        elem_ann_folder = elem_annas.run_anns(outfolder, "extendor")

        """ Step 9: merge structure results with element annotations """
        anns_file = f"{outfolder}/elem_anns/{elem_ann_folder}_work/{elem_ann_folder}/element_annotations/element_annotations_anchors.tsv"
        df_anns = pd.read_csv(anns_file, sep='\t')
        df = elem_annas.merge_anns_struc(df_anns, df)
        df.to_csv(f'{outfolder}/structure_on_targets.tsv', index=False, sep='\t')

def run_SS_target():
    arguments = argument_parser_target()
    SS_target(**arguments)

if __name__ == "__main__":
    arguments = argument_parser_target()
    SS_target(**arguments)

