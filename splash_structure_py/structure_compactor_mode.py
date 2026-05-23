"""
The script takes in two arguments:
1. path to compactor files
2. working folder name
3. optional: -a to run element annotation on compactors
"""
import sys
import os
import argparse
import pandas as pd
import subprocess
from statsmodels.stats.multitest import multipletests
from pandarallel import pandarallel

from splash_structure_py.src.parse_args import *
from splash_structure_py.src.process_targets import *
from splash_structure_py.src.non_wcf import build_valid_set
import splash_structure_py.src.find_comp_mut as find_comp_mut
import splash_structure_py.src.get_pval as get_pval
import splash_structure_py.src.elem_annas as elem_annas


def SS_compactor(output_prefix, compactor_file, element_annotation,
                 wobble=False, titv=0.5, noncanon=None):
    """ Step 0: Preparation """
    # Resolve the non-canonical specification (gate folded into
    # `noncanon`, mirroring SS_target). Backward compat: if `noncanon`
    # is not given explicitly, derive it from the legacy `wobble` bool.
    #   noncanon given          -> use it (gate + parameterize)
    #   noncanon None, wobble=T -> "GU"  (== old --wobble behaviour)
    #   noncanon None, wobble=F -> "none" (legacy WCF-only path)
    if noncanon is None:
        noncanon = "GU" if wobble else "none"
    extended = str(noncanon).strip().lower() not in {"none", ""}
    valid = build_valid_set(noncanon) if extended else None

    # Initialize parallelization. Create folder to save results
    pandarallel.initialize()
    outfolder = f'{output_prefix}_results'
    os.makedirs(outfolder, exist_ok=True)
    os.makedirs(f'{outfolder}/interm_compactor', exist_ok=True)

    """ Step 1: Read in compactors and process dataframe using Julia script """
    # Define the command to run the Julia script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    julia_script_path = os.path.join(current_dir, 'src', 'process_compactor_4_segments.jl')
    julia_command = f"julia {julia_script_path} {compactor_file} {outfolder}/interm_compactor"

    # Run the Julia script using subprocess and wait for it to finish
    completed_process = subprocess.run(julia_command, shell=True)

    """ Step 2: Read in processed compactors & add number of compactors """
    # Check if the Julia script ran successfully (exit code 0)
    if completed_process.returncode == 0:
        df = pd.read_csv(f"{outfolder}/interm_compactor/processed_compactors.tsv", sep = '\t')
    else:
        print("Julia script encountered an error or did not finish successfully.")
        sys.exit(1)

    # exit program if no compactor is left after abundance filtering
    if len(df) == 0:
        print("No structure is found for any anchor. Exiting..")
        return

    """
    Step 3: Find parameters that are to be used in anchor-p computation,
    along with three types of notations
    """
    # find stem loop index (wobble-aware when extended)
    if extended:
        df[["stem_start_idx_1", "stem_end_idx_1", "rc_start_idx_1", "rc_end_idx_1", "stemL_1"]] = pd.DataFrame(df.parallel_apply(lambda x: find_stem_ind_wobble(x.base_S1, 5, valid), axis=1).tolist())
        df[["stem_start_idx_2", "stem_end_idx_2", "rc_start_idx_2", "rc_end_idx_2", "stemL_2"]] = pd.DataFrame(df.parallel_apply(lambda x: find_stem_ind_wobble(x.base_S2, 5, valid), axis=1).tolist())
    else:
        df[["stem_start_idx_1", "stem_end_idx_1", "rc_start_idx_1", "rc_end_idx_1", "stemL_1"]] = pd.DataFrame(df.parallel_apply(lambda x: find_stem_ind(x.base_S1, 5), axis=1).tolist())
        df[["stem_start_idx_2", "stem_end_idx_2", "rc_start_idx_2", "rc_end_idx_2", "stemL_2"]] = pd.DataFrame(df.parallel_apply(lambda x: find_stem_ind(x.base_S2, 5), axis=1).tolist())

    # Add a column of number of stem-loop structure found in the compactors
    df['num_stem_loop'] = (df['stemL_1'] != 0).astype(int) + (df['stemL_2'] != 0).astype(int)

    # drop anchors without stem using condition num_stem_loop == 0
    df = df[df.num_stem_loop != 0].reset_index(drop = True)

    # exit program if no structure is found in any target
    if len(df) == 0:
        print("No structure is found for any anchor. Exiting...")
        return

    # find mutations in stem & addition two columns for structure notations
    if extended:
        df[["totaMut_1", "stemMut_1", "E_1", "strucNotation_1", "n_p_vector_1", "b_vector_1"]] = pd.DataFrame(df.parallel_apply(lambda x: find_comp_mut.find_mutation_ext(x.base_S1, \
                                                        x.S1, x.stem_start_idx_1, x.stem_end_idx_1, x.rc_start_idx_1, x.rc_end_idx_1, valid), axis=1).tolist())
    else:
        df[["totaMut_1", "stemMut_1", "compMut_1", "strucNotation_1"]] = pd.DataFrame(df.parallel_apply(lambda x: find_comp_mut.find_mutation(x.base_S1, \
                                                        x.S1, x.stem_start_idx_1, x.stem_end_idx_1, x.rc_start_idx_1, x.rc_end_idx_1), axis=1).tolist())
    df["db_strucNotation_1"] = df.parallel_apply(lambda x: find_comp_mut.db_notation_from_old_notaion(x.strucNotation_1), axis=1)
    df["symbol_strucNotation_1"]= df.parallel_apply(lambda x: find_comp_mut.symbol_notation_from_old_notaion(x.strucNotation_1, x.db_strucNotation_1), axis=1)

    if extended:
        df[["totaMut_2", "stemMut_2", "E_2", "strucNotation_2", "n_p_vector_2", "b_vector_2"]] = pd.DataFrame(df.parallel_apply(lambda x: find_comp_mut.find_mutation_ext(x.base_S2, \
                                                        x.S2, x.stem_start_idx_2, x.stem_end_idx_2, x.rc_start_idx_2, x.rc_end_idx_2, valid), axis=1).tolist())
    else:
        df[["totaMut_2", "stemMut_2", "compMut_2", "strucNotation_2"]] = pd.DataFrame(df.parallel_apply(lambda x: find_comp_mut.find_mutation(x.base_S2, \
                                                        x.S2, x.stem_start_idx_2, x.stem_end_idx_2, x.rc_start_idx_2, x.rc_end_idx_2), axis=1).tolist())
    df["db_strucNotation_2"] = df.parallel_apply(lambda x: find_comp_mut.db_notation_from_old_notaion(x.strucNotation_2), axis=1)
    df["symbol_strucNotation_2"]= df.parallel_apply(lambda x: find_comp_mut.symbol_notation_from_old_notaion(x.strucNotation_2, x.db_strucNotation_2), axis=1)

    # compute compactor_p (target_p in target mode) using summation of two segaments
    df['stemL'] = df['stemL_1'] + df['stemL_2']
    df['totaMut'] = df['totaMut_1'] + df['totaMut_2']
    df['stemMut'] = df['stemMut_1'] + df['stemMut_2']
    if extended:
        # E and stem-composition concatenation b_1 ⊕ b_2; DP is order-
        # invariant in b so the concatenation convention is arbitrary.
        df['E'] = df['E_1'] + df['E_2']
        df['b_vector'] = df.apply(lambda x: list(x['b_vector_1']) + list(x['b_vector_2']), axis=1)
        df['n_p_vector'] = df.apply(lambda x: list(x['n_p_vector_1']) + list(x['n_p_vector_2']), axis=1)
    else:
        df['compMut'] = df['compMut_1'] + df['compMut_2']

    """ Step 4: Calculate structure target-p """
    df["compactor_p"] = df.parallel_apply(lambda x: get_pval.target_p(2 * len(x['base_S1']), x['stemL'], x['totaMut'], x['stemMut'], x['compMut']), axis=1) 

    """ Step 5: Calculate anchor_score_per_split """
    df["anchor_score_per_split"] = df["compactor_weight"] * df["compactor_p"]
    df["anchor_score_per_split"] = df.groupby(["anchor", "segment_index"])["anchor_score_per_split"].transform("sum")
    df['anchor_split'] = df.apply(lambda x: x['anchor'] + '_' + x['segment_index'], axis=1)

    """ Step 6: Calculate anchor_p """
    df = get_pval.wrap_anchor_p_compactor(df)

    """ Step 7: BH correction on anchors with number of compactor > 2 """
    # Filter the DataFrame to keep rows with 'anchor_split' counts greater than 2
    df = df[df.groupby('anchor_split')['compactor'].transform('count') > 2].reset_index(drop=True)
    
    # BH correction on anchors with number of num_compactor_split > 2 (this filter can be added before) 
    df_temp = df[['anchor_split', 'anchor_p']].drop_duplicates()
    correction = multipletests(df_temp['anchor_p'], alpha=0.05, method='fdr_bh')
    df_temp['anchor_p_BH'] = correction[1]
    
    # Merge back
    df_temp = df_temp.drop(columns=['anchor_p'])
    df = df.merge(df_temp, on='anchor_split', how = 'left')

    # Sort dataframe
    df = df.sort_values(by = ['anchor_p_BH', 'anchor_split'], ascending=True).reset_index(drop=True)

    """ Step 8: SAVE """
    df.to_csv(f'{outfolder}/structure_on_compactors.tsv', index=False, sep='\t')

    if element_annotation:
        """ Step 9: elememt annotations (optional, toggle on by -a)  """
        # run element annotations
        elem_ann_folder = elem_annas.run_anns(outfolder, "compactor", 40)

        """ Step 10: merge structure results with element annotations """
        anns_file = f"{outfolder}/elem_anns/{elem_ann_folder}_work/{elem_ann_folder}/element_annotations/element_annotations_anchors.tsv"
        df_anns = pd.read_csv(anns_file, sep='\t')
        df = elem_annas.merge_anns_struc(df_anns, df, "compactor")
        df.to_csv(f'{outfolder}/structure_on_compactors.tsv', index=False, sep='\t')

def run_SS_compactor():
    arguments = argument_parser_compactor()
    SS_compactor(**arguments)

if __name__ == "__main__":
    arguments = argument_parser_compactor()
    SS_compactor(**arguments)
