"""
This script merges the structure results with the element annotations.
Two input files are needed:
1. Full path to the structure results file (e.g. compactors_20mers.fasta_RFAM.tsv)
2. Full path to the element annotations file (e.g. elem_anns/nf_annotations_work/nf_annotations/element_annotations/element_annotations_anchors.tsv)
Output file will be saved in the same directory as the structure results file with the name
"""
import pandas as pd
import numpy as np
import sys
import os
import subprocess
import time
import multiprocessing

def process_chunk(chunk):
    results = {}
    for index, row in chunk.iterrows():
        anchor = row['anchor']
        hit_info = {}

        for col in chunk.columns:
            if 'hits' in col and '_pos' not in col:
                if row[col] != '*':
                    pos_col = col.replace('hits', 'hits_pos')
                    hit_info[col] = row[pos_col]

        results[anchor] = hit_info

    return results

def parallel_process(df, num_partitions, num_cores):
    df_split = np.array_split(df, num_partitions)
    pool = multiprocessing.Pool(num_cores)
    results = pool.map(process_chunk, df_split)
    pool.close()
    pool.join()
    return {k: v for d in results for k, v in d.items()}

def helper_creat_anchor_list(outfolder: str, seq_type: str, seq_len: int = None):
    # create anchor list for annotations
    if seq_type == "compactor":
        input_file = f'{outfolder}/structure_on_compactors_{seq_len}mers.tsv'
        output_file = f'{outfolder}/elem_anns/compactors_{seq_len}.txt'
        command = f"awk -F'\t' 'NR==1 {{print $1}} NR>1 {{print $2}}' {input_file} > {output_file}"
    else:
        input_file = f'{outfolder}/structure_on_targets.tsv'
        output_file = f'{outfolder}/elem_anns/extendors.txt'
        command = f"awk -F'\t' 'NR==1 {{print $1}} NR>1 {{print $1 $8}}' {input_file} > {output_file}"

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    return os.path.abspath(output_file)

def run_anns(outfolder: str, seq_type: str = "extendor", seq_len: int = None):
    os.makedirs(f"{outfolder}/elem_anns/", exist_ok=True)
    
    if seq_type == "compactor":
        if seq_len is None:
            raise ValueError("seq_len is required for 'compactor' seq_type.")
        elem_ann_folder = f"nf_anns_{seq_type}_{seq_len}"
    else:
        elem_ann_folder = f"nf_anns_{seq_type}"

    # create anchor list for annotations
    anchor_list = helper_creat_anchor_list(outfolder, seq_type, seq_len)

    # run element annotations and capture its ID
    command = f"sbatch /oak/stanford/groups/horence/juliew/structure/scripts/elem_anns.sh {anchor_list} {outfolder} {elem_ann_folder}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split()[-1]

    # Poll the job status
    while True:
        status_command = f"squeue -j {job_id}"
        status_result = subprocess.run(status_command, shell=True, capture_output=True, text=True)

        if job_id not in status_result.stdout:
            break  # Job is no longer in the queue

        time.sleep(10)  # Wait for some time before checking again

    return elem_ann_folder

def merge_anns_struc(df_anns, df_struc, seq_type: str = "extendor"):
    num_partitions = 20                                     # number of partitions to split dataframe
    num_cores = multiprocessing.cpu_count()                 # number of cores on your machine
    hits_results = parallel_process(df_anns, num_partitions, num_cores)
    if seq_type == "compactor":
        df_struc['EA'] = df_struc.apply(lambda x: hits_results[x['compactor']], axis=1)
    else:
        df_struc['EA'] = df_struc.apply(lambda x: hits_results[x['anchor'] + x['base_target']], axis=1)
    return df_struc
