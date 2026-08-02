import pandas as pd
import sys
import csv

df_structure = pd.read_csv(sys.argv[1], sep='\t').loc[lambda df: df['anchor_p_BH'] < 0.1]
df_rfam = pd.read_csv(sys.argv[2],sep ='\t',quoting=csv.QUOTE_NONE)
if 'query name' not in df_structure.columns:
    df_structure['query name'] = pd.Series([f'Sequence{i+1}' for i in range(len(df_structure))])
df_structure=df_structure.merge(df_rfam.drop_duplicates('query name'), how='left', on='query name')
df_structure.to_csv(f'{sys.argv[1][0:-4]}.RFAM.tsv', index=False, sep='\t')
