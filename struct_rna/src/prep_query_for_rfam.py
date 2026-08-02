import pandas as pd
import sys

df = pd.read_csv(sys.argv[1], sep='\t').loc[lambda df: df['anchor_p_BH'] < 0.1]

with open(f"{sys.argv[2]}.fasta", "w") as f:
    if sys.argv[2] == "extendor":
        for i in range(df.shape[0]):
            f.write(f">Sequence{i+1}\n{df.loc[i, 'anchor']+df.loc[i, 'target']}\n")
    else:
        for i in range(df.shape[0]):
            f.write(f">Sequence{i+1}\n{df.loc[i, 'compactor']}\n")
