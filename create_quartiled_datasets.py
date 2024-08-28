"""
This script will create the quartiled datasets for the curriculum learning training procedure.
"""


import pandas as pd
from tqdm import tqdm
seed = 0
df = pd.read_csv(f"/home/rsaha/projects/babylm/src/datasets/multimodal_train/all_multimodal_all_concaps_with_pos_tags_with_noun_counts_assigned_max_replaced_train_seed_{seed}.tsv", sep="\t")

# Now for each seed, create a separate dataset for each quartile.

# The dataframes should contain the difficulty values for each seed that are 'LESS THAN' the following values.
# The values are same of reach seed.
predefined_quartiles_difficulty_values = [3, 4, 6]  # These values are obtained from the histogram of the noun_counts_seed_0_replaced column.

for seed in tqdm(range(0, 3)):
    df_quartile_1 = df[df[f'noun_counts_seed_{seed}_replaced'] < predefined_quartiles_difficulty_values[0]]
    df_quartile_2 = df[df[f'noun_counts_seed_{seed}_replaced'] < predefined_quartiles_difficulty_values[1]]
    df_quartile_3 = df[df[f'noun_counts_seed_{seed}_replaced'] < predefined_quartiles_difficulty_values[2]]
    # No need to create the fourth quartile, because it will contain all the rows, which is df itself.
    # Now save the dataframes to tsv files.
    df_quartile_1.to_csv(f"/home/rsaha/projects/babylm/src/datasets/multimodal_train/curriculum_tsvs_95_train_replaced/quartile_1_seed_{seed}_assigned_max_replaced_train.tsv", sep="\t", index=False)
    df_quartile_2.to_csv(f"/home/rsaha/projects/babylm/src/datasets/multimodal_train/curriculum_tsvs_95_train_replaced/quartile_2_seed_{seed}_assigned_max_replaced_train.tsv", sep="\t", index=False)
    df_quartile_3.to_csv(f"/home/rsaha/projects/babylm/src/datasets/multimodal_train/curriculum_tsvs_95_train_replaced/quartile_3_seed_{seed}_assigned_max_replaced_train.tsv", sep="\t", index=False)
    