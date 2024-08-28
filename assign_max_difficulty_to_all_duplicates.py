import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
import numpy as np
from tqdm import tqdm
import sys
df = pd.read_csv("/home/rsaha/projects/babylm/src/datasets/multimodal_train/all_multimodal_all_concaps_with_pos_tags_with_noun_counts.tsv", sep="\t")


# In df_mscoco, there are some duplicate urls. For each of these duplicate urls, get the noun_counts for seed 0, and then assign the maximum noun_count to the duplicate urls.
# This is because the noun_counts for the same image_url should be the same.

# Get the list of urls that are repeated in df_mscoco.
repeated_urls = df[df.duplicated(subset=['image_url'], keep=False)]['image_url'].tolist()


seed = int(sys.argv[1])

for image_url in tqdm(repeated_urls):
    noun_counts = df[df["image_url"]==image_url][f"noun_counts_seed_{seed}"].tolist()
    max_noun_count = max(noun_counts)
    df.loc[df["image_url"]==image_url, f"noun_counts_seed_{seed}"] = max_noun_count

# Save the df to a new .tsv file.
df.to_csv(f"/home/rsaha/projects/babylm/src/datasets/multimodal_train/all_multimodal_all_concaps_with_pos_tags_with_noun_counts_assigned_max_seed_{seed}.tsv", sep="\t", index=False)




# Now, load the df for each seed and then replace the column
# """
# NOTE: Do this after the running the previous block.
# """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
import numpy as np
from tqdm import tqdm
import sys
df = pd.read_csv("/home/rsaha/projects/babylm/src/datasets/multimodal_train/all_multimodal_all_concaps_with_pos_tags_with_noun_counts.tsv", sep="\t")

for seed in [0, 1, 2]:
    print(f"Processing seed {seed}")
    df_seed = pd.read_csv(f"/home/rsaha/projects/babylm/src/datasets/multimodal_train/all_multimodal_all_concaps_with_pos_tags_with_noun_counts_assigned_max_seed_{seed}.tsv", sep="\t")
    df[f"noun_counts_seed_{seed}_replaced"] = df_seed[f"noun_counts_seed_{seed}"]
    
# Save the df to a new .tsv file.
df.to_csv(f"/home/rsaha/projects/babylm/src/datasets/multimodal_train/all_multimodal_all_concaps_with_pos_tags_with_noun_counts_assigned_max_replaced.tsv", sep="\t", index=False)


"""
The following code is for creating a 5% held out validation set.
"""
import pandas as pd
from tqdm import tqdm
import numpy as np


df = pd.read_csv("/home/rsaha/projects/babylm/src/datasets/multimodal_train/all_multimodal_all_concaps_with_pos_tags_with_noun_counts_assigned_max_replaced.tsv", sep="\t")
seed = 0
# Randomly sample 5% of the rows and store it in df_val.
df_val = df.sample(frac=0.05, random_state=seed)

# Remove the rows that are present in df_val from df.
df_train = df.drop(df_val.index)

# Save the df and df_val to tsv files.
df_train.to_csv(f"/home/rsaha/projects/babylm/src/datasets/multimodal_train/all_multimodal_all_concaps_with_pos_tags_with_noun_counts_assigned_max_replaced_train_seed_{seed}.tsv", sep="\t", index=False)
df_val.to_csv(f"/home/rsaha/projects/babylm/src/datasets/multimodal_train/all_multimodal_all_concaps_with_pos_tags_with_noun_counts_assigned_max_replaced_val_seed_{seed}.tsv", sep="\t", index=False)

