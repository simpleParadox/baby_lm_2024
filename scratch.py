import pandas as pd
import json



# Read in the following file as a text file and create a list for each line.
# /home/rsaha/projects/babylm/data/train_50M_multimodal_clean/local_narr_captions.train

local_narr_captions = open("/home/rsaha/projects/babylm/data/train_50M_multimodal_clean/local_narr_captions.train", "r").readlines()



# Count the number of words in the captions for conceptual captions dataset.
file_path = "/home/rsaha/projects/babylm/cc_3m_training_exists/concatenated_training_exists_with_captions_non_reduced_filtered.tsv"
cc_3m_training_exists = pd.read_csv(file_path, sep="\t", compression='gzip')



# Select the rows only where exists=1, because some rows also contain 0.
cc_3m_training_exists_filtered = cc_3m_training_exists[cc_3m_training_exists['exists'] == 1]


print("Number of rows in the filtered dataframe before dropping rows: ", cc_3m_training_exists_filtered.shape[0])

# Randomly drop 3% of the rows.
cc_3m_training_exists_filtered_reduced = cc_3m_training_exists_filtered.sample(frac=0.97, random_state=42)

# Print the number of rows.
print("Number of rows in the filtered dataframe reduced: ", cc_3m_training_exists_filtered_reduced.shape[0])

# Save the filtered dataframe to a tsv file.
cc_3m_training_exists_filtered.to_csv("cc_3m_training_exists/concatenated_training_exists_with_captions_non_reduced.tsv", sep="\t", index=False, compression='gzip')

# Create a list with all the captions from the filtered dataframe.
existing_captions = cc_3m_training_exists_filtered['caption'].tolist()

# Flatten the list of captions into a single string.
existing_captions_string = " ".join(existing_captions)

# Count the number of words in the string.
number_of_words = len(existing_captions_string.split(" "))
print("Number of words in the captions for the conceptual captions dataset: ", number_of_words)
from pathlib import Path
DATA_ROOT = Path("./")
split = 'caption_data'
INPUT_DIR = DATA_ROOT / 'data' / split
OUTPUT_DIR = DATA_ROOT / 'data' / f'{split}_multimodal_clean'

OUTPUT_DIR.mkdir(exist_ok=True)

new_file_name = 'cc_3M_captions_reduced_97' + ".train"
all_captions = '\n'.join(existing_captions)

(OUTPUT_DIR / new_file_name).write_text(all_captions)


# # Load the local_narr_captions.json file to calculate the number of words for that.
# local_narr_captions = json.load(open("data/caption_data/local_narr_captions.json"))

# # For each caption in the local_narr_captions, count the number of words.
# number_of_words_local_narr = [len(caption.split()) for caption in local_narr_captions]

# # Calculate the total number of words in the local_narr_captions.
# total_number_of_words_local_narr = sum(number_of_words_local_narr)



import sys
sys.path.append('/home/rsaha/projects/babylm/')
sys.path.append('/home/rsaha/projects/babylm/data/')
sys.path.append('/home/rsaha/projects/babylm/src/')


from tokenizers import Tokenizer, decoders, models, trainers, processors, pre_tokenizers
from tokenizers.normalizers import NFKC

from pathlib import Path
from utils.mrclean import *
from tqdm import tqdm
import json


# Do preprocessing of the json captions for training the tokenizer.
DATA_ROOT = Path("./")
SEQ_LENGTH = 128 # this is a legacy parameter, it does not affect cleaning
DATA_SPLITS = ['train_50M']

CLEANUP_FUNCTIONS = {
    'aochildes': cleanup_aochildes,
    'bnc_spoken': cleanup_bnc_spoken,
    'cbt': cleanup_cbt,
    'childes': cleanup_children_stories,
    'gutenberg': cleanup_gutenberg,
    'open_subtitles': cleanup_open_subtitles,
    'qed': cleanup_qed,
    'simple_wiki': cleanup_simple_wikipedia,
    'switchboard': cleanup_switchboard,
    'wikipedia': cleanup_wikipedia,
    'cc_3m_caption': cleanup_captions,
    'local_narr_captions': cleanup_captions
}

split = 'caption_data'
INPUT_DIR = DATA_ROOT / 'data' / split
OUTPUT_DIR = DATA_ROOT / 'data' / f'{split}_multimodal_clean'

OUTPUT_DIR.mkdir(exist_ok=True)

train_files = [f for f in INPUT_DIR.iterdir() if f.is_file() and f.suffix in ['.json']]

for file in train_files:
    captions = json.load(open(file))
    # Create a .txt file with the extension .train and the file name as file.stem.
    
    new_file_name = file.stem + ".train"
    # Create a string that contains all the captions, but each caption is on a new line.
    all_captions = '\n'.join(captions)
    
    # Write to a new file.
    (OUTPUT_DIR / new_file_name).write_text(all_captions)
    
    

# local_narr_captions = json.load(open("caption_data/local_narr_captions.json"))
# # Flatten the list of captions into a single string.
# local_narr_captions_string = " ".join(local_narr_captions)
# # Count the number of words in the string.
# number_of_words_local_narr = len(local_narr_captions_string.split())


# def cleanup_captions(text, seq_length):
#     # Put each caption on a new line.
#     return text








"""
Create a new tsv file that contains the concatenated training exists files with the captions from the original dataset.
This will later be used to train a model on the images that already exists in the disk.
"""


# import datasets
# from datasets import load_dataset
# import pandas as pd
# import glob

# # First load all the individual tsvs in the cc_3m_training_exists folder and concatenate them.

# all_training_exists_files = glob.glob("cc_3m_training_exists/*.tsv")

# all_dfs = []

# for file in all_training_exists_files:
#     all_dfs.append(pd.read_csv(file, sep="\t", header=None, compression='gzip'))

# concat_training_exists = pd.concat(all_dfs, ignore_index=True)

# # Rename columns to match the original dataset.
# mapping = {0: "image_url", 1: "folder", 2: "exists"}
# concat_training_exists_renamed = concat_training_exists.rename(columns=mapping, inplace=False)


# ds = load_dataset("google-research-datasets/conceptual_captions", "unlabeled")
# df_train = ds['train']


# df_train_captions = df_train['caption']

# # Create a new column in the training exists dataframe that contains the captions.
# concat_training_exists_renamed['caption'] = df_train_captions


# # Save the new dataframe to a tsv file with gzip compression.
# concat_training_exists_renamed.to_csv("cc_3m_training_exists/concatenated_training_exists_with_captions.tsv", sep="\t", index=False, compression='gzip')