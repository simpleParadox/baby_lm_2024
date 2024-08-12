"""
Count the number of words in all the train files.
"""

from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append('/home/rsaha/projects/babylm/')
sys.path.append('/home/rsaha/projects/babylm/data/')
sys.path.append('/home/rsaha/projects/babylm/src/')


# Do preprocessing of the json captions for training the tokenizer.
DATA_ROOT = Path("./")

data_dir = Path("./data/train_50M_multimodal_clean/") # Make sure the path is correct here.

paths = [str(f) for f in data_dir.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".train"] and f.name not in ["cc_3M_captions_reduced.train"]]
# paths = [str(f) for f in data_dir.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and (f.name not in ["cc_3M_captions_reduced.train", "local_narr_captions.train"])]
# paths.append("data/caption_data_multimodal_clean/cc_3M_captions_non_reduced.train")
print(paths)


texts = []
for path in tqdm(paths):
    print("Path: ", path)
    with open(path, 'r') as f:
        data = f.readlines()
        for line in tqdm(data):
            texts.append(line)
            
# For each text in texts, count the number of words.
number_of_words = [len(text.split()) for text in tqdm(texts)]
print("Number of words: ", sum(number_of_words))