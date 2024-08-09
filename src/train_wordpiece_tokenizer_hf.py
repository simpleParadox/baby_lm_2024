from transformers import AutoProcessor, AutoModelForCausalLM

# processor = AutoProcessor.from_pretrained("babylm/git-2024", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/git-base")
tokenizer = processor.tokenizer
print("Tokenizer: ", tokenizer)
from pathlib import Path
# From the paths variable, create a new pandas with one column called 'text'.
# For each path in paths, read the file and append the text to the 'text' column.
# Save the pandas to a csv file.
import pandas as pd
import os
import json
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm

import sys
sys.path.append('/home/rsaha/projects/babylm/')
sys.path.append('/home/rsaha/projects/babylm/data/')
sys.path.append('/home/rsaha/projects/babylm/src/')


# Do preprocessing of the json captions for training the tokenizer.
DATA_ROOT = Path("./")

data_dir = Path("./data/train_50M_multimodal_clean/") # Make sure the path is correct here.

paths = [str(f) for f in data_dir.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".train"]]
# paths = [str(f) for f in data_dir.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and (f.name not in ["cc_3M_captions_reduced.train", "local_narr_captions.train"])]
print(paths)


texts = []
for path in tqdm(paths):
    with open(path, 'r') as f:
        data = f.readlines()
        for line in tqdm(data):
            texts.append(line)

# Calculate the length of the longest text (in characters).
caption_lengths = {} 
for i, text in tqdm(enumerate(texts)):
    caption_lengths[i] = len(text)
# Get the index of the longest text.
max_caption_length = max(caption_lengths, key=caption_lengths.get)
# Get the longest text.
longest_caption = texts[max_caption_length]
data_dict = {'text': texts}

# data = pd.DataFrame(texts, columns=['text'])
data = datasets.Dataset.from_dict(data_dict)

def get_training_corpus():
    dataset = data["text"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples

training_corpus = get_training_corpus()
# Train 
tokenizer.train_new_from_iterator(training_corpus, vocab_size=32768)
tokenizer.save_pretrained("/home/rsaha/projects/babylm/src/tokenizer/hf_wordpiece_tokenizer_from_bert-base-uncased/")
# tokenizer.train_new_from_iterator(get_training_corpus, vocab_size=30522, min_frequency=2, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])