"""
Train the CLIP tokenizers from scratch.
"""

import sys
sys.path.append('/home/rsaha/projects/babylm/')
sys.path.append('/home/rsaha/projects/babylm/data/')
sys.path.append('/home/rsaha/projects/babylm/src/')


from tokenizers import Tokenizer, decoders, models, trainers, processors, pre_tokenizers
from tokenizers.normalizers import NFKC

from pathlib import Path
from utils.mrclean import *


from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer


# Do preprocessing of the json captions for training the tokenizer.
DATA_ROOT = Path("./")

data_dir = Path("./data/train_50M_multimodal_clean/") # Make sure the path is correct here.

# paths = [str(f) for f in data_dir.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".train"]]
paths = [str(f) for f in data_dir.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and (f.name not in ["cc_3M_captions_reduced.train", "local_narr_captions.train"])]
print(paths)

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.pre_tokenizer = Whitespace()


tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)



trainer = WordPieceTrainer(vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
# files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
tokenizer.train(paths, trainer)

tokenizer_path =  DATA_ROOT / "src/tokenizer/multi_50m_and_captions_tokenizer_bert_wordpiece_text_only.json"
tokenizer.save(str(tokenizer_path), pretty=True)



tokenizer = Tokenizer.from_file(str(tokenizer_path))
text = "hello The quick brown fox jumps over the lazy dog."

encoded = tokenizer.encode(text)
print(f"Encoded String: {encoded.tokens}")

print(f"Encoded IDs: {encoded.ids}")

decoded = tokenizer.decode(encoded.ids, skip_special_tokens=True)
print(f"Decoded String: {decoded}")










from transformers import AutoProcessor, AutoModelForCausalLM

processor = AutoProcessor.from_pretrained("babylm/git-2024", trust_remote_code=True)
tokenizer = processor.tokenizer
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
# paths = [str(f) for f in data_dir.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and (f.name in ["cc_3M_captions_reduced.train", "local_narr_captions.train"])]
print(paths)


texts = []
for path in tqdm(paths):
    with open(path, 'r') as f:
        data = f.readlines()
        for line in tqdm(data):
            texts.append(line)

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
tokenizer.save_pretrained("/home/rsaha/projects/babylm/src/tokenizer/hf_wordpiece_tokenizer_from_git/")
# tokenizer.train_new_from_iterator(get_training_corpus, vocab_size=30522, min_frequency=2, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])