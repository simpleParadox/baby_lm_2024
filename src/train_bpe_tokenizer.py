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
import tqdm
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
    'cc_3M_captions': cleanup_captions,
    'local_narr_captions': cleanup_captions
}
# for split in DATA_SPLITS:
#     INPUT_DIR = DATA_ROOT / 'data' / split
#     OUTPUT_DIR = DATA_ROOT / 'data' / f'{split}_multimodal_clean'
    
#     OUTPUT_DIR.mkdir(exist_ok=True)

#     train_files = [f for f in INPUT_DIR.iterdir() if f.is_file() and f.suffix in ['.train']]
    
#     for file in train_files:
#         text = file.read_text()
#         cleaned_text = CLEANUP_FUNCTIONS[file.stem](text, SEQ_LENGTH)
#         (OUTPUT_DIR / file.name).write_text(cleaned_text)
#         print(f"🧹 Cleaned '{file.name}' (size {len(text)} -> {len(cleaned_text)}) in {split}")



# Also do the same for the caption data.
# DATA_SPLITS = ['caption_data']
# for split in DATA_SPLITS:
#     INPUT_DIR = DATA_ROOT / 'data' / split
#     OUTPUT_DIR = DATA_ROOT / 'data' / 'train_50M_multimodal_clean'
    
#     OUTPUT_DIR.mkdir(exist_ok=True)

#     train_files = [f for f in INPUT_DIR.iterdir() if f.is_file() and f.suffix in ['.train']]
    
#     for file in train_files:
#         text = file.read_text()
#         print("File name: ", file.stem)
#         cleaned_text = CLEANUP_FUNCTIONS[file.stem](text, SEQ_LENGTH)
#         (OUTPUT_DIR / file.name).write_text(cleaned_text)
#         print(f"🧹 Cleaned '{file.name}' (size {len(text)} -> {len(cleaned_text)}) in {split}")



data_dir = Path("./data/train_50M_multimodal_clean/") # Make sure the path is correct here.

paths = [str(f) for f in data_dir.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".train"]]


tokenizer = Tokenizer(models.BPE())

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True) 
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
tokenizer.normalizer = NFKC()


trainer = trainers.BpeTrainer(vocab_size=16000, min_frequency=2, special_tokens=["<pad>", "<s>", "</s>"])
tokenizer.train(paths, trainer)

tokenizer_path =  DATA_ROOT / "src/tokenizer/multi_50m_and_captions_tokenizer_bpe.json"
tokenizer.save(str(tokenizer_path), pretty=True)




# Now train a BertWordPiece tokenizer using the similar framework above.








tokenizer = Tokenizer.from_file(str(tokenizer_path))
text = "hello The quick brown fox jumps over the lazy dog."

encoded = tokenizer.encode(text)
print(f"Encoded String: {encoded.tokens}")

print(f"Encoded IDs: {encoded.ids}")

decoded = tokenizer.decode(encoded.ids, skip_special_tokens=True)
print(f"Decoded String: {decoded}")