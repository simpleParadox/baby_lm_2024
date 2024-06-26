import sys
sys.path.append('/home/rsaha/projects/babylm/')
sys.path.append('/home/rsaha/projects/babylm/data/')
sys.path.append('/home/rsaha/projects/babylm/src/')


from tokenizers import Tokenizer, decoders, models, trainers, processors, pre_tokenizers
from tokenizers.normalizers import NFKC

from pathlib import Path
from utils.mrclean import *


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
    'cc_3M_captions_reduced': cleanup_captions,
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
DATA_SPLITS = ['caption_data']
for split in DATA_SPLITS:
    INPUT_DIR = DATA_ROOT / 'data' / split
    OUTPUT_DIR = DATA_ROOT / 'data' / 'train_50M_multimodal_clean'
    
    OUTPUT_DIR.mkdir(exist_ok=True)

    train_files = [f for f in INPUT_DIR.iterdir() if f.is_file() and f.suffix in ['.train']]
    
    for file in train_files:
        text = file.read_text()
        print("File name: ", file.stem)
        cleaned_text = CLEANUP_FUNCTIONS[file.stem](text, SEQ_LENGTH)
        (OUTPUT_DIR / file.name).write_text(cleaned_text)
        print(f"🧹 Cleaned '{file.name}' (size {len(text)} -> {len(cleaned_text)}) in {split}")

