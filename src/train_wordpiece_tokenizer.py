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

paths = [str(f) for f in data_dir.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".train"]]
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

tokenizer_path =  DATA_ROOT / "src/tokenizer/multi_50m_and_captions_tokenizer_bert_wordpiece.json"
tokenizer.save(str(tokenizer_path), pretty=True)



tokenizer = Tokenizer.from_file(str(tokenizer_path))
text = "hello The quick brown fox jumps over the lazy dog."

encoded = tokenizer.encode(text)
print(f"Encoded String: {encoded.tokens}")

print(f"Encoded IDs: {encoded.ids}")

decoded = tokenizer.decode(encoded.ids, skip_special_tokens=True)
print(f"Decoded String: {decoded}")