import sys
sys.path.append('/home/rsaha/projects/babylm/')
sys.path.append('/home/rsaha/projects/babylm/src/')





import torch
from transformers import LlamaForCausalLM, LlamaConfig
from transformers import LlamaTokenizerFast

tokenizer_path = "/home/rsaha/projects/babylm/src/tokenizer/new_bpe_strict_small_tokenizer.json"

llama_tokenizer = LlamaTokenizerFast(tokenizer_file=tokenizer_path)

llama_tokenizer.bos_token = "<s>"
llama_tokenizer.eos_token = "</s>"
llama_tokenizer.pad_token = "<pad>"

SEQ_LENGTH = 128

NUM_EPOCHS=10

config = LlamaConfig(
    vocab_size=llama_tokenizer.vocab_size,
    hidden_size=256,
    num_hidden_layers=16,
    intermediate_size=2048,
    num_attention_heads=16,
    bos_token_id=llama_tokenizer.convert_tokens_to_ids("<s>"),
    eos_token_id=llama_tokenizer.convert_tokens_to_ids("</s>"),
    pad_token_id=llama_tokenizer.convert_tokens_to_ids("<pad>"),
    max_position_embeddings=2*SEQ_LENGTH,
)


# Create the LLaMa model from scratch
model = LlamaForCausalLM(config)
