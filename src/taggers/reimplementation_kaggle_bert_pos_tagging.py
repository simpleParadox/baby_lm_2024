# %%
from datasets import Dataset
from pathlib import Path
import os
import numpy as np
import re
from tqdm import tqdm
import pickle
import sys
import nltk
from nltk.data import load as nltk_load
import os
import torch
from transformers import BertTokenizer, PreTrainedTokenizerFast
from transformers import AutoModelForTokenClassification, BertConfig, BertForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import get_scheduler
from accelerate import Accelerator

os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# %% [markdown]
# Create a huggingface Dataset from the stored sentence and pos tag data.

# %%

device = sys.argv[1]

class TQDMBytesReader(object):
    # For showing the progress bar while reading the stored pickle file.
    def __init__(self, fd, **kwargs):
        self.fd = fd
        from tqdm import tqdm
        self.tqdm = tqdm(**kwargs)

    def read(self, size=-1):
        bytes = self.fd.read(size)
        self.tqdm.update(len(bytes))
        return bytes

    def readline(self):
        bytes = self.fd.readline()
        self.tqdm.update(len(bytes))
        return bytes

    def __enter__(self):
        self.tqdm.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        return self.tqdm.__exit__(*args, **kwargs)


# %%
def load_data(file_name):
    # if os.path.exists(f"/home/rsaha/projects/babylm/src/taggers/processed_tagger_data/processed_pos_training_data_{file_name}.pkl"):
    if os.path.exists(f"/home/rsaha/projects/babylm/src/taggers/data/{file_name}.pkl"):
        print("Loading data from file ...")
        with open(f"/home/rsaha/projects/babylm/src/taggers/processed_tagger_data/processed_pos_training_data_{file_name}.pkl", "rb") as f:
            total = os.path.getsize(f"/home/rsaha/projects/babylm/src/taggers/processed_tagger_data/processed_pos_training_data_{file_name}.pkl")
            with TQDMBytesReader(f, total=total) as pbfd:
                up = pickle.Unpickler(pbfd)
                X_data, y_data = up.load()
            return X_data, y_data

# %%
def extract_features(sentence, index):
  return {
      'word':sentence[index],
      'is_first':index==0,
      'is_last':index ==len(sentence)-1,
      'is_capitalized':sentence[index][0].upper() == sentence[index][0],
      'is_all_caps': sentence[index].upper() == sentence[index],
      'is_all_lower': sentence[index].lower() == sentence[index],
      'is_alphanumeric': int(bool((re.match('^(?=.*[0-9]$)(?=.*[a-zA-Z])',sentence[index])))),
      'prefix-1':sentence[index][0],
      'prefix-2':sentence[index][:2],
      'prefix-3':sentence[index][:3],
      'prefix-3':sentence[index][:4],
      'suffix-1':sentence[index][-1],
      'suffix-2':sentence[index][-2:],
      'suffix-3':sentence[index][-3:],
      'suffix-3':sentence[index][-4:],
      'prev_word':'' if index == 0 else sentence[index-1],
      'next_word':'' if index < len(sentence) else sentence[index+1],
      'has_hyphen': '-' in sentence[index],
      'is_numeric': sentence[index].isdigit(),
      'capitals_inside': sentence[index][1:].lower() != sentence[index][1:],
  }

# %%
data_dir = Path("/home/rsaha/projects/babylm/src/taggers/data/")
paths = [str(f) for f in data_dir.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".pkl"]]
print("Paths: ", paths)

file_names = []
# Only select the cc_3m and local_narr files and store it in filtered_paths.
# filtered_paths = paths #[]
# # for path in paths:
# #     if "cc_3M" in path or "local_narr" in path:
# #         filtered_paths.append(path)

for path in tqdm(paths, desc="Paths"):

    file_name = Path(path).name
    # Drop the .train extension
    file_name = file_name.split(".")[0]
    file_names.append(file_name)

# %%
# Load the data from each file_name in file_names.
all_X_data = []
all_y_data = []
for file_name in tqdm(file_names):
    print(file_name)
    if file_name == "pos_tagging_dataset_all_sentences_open_subtitles":
        data = pickle.load(open(f"/home/rsaha/projects/babylm/src/taggers/data/{file_name}.pkl", "rb"))
        # all_X_data.extend(X_data)
        # all_y_data.extend(y_data)



# %%
# Create a dictionary of numbers where each tag (the second element in the tuple) is assigned a unique number. This will be the class labels.
tagdict = nltk_load('help/tagsets/upenn_tagset.pickle')
label_names = {t: i for i, t in enumerate(tagdict.keys())}
label_names['#'] = len(label_names)


# Remove all the empty lists from the data.
data = [d for d in data if d [0]!= []]


print("label_names", label_names)
# %%
# Data has many tuples and each tuple has two lists. The first list is the list of words and the second list is the list of tags.
# Create a third separate list of lists where each list contains the number from the label_names dictionary based on the key tag.
# This will be the class labels.
tag_to_class_mapping = []
for i in tqdm(range(len(data))):
    tag_to_class_mapping.append([label_names[tag] for tag in data[i][1]])

# %%
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels

# %%
# Load the tokenizer.
from transformers import BertTokenizer, PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained('/home/rsaha/projects/babylm/src/tokenizer/hf_wordpiece_tokenizer_from_git/')

# %%
labels = tag_to_class_mapping[0]
inputs = tokenizer(data[0][0], is_split_into_words=True)
word_ids = inputs.word_ids()


# %%
print(labels)
print(data[0][1])
print(word_ids)
print(align_labels_with_tokens(labels, word_ids))

# %%
# First create a dataframe from the sentence, tags, and class labels.
# NOTE: Each example in the data variable has two lists. The first list is the list of words and the second list is the list of tags.

import pandas as pd
df = pd.DataFrame(data, columns=["sentence", "tags"])
df["class_labels"] = tag_to_class_mapping


# %%
# Create train and validation splits using train_test_split.
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# %%
from datasets import Dataset
df_dataset_train = Dataset.from_pandas(train_df)
df_dataset_val = Dataset.from_pandas(val_df)


# %%
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["sentence"], truncation=True, is_split_into_words=True,
        max_length=50
    )
    all_labels = examples["class_labels"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

# %%
df_dataset_tokenized_train = df_dataset_train.map(tokenize_and_align_labels, batched=True,
                                      remove_columns=df_dataset_train.column_names, num_proc=20)
df_dataset_tokenized_eval = df_dataset_val.map(tokenize_and_align_labels, batched=True,
                                      remove_columns=df_dataset_train.column_names, num_proc=20)

# %%
# df_dataset_tokenized_train[0]

# %%
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# %%
# data_collator

# %%
batch = data_collator([df_dataset_tokenized_train[i] for i in range(4)])
print(batch["labels"])
for i in range(4):
    print(df_dataset_tokenized_train[i]["labels"])

# %%
id2label = {i: label for i, label in enumerate(label_names.keys())}
label2id = {v: k for k, v in id2label.items()}  # This is nothing but the label_names dictionary. But keeping it like this for consistency with the Kaggle notebook.

# %%
# label2id

# %%
from transformers import AutoModelForTokenClassification, BertConfig, BertForTokenClassification
bert_config = BertConfig()
model_checkpoint = "bert-base-uncased"
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, id2label=id2label, label2id=label2id)
# model = AutoModelForTokenClassification.from_config(
#     teacher_model.config,
# )

# %%
import evaluate

metric = evaluate.load("seqeval")

# %%
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

# %%
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    df_dataset_tokenized_train,
    shuffle=False,
    collate_fn=data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    df_dataset_tokenized_eval, collate_fn=data_collator, batch_size=8
)

# %% [markdown]
# ### Test the following code

# %%
from torch.optim import AdamW, Adam

# optimizer = Adam(model.parameters(), lr=2e-5)
optimizer = Adam(model.parameters(), lr=2e-5)

# %%
from accelerate import Accelerator
import os
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# accelerator = Accelerator(mixed_precision="fp16")
# model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
#     model, optimizer, train_dataloader, eval_dataloader
# )
# import torch
# scaler = torch.cuda.amp.GradScaler(enabled=True)

# %%
from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# %%
def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions

# %%
# Put the device to the GPU.
model.to(device)

# %%
from tqdm.auto import tqdm
import torch

progress_bar = tqdm(range(num_training_steps))

for epoch in tqdm(range(num_train_epochs)):
    # Training
    model.train()
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        print("batch: ", batch)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
            
        # loss.backward()
        #     # # accelerator.backward(loss)

        # optimizer.step()
        # # scaler.scale(loss).backward()
        # # scaler.step(optimizer)
        # # # lr_scheduler.step()
        # # scaler.update()
        # optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    # model.eval()
    # for batch in eval_dataloader:
    #     with torch.no_grad():
    #         outputs = model(**batch)

    #     predictions = outputs.logits.argmax(dim=-1)
    #     labels = batch["labels"]

    #     # Necessary to pad predictions and labels for being gathered
    #     predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
    #     labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

    #     predictions_gathered = accelerator.gather(predictions)
    #     labels_gathered = accelerator.gather(labels)

    #     true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
    #     metric.add_batch(predictions=true_predictions, references=true_labels)

    # results = metric.compute()
    # print(
    #     f"epoch {epoch}:",
    #     {
    #         key: results[f"overall_{key}"]
    #         for key in ["precision", "recall", "f1", "accuracy"]
    #     },
    # )

# %%
# batch['labels']

# %% [markdown]
# # Code below is for testing and not part of the pipeline.

# %%
# Now I have three lists: data contains two lists of words and tags, tag_to_class_mapping contains the corresponding class labels.
# labels = tag_to_class_mapping[0]


# # %%
# inputs.word_ids()

# # %%
# batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
# print(batch["labels"])
# for i in range(2):
#     print(tokenized_datasets["train"][i]["labels"])

# # %%
# data = datasets.Dataset.from_dict(data_dict)


