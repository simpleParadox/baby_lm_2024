"""
Use the following command to run the script:
CUDA_VISIBLE_DEVICES=2 python src/taggers/reimplementation_kaggle_bert_pos_tagging.py --batch_size 512 --num_train_epochs 5 --seed 2 --train_on_full_data
"""

import warnings
warnings.filterwarnings('ignore')
from datasets import Dataset
from pathlib import Path
import os
import numpy as np
import re
from tqdm import tqdm
import pickle
import nltk
import sys
from nltk.data import load as nltk_load
import json
import wandb
import argparse
import torch
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
# Create a huggingface Dataset from the stored sentence and pos tag data.

parser = argparse.ArgumentParser(description="Train a BERT POS Tagger from scratch using the Upenn tagset.  use the --train_on_full_data flag to train on the full data.")
parser.add_argument("--batch_size", type=int, default=512, help="The batch size for training.")
parser.add_argument("--num_train_epochs", type=int, default=5, help="The number of training epochs.")  # Five epochs should be enough.
parser.add_argument("--seed", type=int, default=0, help="The seed for the random number generator.")
parser.add_argument('--test_run', action='store_true', help="Whether to run the script in test mode.") # Default value is False.
parser.add_argument('--train_on_full_data', action='store_true', help="Whether to train on the full data or not. If provided, the model will be trained on the full data.") # Default value is False.
args = parser.parse_args()



torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False


wandb.init(project="train_bert_pos_tagger")
# wandb.init(project="train_bert_pos_tagger", mode='disabled')

data_dir = Path("/home/rsaha/projects/babylm/src/taggers/data/")
paths = [str(f) for f in data_dir.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".pkl"]]
print("Number of files: ", len(paths))

file_names = []

# Log the args to wandb.
args_dict = vars(args)
wandb.log(args_dict)

for path in tqdm(paths, desc="Paths"):

    file_name = Path(path).name
    # Drop the .train extension
    file_name = file_name.split(".")[0]
    file_names.append(file_name)

# Load the data from each file_name in file_names.
data = []
if args.test_run:
    print("Test run: ", args.test_run)
    temp_data = pickle.load(open(f"/home/rsaha/projects/babylm/src/taggers/data/pos_tagging_dataset_all_sentences_switchboard.pkl", "rb"))
    data.extend(temp_data)
else:
    for file_name in tqdm(file_names):
        print("Using all the files.")
        print(file_name)
        temp_data = pickle.load(open(f"/home/rsaha/projects/babylm/src/taggers/data/{file_name}.pkl", "rb"))
        data.extend(temp_data)
print("Length of data: ", len(data))

# Create a dictionary of numbers where each tag (the second element in the tuple) is assigned a unique number. This will be the class labels.
tagdict = nltk_load('help/tagsets/upenn_tagset.pickle')
label_names = {t: i for i, t in enumerate(tagdict.keys())}
label_names['#'] = len(label_names)

# Remove all the empty lists from data.
data = [d for d in data if d [0]!= []]


tag_to_class_mapping_for_data = []
for i in tqdm(range(len(data))):
    tag_to_class_mapping_for_data.append([label_names[tag] for tag in data[i][1]])

from transformers import BertTokenizer, PreTrainedTokenizerFast
model_checkpoint = "bert-base-uncased"
# tokenizer = PreTrainedTokenizerFast.from_pretrained('/home/rsaha/projects/babylm/src/tokenizer/hf_wordpiece_tokenizer_from_git/')
tokenizer = PreTrainedTokenizerFast.from_pretrained('/home/rsaha/projects/babylm/src/tokenizer/hf_wordpiece_tokenizer_from_bert-base-uncased/')

import pandas as pd
df = pd.DataFrame(data, columns=["sentence", "tags"])
df["class_labels"] = tag_to_class_mapping_for_data


from sklearn.model_selection import train_test_split

if args.train_on_full_data:
    train_df = df
else:
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=args.seed)

from datasets import Dataset
df_dataset_train = Dataset.from_pandas(train_df)
if not args.train_on_full_data:
    df_dataset_val = Dataset.from_pandas(val_df)


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
unk_count = 0
def tokenize_and_align_labels(examples):
    # print("Example sentence:  ", examples["sentence"])
    tokenized_inputs = tokenizer(
        examples["sentence"], truncation=True, is_split_into_words=True, max_length=50
    )
    split_tokenized_inputs = [
        tokenizer.tokenize(e, truncation=True, 
                           is_split_into_words=True, 
                           max_length=50, 
                           add_special_tokens=True) for e in examples['sentence']]
    # Increase unk_count if the tokenized inputs contain [UNK]
    global unk_count
    for split in split_tokenized_inputs:
        if "[UNK]" in split:
            unk_count += 1
    all_labels = examples["class_labels"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        # print("Word IDs: ", word_ids)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

df_dataset_tokenized_train = df_dataset_train.map(tokenize_and_align_labels, batched=True,
                                      remove_columns=df_dataset_train.column_names, num_proc=20)
if not args.train_on_full_data:
    df_dataset_tokenized_eval = df_dataset_val.map(tokenize_and_align_labels, batched=True,
                                        remove_columns=df_dataset_train.column_names, num_proc=20)



data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

id2label = {i: label for i, label in enumerate(label_names.keys())}
label2id = {v: k for k, v in id2label.items()}  # This is nothing but the label_names dictionary. But keeping it like this for consistency with the Kaggle notebook.


model_checkpoint = "bert-base-uncased"
teacher_model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, id2label=id2label, label2id=label2id)

# Now initialize a model from scratch.
model = AutoModelForTokenClassification.from_config(
    teacher_model.config,
)

import evaluate

metric = evaluate.load("seqeval")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
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

batch_size = args.batch_size
wandb.config.update({"batch_size": batch_size})
train_dataloader = DataLoader(
    df_dataset_tokenized_train,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
    num_workers=28
)
eval_dataloader = None
if not args.train_on_full_data:
    eval_dataloader = DataLoader(
        df_dataset_tokenized_eval, collate_fn=data_collator, batch_size=batch_size,
        num_workers=20
    )

from torch.optim import AdamW, Adam

optimizer = Adam(model.parameters(), lr=1e-5)

# %%
from accelerate import Accelerator
import os

accelerator = Accelerator(mixed_precision="fp16")
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
scaler = torch.cuda.amp.GradScaler(enabled=True)

# %%
from transformers import get_scheduler

num_train_epochs = args.num_train_epochs
wandb.config.update({"num_train_epochs": num_train_epochs})
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

label_names_temp = list(label_names.keys())
def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions




seed = args.seed

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))
min_val_loss = np.inf
global_step = 0
for epoch in tqdm(range(num_train_epochs)):
    # Training.
    running_loss = 0.0
    average_running_loss = 0.0
    model.train()
    for batch_step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        running_loss += (loss.item() * batch["input_ids"].size(0))
        average_running_loss += loss.item()
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)
        progress_bar.set_description(f"epoch {epoch} loss: {running_loss / len(train_dataloader.dataset)}")
        global_step += 1

        if batch_step % 100 == 0:
            average_train_loss_per_batch = average_running_loss / (batch_step + 1)
            wandb.log({"average_train_loss_per_batch": average_train_loss_per_batch, "epoch": epoch, "batch_step": batch_step})
            
    
    wandb.log({"epoch_loss": running_loss / len(train_dataloader.dataset), "epoch": epoch})


    # Evaluation only if not training on the full data.
    if not args.train_on_full_data:
        model.eval()
        eval_loss = 0.0
        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                eval_loss += (outputs.loss.item() * batch["input_ids"].size(0))

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]

            # Necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)

            true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
            metric.add_batch(predictions=true_predictions, references=true_labels)
            
        eval_loss /= len(eval_dataloader.dataset)    
        wandb.log({"val_loss": eval_loss, "epoch": epoch}) # Just using the the len(eval_dataloader) is fine because the loss is already averaged over the number of batches.

        results = metric.compute()
        print(
            f"epoch {epoch}:",
            {
                key: results[f"overall_{key}"]
                for key in ["precision", "recall", "f1", "accuracy"]
            },
        )
        # Create a dictionary from the results.
        results_dict = {key: results[f"overall_{key}"] for key in ["precision", "recall", "f1", "accuracy"]}
        
        
        # Log the results to wandb after every epoch.
        wandb.log(results_dict)
        # Save the model checkpoint for each epoch.
        # Save after every five epochs.
        if epoch % 5 == 0:
            model.save_pretrained(f"/home/rsaha/projects/babylm/src/taggers/bert-base-uncased_tagger_checkpoints_seed_{seed}/model_epoch_{epoch}/")
        
        if eval_loss < min_val_loss:
            min_val_loss = eval_loss
            args = {"min_val_loss": min_val_loss, "epoch": epoch}
            
            # Add the results_dict to the args dictionary.
            args.update(results_dict)
            # Save the best model.
            model.save_pretrained(f"/home/rsaha/projects/babylm/src/taggers/bert-base-uncased_tagger_checkpoints_seed_{seed}/best_model/")
            json.dump(args, open(f"/home/rsaha/projects/babylm/src/taggers/bert-base-uncased_tagger_checkpoints_seed_{seed}/best_args.json", "w"))


if args.train_on_full_data:
    # Save the model after training on the whole dataset.
    model.save_pretrained(f"/home/rsaha/projects/babylm/src/taggers/bert-base-uncased_tagger_checkpoints_full_data/seed_{seed}/model_after_training_{num_train_epochs}_epochs/")