import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from transformers import AutoModelForTokenClassification, PreTrainedTokenizerFast
from tqdm import tqdm
from pathlib import Path
from nltk.data import load as nltk_load
import pickle
import evaluate
from torch.data.utils import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Read the .tsv file and store it in a pandas dataframe.
df = pd.read_csv("/home/rsaha/projects/babylm/src/datasets/multimodal_train/all_multimodal_all_concaps.tsv", sep="\t", compression='gzip')

# Load the POS tagger model and tokenizer.
model = AutoModelForTokenClassification.from_pretrained("/home/rsaha/projects/babylm/src/taggers/bert-base-uncased_tagger_checkpoints_full_data/seed_0/model_after_training_5_epochs/")
tokenizer = PreTrainedTokenizerFast.from_pretrained("/home/rsaha/projects/babylm/src/tokenizer/hf_wordpiece_tokenizer_from_bert-base-uncased/")

# Put model to device.
model.to(device)

model.eval()

data_dir = Path("/home/rsaha/projects/babylm/src/taggers/data/")
paths = [str(f) for f in data_dir.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".pkl"] and f.name in ["pos_tagging_dataset_all_sentences_cc_3M_captions_non_reduced_filtered.pkl", "pos_tagging_dataset_all_sentences_local_narr_captions.pkl"]]
print("Number of files: ", len(paths))


file_names = []


for path in tqdm(paths, desc="Paths"):

    file_name = Path(path).name
    # Drop the .train extension
    file_name = file_name.split(".")[0]
    file_names.append(file_name)

# Load the data from each file_name in file_names.
data = []
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

import pandas as pd
df = pd.DataFrame(data, columns=["sentence", "tags"])
df["class_labels"] = tag_to_class_mapping_for_data



train_df = df

from datasets import Dataset
df_dataset_train = Dataset.from_pandas(train_df)





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
                           max_length=5, 
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




train_dataloader = DataLoader(
    df_dataset_tokenized_train,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
    num_workers=28
)

id2label = {i: label for i, label in enumerate(label_names.keys())}
label2id = {v: k for k, v in id2label.items()}  # This is nothing but the label_names dictionary. But keeping it like this for consistency with the Kaggle notebook.



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



pos_tags = []
for example in tqdm(df_dataset_tokenized_train):
    inputs = example['input_ids']
    attention_mask = example['attention_mask']
    with torch.no_grad():
        outputs = model(inputs.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
    predicted_labels = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    pos_tags.append([id2label[label] for label in predicted_labels])