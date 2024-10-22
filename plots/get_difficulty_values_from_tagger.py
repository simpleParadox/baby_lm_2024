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
from torch.utils.data import DataLoader
from accelerate import Accelerator
import os
from transformers import DataCollatorForTokenClassification

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Read the .tsv file and store it in a pandas dataframe.
# df = pd.read_csv("/home/rsaha/projects/babylm/src/datasets/multimodal_train/all_multimodal_all_concaps.tsv", sep="\t", compression='gzip')



data_dir = Path("/home/rsaha/projects/babylm/src/taggers/data_for_training/")
# paths = [str(f) for f in data_dir.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".pkl"] and f.name in ["pos_tagging_dataset_all_sentences_cc_3M_captions_non_reduced_filtered.pkl", "pos_tagging_dataset_all_sentences_local_narr_captions.pkl"]]
paths = [str(f) for f in data_dir.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".pkl"] and f.name in ["pos_tagging_dataset_all_captions_for_inference_all_concaps_tsv.pkl"]]
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
    temp_data = pickle.load(open(f"/home/rsaha/projects/babylm/src/taggers/data_for_training/{file_name}.pkl", "rb"))
    data.extend(temp_data)
print("Length of data: ", len(data))

assert len(data) == 3043190 # This is the number of samples in the all_concaps.tsv file.


# Create a dictionary of numbers where each tag (the second element in the tuple) is assigned a unique number. This will be the class labels.
tagdict = nltk_load('help/tagsets/upenn_tagset.pickle')
label_names = {t: i for i, t in enumerate(tagdict.keys())}
label_names['#'] = len(label_names)

# Remove all the empty lists from data.
## data = [d for d in data if d [0]!= []]


tag_to_class_mapping_for_data = []
for i in tqdm(range(len(data))):
    tag_to_class_mapping_for_data.append([label_names[tag] for tag in data[i][1]])

# import pandas as pd
df = pd.DataFrame(data, columns=["sentence", "tags"])
df["class_labels"] = tag_to_class_mapping_for_data

from datasets import Dataset
df_dataset_train = Dataset.from_pandas(df)





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
def tokenize_and_align_labels(examples):
    # print("Example sentence:  ", examples["sentence"])
    tokenized_inputs = tokenizer(
        examples["sentence"], truncation=True, is_split_into_words=True, max_length=50 # NOTE: is_split_into_words=False is important.
    )
    all_labels = examples["class_labels"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        # print("Word IDs: ", word_ids)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

df_dataset_tokenized_train = df_dataset_train.map(tokenize_and_align_labels, batched=True,
                                      remove_columns=df_dataset_train.column_names, num_proc=10)




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


data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
train_dataloader = DataLoader(
    df_dataset_tokenized_train,
    shuffle=False,
    collate_fn=data_collator,
    batch_size=512,
    num_workers=4
)

df_original = pd.read_csv("/home/rsaha/projects/babylm/src/datasets/multimodal_train/all_multimodal_all_concaps.tsv", sep="\t", compression='gzip')

# TODO: Fix the seed for this model.
seed = 2
# Load the POS tagger model and tokenizer.
model = AutoModelForTokenClassification.from_pretrained(f"/home/rsaha/projects/babylm/src/taggers/bert-base-uncased_tagger_checkpoints_full_data/seed_{seed}/model_after_training_5_epochs/")
tokenizer = PreTrainedTokenizerFast.from_pretrained("/home/rsaha/projects/babylm/src/tokenizer/hf_wordpiece_tokenizer_from_bert-base-uncased/")



accelerator = Accelerator(mixed_precision="fp16")
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, None, train_dataloader, None
)

pos_tags = []
pos_labels = []
for batch in tqdm(train_dataloader):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    predictions = outputs.logits.argmax(dim=-1)
    labels = batch["labels"]
    # Necessary to pad predictions and labels for being gathered
    predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
    labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
    predictions_gathered = accelerator.gather(predictions)
    labels_gathered = accelerator.gather(labels)
    true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
    pos_tags.extend(true_predictions)


from joblib import dump, load
dump(pos_tags, f"/home/rsaha/projects/babylm/src/taggers/predicted_pos_tags_from_bert_tagger_captions_only_upenn_all_concaps_seed_{seed}.pkl")

noun_counts = []
for tags in tqdm(pos_tags):
    noun_count = 0
    for tag in tags:
        if tag == "NN" or tag == "NNP":
            noun_count += 1
    noun_counts.append(noun_count)
    
df_original[f"noun_counts_seed_{seed}"] = noun_counts
df_original.to_csv("/home/rsaha/projects/babylm/src/datasets/multimodal_train/all_multimodal_all_concaps_with_pos_tags_with_noun_counts.tsv", sep="\t", index=False)






# # For each item in the pos_tags list, create a new list that stores the corresponding number of nouns (with the NN, and NNP tags).
from joblib import load, dump
pos_tags = load("/home/rsaha/projects/babylm/src/taggers/predicted_pos_tags_from_bert_tagger_captions_only_upenn.pkl")
noun_counts = []
for tags in tqdm(pos_tags):
    noun_count = 0
    for tag in tags:
        if tag == "NN" or tag == "NNP":
            noun_count += 1
    noun_counts.append(noun_count)
dump(noun_counts, "/home/rsaha/projects/babylm/data/noun_counts_difficulty_captions_only_full_data_upenn.pkl")
# Create a histogram of the noun_counts in seaborn.



# Directly load the difficulty scores from the dataframe because the noun_counts are already stored as separate columns in the .tsv file.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
import numpy as np
from tqdm import tqdm
seed = 0
df = pd.read_csv(f"/home/rsaha/projects/babylm/src/datasets/multimodal_train/all_multimodal_all_concaps_with_pos_tags_with_noun_counts_assigned_max_replaced_train_seed_{seed}.tsv", sep="\t")


fontsize = 30
plt.clf()
fig, ax = plt.subplots(figsize=(15, 10))
# plt.hist(noun_counts, bins=22)
sns.histplot(ax=ax, data=df, x=f"noun_counts_seed_{seed}_replaced", discrete=True, cumulative=True)
plt.xlabel("Difficulty: Number of Nouns", fontsize=fontsize)
plt.ylabel("Frequency", fontsize=fontsize)

# Now add the count above each bar in the histogram.
# Create a list to store the frequency of each bin.
noun_counts = df[f"noun_counts_seed_{seed}_replaced"].tolist()

# frequency = []
# for i in range(0, max(noun_counts) + 1):
#     frequency.append(noun_counts.count(i))



# for i in range(0, max(noun_counts) + 1):
#     plt.text(i, max(frequency) - frequency[i], frequency[i], ha='center', va='bottom', rotation=90)


# Draw a vertical line for the 25th, 50th, and 75th percentiles. There is an offset of 0.5 so that it is easier to comprehend.
plt.axvline(x=np.percentile(noun_counts, 25)+0.5, color='r', linestyle='--', label="Q1", linewidth=3)
plt.axvline(x=np.percentile(noun_counts, 50)+0.5, color='black', linestyle='--', label="Q3", linewidth=3)
plt.axvline(x=np.percentile(noun_counts, 75)+0.5, color='b', linestyle='--', label="Q3", linewidth=3)


# Add legends for the vertical lines.
plt.legend(fontsize=fontsize-5)

ax.ticklabel_format(style='sci', axis='y', useOffset=False)
ax.yaxis.get_offset_text().set_fontsize(fontsize-5)
# Add xticks to the histogram.
plt.xticks(range(0, max(noun_counts) + 1, 1), fontsize=20)
plt.yticks(fontsize=fontsize)

plt.title("Difficulty as measured by the number of nouns", fontsize=fontsize)
plt.tight_layout()
plt.savefig(f"/home/rsaha/projects/babylm/difficulty_histogram_nouns_captions_only_bert_pos_tagger_cumulative_assigned_max_replaced_train_seed_{seed}.png")
