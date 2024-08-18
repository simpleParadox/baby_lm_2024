"""
This script is responsible for creating the dataset for the BERT POS tagger training.
"""
#Regex module for checking alphanumeric values.
import re
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

def transform_to_dataset(tagged_sentences):
  X, y = [], []
  for sentence, tags in tagged_sentences:
    sent_word_features, sent_tags = [],[]
    for index in range(len(sentence)):
      sent_word_features.append(extract_features(sentence, index)),
      sent_tags.append(tags[index])
    X.append(sent_word_features)
    y.append(sent_tags)
  return X, y

# %%
#This cell loads the Penn Treebank corpus from nltk into a list variable named penn_treebank.

#No need to install nltk in google colab since it is preloaded in the environments.
#!pip install nltk
import nltk
nltk.download('treebank')

#Ensure that the treebank corpus is downloaded

#Load the treebank corpus class
# from nltk.corpus import treebank

#Now we iterate over all samples from the corpus (the fileids - that are equivalent to sentences) 
#and retrieve the word and the pre-labeled PoS tag. This will be added as a list of tuples with 
#a list of words and a list of their respective PoS tags (in the same order).
# penn_treebank = []
# for fileid in treebank.fileids():
#   tokens = []
#   tags = []
#   for word, tag in treebank.tagged_words(fileid):
#     tokens.append(word)
#     tags.append(tag)
#   penn_treebank.append((tokens, tags))

# %%

# %%
from nltk.tag import pos_tag
from pathlib import Path

def process_text_file(lines):
    n = len(lines)
    pattern = r"\b\w+(?:'\w+)?\b|\b\w+(?:-\w+)*\b|\d+(?:\.\d+)?|\S"  # Only consider the words.
    k = 0
    sentences_list = []
    for line in lines:
        words = []
        tags = []
        sentence = line.strip()
        # Split the sentence using the refined regex pattern
        tokens = re.findall(pattern, sentence)
        tagged_sentence = pos_tag(tokens)
        for word, tag in tagged_sentence:
            words.append(word)
            tags.append(tag)
        k += 1
        sentences_list.append((words, tags))
        print("Completed line {0} out of {1}".format(k, n), end="\r")
    return sentences_list

# Example usage:
import pandas as pd
import pickle
df = pd.read_csv("/home/rsaha/projects/babylm/src/datasets/multimodal_train/all_multimodal_all_concaps.tsv", sep="\t", compression='gzip')


result = process_text_file(df['caption'].values)

pickle.dump(result, open(f"data/pos_tagging_dataset_all_captions_for_inference_all_concaps_tsv.pkl", "wb"))
