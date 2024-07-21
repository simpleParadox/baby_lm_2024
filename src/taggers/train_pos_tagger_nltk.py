# %%
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
from nltk.corpus import treebank

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
from pathlib import Path
data_dir = Path("../../data/train_50M_multimodal_clean/")
paths = [str(f) for f in data_dir.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".train"]]
print("Paths: ", paths)
# %%
from nltk.tag import pos_tag
from pathlib import Path

def process_text_file(file_path):
    words = []
    tags = []
    f = open(file_path, 'rb')
    n = sum(1 for _ in f)  # count the number of lines in the file
    print("Total lines in file: ", n)
    f.close()
    pattern = r"\b\w+(?:'\w+)?\b|\b\w+(?:-\w+)*\b|\d+(?:\.\d+)?|\S"  # Only consider the words.
    k = 0
    with open(file_path, 'r') as file:
        for line in file:
            sentence = line.strip()

            # Split the sentence using the refined regex pattern
            tokens = re.findall(pattern, sentence)
            tagged_sentence = pos_tag(tokens)
            # print(tagged_sentence)
            
            for word, tag in tagged_sentence:
                words.append(word)
                tags.append(tag)
            k += 1
            
            print("Completed line {0} out of {1}".format(k, n), end="\r")
    return list(zip(words, tags))

# Example usage:
global_list = []
for file_path in paths:
    # file_path = paths[i]
    print("File path: ", file_path)
    result = process_text_file(file_path)
    global_list.extend(result)

global_list_set = set(global_list)  # Remove duplicates

# %%
# Save the global list to a csv file
# Store global list as a csv file
import pandas as pd
df = pd.DataFrame(global_list_set, columns=["Word", "Tag"])
df.to_csv("pos_tagging_dataset_no_duplicates.csv", index=False, compression='gzip')


df = pd.DataFrame(global_list, columns=["Word", "Tag"])
df.to_csv("pos_tagging_dataset_with_duplicates.csv", index=False, compression='gzip')

# with open("../../data/train_50M_multimodal_clean/pos_tags_all_caption_and_text.txt", "w") as file:
#     for word, tag in global_list:
#         file.write("{0} {1}\n".format(word, tag))
#     file.close()

# %%


# %%
# pos_tag(['I','am','going','to','school', '.'])

# # %%
# # Refined regex pattern
# text = "Here's an example sentence: with numbers 123 and punctuations, hyphens - and more! high-speed and it's 3.14 and example@example.com"

# # Refined regex pattern
# pattern = r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[A-Za-z]+(?:-[A-Za-z]+)*|[0-9]+(?:\.[0-9]+)?|[^\w\s]"

# # Split the sentence using the refined regex pattern
# tokens = re.findall(pattern, text)

# # %%
# features = [extract_features(tokens, i) for i in range(len(tokens))]

# # %%
# features[-2]

# # %%
# tokens


