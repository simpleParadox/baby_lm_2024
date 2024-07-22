"""
Train a custom POS tagger using the custom dataset augmented using the penn treebank dataset.
"""

# All the data is stored in the src/taggers/data/ directory.

from pathlib import Path
import os
import numpy as np
data_dir = Path("data/")
paths = [str(f) for f in data_dir.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".pkl"]]
print("Paths: ", paths)




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



# Now load all the data from the pkl files and create a single list.
print("Loading data, please wait ...")
import pickle
from tqdm import tqdm
data = []
for path in tqdm(paths):
  with open(path, "rb") as f:
    data.extend(pickle.load(f))






"""
Create a train, validation, and test split for the data.
"""
full_range = np.arange(len(data))
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Calculate the split indices
num_samples = len(full_range)
train_end = int(train_ratio * num_samples)
val_end = train_end + int(val_ratio * num_samples)

# Shuffle the indices to ensure randomness
np.random.shuffle(full_range)

# Split the indices into train, val, and test sets.
train_indices = full_range[:train_end]
val_indices = full_range[train_end:val_end]
test_indices = full_range[val_end:]

# Convert the indices to lists (optional, as they are already arrays)
train_indices = train_indices.tolist()
val_indices = val_indices.tolist()
test_indices = test_indices.tolist()


# Create the train, val, and test sets
train_data = [data[i] for i in train_indices]
val_data = [data[i] for i in val_indices]
test_data = [data[i] for i in test_indices]

X_train, y_train = transform_to_dataset(train_data)
X_val, y_val = transform_to_dataset(val_data)
X_test, y_test = transform_to_dataset(test_data)


#Ignoring some warnings for the sake of readability.
import warnings
warnings.filterwarnings('ignore')

#First, install sklearn_crfsuite, as it is not preloaded into Colab. 
from sklearn_crfsuite import CRF

#This loads the model. Specifics are: 
#algorithm: methodology used to check if results are improving. Default is lbfgs (gradient descent).
#c1 and c2:  coefficients used for regularization.
#max_iterations: max number of iterations (DUH!)
#all_possible_transitions: since crf creates a "network", of probability transition states,
#this option allows it to map even "connections" not present in the data.
penn_crf = CRF(
    algorithm='lbfgs',
    c1=0.01,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)


# Start initiating the hyperparameter tuning process for nested cross-validation.

params_space = {
    'c1': [0.01, 0.1, 1],
    'c2': [0.01, 0.1, 1]
}

from sklearn.model_selection import GridSearchCV

SEEDS = [0, 1, 2, 3, 4]

for seed in SEEDS:
    penn_crf = CRF(
        algorithm='lbfgs',
        max_iterations=100,
        all_possible_transitions=True
    )
    penn_crf.fit(X_train, y_train)
    y_pred = penn_crf.predict(X_val)
    print("Seed: ", seed)
    print("Validation accuracy: ", metrics.flat_accuracy_score(y_val, y_pred))
    print("Validation F1 score: ", metrics.flat_f1_score(y_val, y_pred, average='weighted'))


#The fit method is the default name used by Machine Learning algorithms to start training.
print("Started training on Penn Treebank corpus!")
penn_crf.fit(X_train, y_train)
print("Finished training on Penn Treebank corpus!")