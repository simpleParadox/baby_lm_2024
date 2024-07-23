"""
Train a custom POS tagger using the custom dataset augmented using the penn treebank dataset.
"""

# All the data is stored in the src/taggers/data/ directory.

from pathlib import Path
import os
import numpy as np
data_dir = Path("/home/rsaha/projects/babylm/src/taggers/data/")
paths = [str(f) for f in data_dir.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".pkl"]]
print("Paths: ", paths)

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
  for sentence, tags in tqdm(tagged_sentences):
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

# Load tagged data and process them.
all_X_data = []
all_y_data = []
for path in tqdm(paths):

    file_name = Path(path).name
    # Drop the .train extension
    file_name = file_name.split(".")[0]


    if os.path.exists(f"/home/rsaha/projects/babylm/src/taggers/processed_tagger_data/processed_pos_training_data_{file_name}.pkl"):
        with open(f"/home/rsaha/projects/babylm/src/taggers/processed_tagger_data/processed_pos_training_data_{file_name}.pkl", "rb") as f:
            total = os.path.getsize(f"/home/rsaha/projects/babylm/src/taggers/processed_tagger_data/processed_pos_training_data_{file_name}.pkl")
            with TQDMBytesReader(f, total=total) as pbfd:
                up = pickle.Unpickler(pbfd)
                X_data, y_data = up.load()
                all_X_data.extend(X_data)
                all_y_data.extend(y_data)
            
        print("Data loaded from file.")
        print("Length of X_data: ", len(X_data))
        print("Length of y_data: ", len(y_data))
    else:
        data = []
        print("path: ", path)
        with open(path, "rb") as f:
            data.extend(pickle.load(f))
        
        # Transform to dataset.
        X_data, y_data = transform_to_dataset(data)
        # Store the processed data in a pickle file.
        with open(f"/home/rsaha/projects/babylm/src/taggers/processed_tagger_data/processed_pos_training_data_{file_name}.pkl", "wb") as f:
            pickle.dump((X_data, y_data), f)


#Ignoring some warnings for the sake of readability.
import warnings
warnings.filterwarnings('ignore')

# #First, install sklearn_crfsuite, as it is not preloaded into Colab. 
from sklearn_crfsuite import CRF

penn_crf = CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True
)


# # Start initiating the hyperparameter tuning process for nested cross-validation.

params_space = {
    'c1': [0.01, 0.1, 1],
    'c2': [0.01, 0.1, 1]
}

from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_val_score

SEEDS = [0, 1, 2, 3, 4]

cross_val_scores = []

for seed in SEEDS:
    print("Seed: ", seed)
    outer_cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=seed)  # Five splits of 90% train, 10% test.
    clf = GridSearchCV(penn_crf, params_space, cv=ShuffleSplit(n_splits=5, test_size=0.1, random_state=seed), verbose=5, n_jobs=-1)  # Will do 5 fold cv on the training set (90% of the data).
    cross_val_scores.append(cross_val_score(clf, X_data, y_data, cv=outer_cv, n_jobs=-1).mean())

print(f"Test scores on {len(SEEDS)} seeds: ", cross_val_scores)