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
from joblib import dump, load, Parallel, delayed, Memory
from multiprocessing import Manager

# Load tagged data and process them.

# manager = Manager()
# all_X_data = manager.list()
# all_y_data = manager.list()

memory = Memory(location='/home/rsaha/projects/babylm/src/taggers/processed_tagger_data/cache', verbose=0)

@memory.cache
def load_data(file_name):
    if os.path.exists(f"/home/rsaha/projects/babylm/src/taggers/processed_tagger_data/processed_pos_training_data_{file_name}.pkl"):
        print("Loading data from file ...")
        with open(f"/home/rsaha/projects/babylm/src/taggers/processed_tagger_data/processed_pos_training_data_{file_name}.pkl", "rb") as f:
            total = os.path.getsize(f"/home/rsaha/projects/babylm/src/taggers/processed_tagger_data/processed_pos_training_data_{file_name}.pkl")
            with TQDMBytesReader(f, total=total) as pbfd:
                up = pickle.Unpickler(pbfd)
                X_data, y_data = up.load()
            # X_data, y_data = pickle.load(f)
            return X_data, y_data
        # all_X_data.extend(X_data)
        # all_y_data.extend(y_data)


    else:
        print("Processing and storing data ...")
        data = []
        print("path: ", path)
        with open(path, "rb") as f:
            data.extend(pickle.load(f))
        
        # Transform to dataset.
        X_data, y_data = transform_to_dataset(data)
        # Store the processed data in a pickle file.
        with open(f"/home/rsaha/projects/babylm/src/taggers/processed_tagger_data/processed_pos_training_data_{file_name}.pkl", "wb") as f:
            pickle.dump((X_data, y_data), f)



file_names = []
# Only select the cc_3m and local_narr files and store it in filtered_paths.
filtered_paths = []
for path in paths:
    if "cc_3M" in path or "local_narr" in path:
        filtered_paths.append(path)

for path in tqdm(filtered_paths, desc="Paths"):

    file_name = Path(path).name
    # Drop the .train extension
    file_name = file_name.split(".")[0]
    file_names.append(file_name)


results = Parallel(n_jobs=28, backend='loky', verbose=20)(delayed(load_data)(file_name) for file_name in file_names)
all_X_data = []
all_y_data = []
print("Appending results ...")
for X_data, y_data in tqdm(results):
    all_X_data.extend(X_data)
    all_y_data.extend(y_data)

del results
print("Data loaded successfully.")
print("Length of all_X_data: ", len(all_X_data))    


#Ignoring some warnings for the sake of readability.
import warnings
warnings.filterwarnings('ignore')

# #First, install sklearn_crfsuite, as it is not preloaded into Colab. 
from sklearn_crfsuite import CRF


# # Start initiating the hyperparameter tuning process for nested cross-validation.

params_space = {
    'c1': [0.01, 0.1, 1],
    'c2': [0.01, 0.1, 1]
}

from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report

SEEDS = [0, 1, 2, 3, 4]

cross_val_scores = []

# Define the number of train / val / test samples as a percentage of the total data.
train_size = 0.8
val_size = 0.1
# Define the absolute number of samples for each split.
train_samples = int(train_size * len(all_X_data))
val_samples = int(val_size * len(all_X_data))

all_X_indices = list(range(len(all_X_data)))
all_y_indices = list(range(len(all_y_data)))



for seed in SEEDS:
    print("Seed: ", seed)

    # First divide the dataset into train and test splits.
    # Then from the train split, divide it into train and validation splits.
    # Then train the model on the train split and validate on the validation split.
    
    # First define the model using each combination of hyperparameters from params_space.
    for c1 in params_space['c1']:
        print("c1: ", c1)
        for c2 in params_space['c2']:
            print("c2: ", c2)
            penn_crf = CRF(
                algorithm='lbfgs',
                max_iterations=100,
                all_possible_transitions=False,
                c1=c1,
                c2=c2,
                verbose=5
            )

            # Split the data into train and test splits.
            print("Splitting data ...")
            X_indices, X_test_indices, y_indices, y_test_indices = train_test_split(all_X_indices, all_y_indices, train_size=train_samples, random_state=seed)

            X_train_indices, X_val_indices, y_train_indices, y_val_indices = train_test_split(X_indices, y_indices, test_size=val_samples, random_state=seed)

            data_X_train = [all_X_data[i] for i in tqdm(X_train_indices)]
            data_X_val = [all_X_data[i] for i in tqdm(X_val_indices)]
            data_y_train = [all_y_data[i] for i in tqdm(y_train_indices)]
            data_y_val = [all_y_data[i] for i in tqdm(y_val_indices)]

            # Train the model.
            print("Training the model ...")
            penn_crf.fit(data_X_train, data_y_train)

            # Evaluate the model.
            print("Evaluating the model ...")
            y_pred = penn_crf.predict(data_X_val)

            # Report the classification metrics.
            print("Classification report calculations ... ")
            report = flat_classification_report(y_pred, data_y_val)
            print(report)

            # Store the report in a log file.
            print("Storing logs and models.")
            with open(f"/home/rsaha/projects/babylm/src/taggers/logs/pos_tagger_log_{seed}.txt", "a") as f:
                f.write(report)
                f.write("\n\n")
            
            # Save the model as a joblib file.
            dump(penn_crf, f"/home/rsaha/projects/babylm/src/taggers/models/pos_tagger_model_seed_{seed}_c{}.joblib")