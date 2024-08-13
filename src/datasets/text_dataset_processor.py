from tqdm import tqdm
import torch
import random
from torch.utils.data import DataLoader
import sys
sys.path.append('/home/rsaha/projects/babylm/src/datasets')
from dataset_processor_parent import DatasetProcessorParent

from torch.utils.data import Dataset
import numpy as np


import pandas as pd
from pathlib import Path

class TextDatasetProcessor(DatasetProcessorParent):

    def __init__(self, device='cuda:0', batch_size=64,
                 root='./', manual_seed=42,
                 n_workers=20, processor=None, dataset_size=-1, do_val=True) -> None:
        '''
        Create a dataloader using only the non-caption data.
        '''

        self.train_dataset: Dataset = None
        self.train_dataloader = None
        self.val_dataset: Dataset = None
        self.val_dataloader = None
        self.root = root
        self.batch_size = batch_size
        self.val_batch_size = 128
        self.n_workers = n_workers
        self.val_n_workers = 28
        self.manual_seed = manual_seed
        self.train_strings: list[str] = None
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.DATA_ROOT = Path(root)
        self.data_dir = Path("./data/train_50M_multimodal_clean/")

        # Train only the non-caption data. NOTE: This is the non-reduced data because later I realized that all the data is necessary to be included.
        self.train_file_paths: list[str] = [str(f) for f in self.data_dir.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and (f.name not in ["cc_3M_captions_non_reduced_filtered.train", "local_narr_captions.train"])]
        
        self.do_val = do_val
        
        
        # For each path, read the lines.
        # Then, concatenate all the lines into a single list.
        texts = []
        for path in tqdm(self.train_file_paths):
            with open(path, 'r') as f:
                data = f.readlines()
                for line in tqdm(data):
                    texts.append(line)
                    
        self.texts = texts
        
        self.full_range = np.arange(0, len(texts))  # 2851072 is the number of samples in the all_multimodal.tsv file.
        
        if do_val:
            # Set the split ratios
            train_ratio = 0.9  # 90% of the data is used for training

            # Calculate the split indices
            num_samples = len(self.full_range)
            train_end = int(train_ratio * num_samples)

            # Shuffle the indices to ensure randomness
            np.random.seed(manual_seed)
            np.random.shuffle(self.full_range)

            # Split the indices into train, val, and test sets
            self.train_indices = self.full_range[:train_end]
            self.val_indices = self.full_range[train_end:]

            # Convert the indices to lists (optional, as they are already arrays)
            self.train_indices = self.train_indices.tolist()
            self.val_indices = self.val_indices.tolist()

            # Calculate the overlap between the indices. Use sets and find set intersection. the length of the intersection should be 0.
            indices_train_set = set(self.train_indices)
            indices_val_set = set(self.val_indices)

            assert len(indices_train_set.intersection(indices_val_set)) == 0
            print("No overlap between the indices of the train and val sets.")

        else:
            self.train_indices = self.full_range[:].tolist()
            self.val_dataset = None
            self.val_indices = []
        
        self.processor = processor 

        self.create_train_val_dataloaders() # Create train and val dataloaders.
        
        self.load_train_dataset()
        if do_val:
            self.load_val_dataset()
        
    def load_train_dataset(self):
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=self.n_workers, worker_init_fn=self.seed_dataloader_worker, generator=torch.Generator().manual_seed(self.manual_seed))
        
    def load_val_dataset(self):
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.val_batch_size, collate_fn=self.collate_fn, num_workers=self.n_workers, worker_init_fn=self.seed_dataloader_worker, generator=torch.Generator().manual_seed(self.manual_seed))
        
        
    def collate_fn(self, batch) -> list[str]:
        '''
        batch is a list of tuples?
        each tuple is of the form (image, caption)
        image is a jpeg image
        caption is a tuple of strings
        '''


        og_captions: list = batch

        captions = og_captions

        outputs2 = captions

        return outputs2
    
    def get_dataset_length(self, split='train') -> int:
        if split == 'train':
            return len(self.train_indices)
        elif split == 'val':
            return len(self.val_indices)
        else:
            return None
    
    def get_num_batches_train(self) -> int:
        return len(self.train_indices) // self.batch_size
    
    def get_num_batches_val(self) -> int:
        return len(self.val_indices) // self.val_batch_size
    
    @staticmethod
    def seed_dataloader_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    def create_train_val_dataloaders(self):
        
        # The only reason to create this dataframe is to select the rows that are in the train_indices.
        # I could use a list comprehension, but this is just cleaner (could take more memory though).
        df = pd.DataFrame(self.texts, columns=['text'])
        
        # Select the rows that are in the train_indices
        self.train_strings = df.iloc[self.train_indices]['text'].tolist()
        self.train_strings = [x for x in self.train_strings if x]
        self.train_dataset = TextDataset(self.train_strings)
        
        if self.do_val:
            self.val_strings = df.iloc[self.val_indices]['text'].tolist()
            self.val_strings = [x for x in self.val_strings if x]
            self.val_dataset = TextDataset(self.val_strings)
        



    def print_dataset_stats(self):

        print()
        print('--- TRAIN DATASET STATS ---')
        print()
        print('no of train samples: ', len(self.train_dataset))
        print()
        print('--- VAL DATASET STATS ---')
        print()
        print('no of val samples: ', len(self.val_indices))



class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text_list) -> None:
        self.text_list: list[str] = text_list

    def __getitem__(self, idx) -> torch.Tensor:
        return self.text_list[idx]

    def __len__(self):
        return len(self.text_list)
