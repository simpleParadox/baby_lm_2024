import clip
import torch
import random
from torch.utils.data import DataLoader, Subset
from datasets.dataset_processor_parent import DatasetProcessorParent

from torch.utils.data import Dataset
import numpy as np

import torchdata.datapipes as dp

import aiohttp
from PIL import Image
import io
from typing import Optional
from torchdata.datapipes.iter import IterDataPipe
# import Generator
from typing import Generator, List, Tuple, Sequence
import asyncio

from torch import Tensor

from pathlib import Path

class TextDatasetProcessor(DatasetProcessorParent):

    def __init__(self, cuda_device='cuda:0', batch_size=64, dataset_size=-1, root='src/datasets/osf/text_data') -> None:

        '''
        Outputs captions as list of original UNTokenized captions
        '''

        self.train_dataset: Dataset = None
        self.train_dataloader = None
        self.val_dataset: Dataset = None
        self.val_dataloader = None

        self.train_data_pipe: IterDataPipe = None
        self.val_data_pipe: IterDataPipe = None

        self.root = root

        self.dataset_size = dataset_size
        
        self.batch_size = batch_size

        file = 'asd.asd.asd'.split()

        self.train_strings: list[str] = None

        self.device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")

        self.train_root = Path(root + '/train_100M/')

        self.train_file_paths: list[str] = [str(f) for f in self.train_root.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".train"]]

        self.val_root = Path(root + '/dev/')

        self.val_file_paths = [str(f) for f in self.val_root.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".dev"]]



    
        # always need to first load train then load val dataset. Fix this confusing requirement later
        self.load_train_dataset()
        self.load_val_dataset()


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


    def get_num_batches(self) -> int:

        return 3318333 // self.batch_size
        return len(self.train_dataloader)
    
    @staticmethod
    def seed_dataloader_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)




    def load_train_dataset(self):

        batch_size = self.batch_size

        self.train_strings = [open(path).read().split('\n') for path in self.train_file_paths]

        

        

        self.train_strings: list[str] = sum(self.train_strings, [])

        # remove empty strings
        self.train_strings = [x for x in self.train_strings if x]

        self.train_dataset = TextDataset(self.train_strings)


        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, collate_fn=self.collate_fn, num_workers=0, worker_init_fn=self.seed_dataloader_worker, generator=torch.Generator().manual_seed(22))

    def load_val_dataset(self):

        self.val_strings = [open(path).read().split('\n') for path in self.val_file_paths]

        self.val_strings: list[str] = sum(self.val_strings, [])

        self.val_dataset = TextDataset(self.val_strings)

        


    def print_dataset_stats(self):

        print()
        print('--- TRAIN DATASET STATS ---')
        print()

        print('no of train samples: ', 3318333)

        print()
        print('--- VAL DATASET STATS ---')
        print()


        print('no of val samples: ', 15840)





class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text_list) -> None:
        self.text_list: list[str] = text_list

    def __getitem__(self, idx) -> torch.Tensor:
        return self.text_list[idx]

    def __len__(self):
        return len(self.text_list)
