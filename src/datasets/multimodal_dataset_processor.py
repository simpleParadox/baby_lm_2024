import clip
import torch
import random
from torch.utils.data import DataLoader, Subset
from datasets.dataset_processor_parent import DatasetProcessorParent

from torch.utils.data import Dataset
import numpy as np

import json

import torchdata.datapipes as dp

import aiohttp
from PIL import Image
import io
from typing import Optional
from torchdata.datapipes.iter import IterDataPipe
# import Generator
from typing import Generator, List, Tuple, Sequence

from bisect import bisect

import numpy as np

from torch import Tensor

from pathlib import Path




class MultiModalDataset(torch.utils.data.Dataset):


    def __init__(self, root: str = 'src/datasets/osf/multimodal_data', dataset_size=-1) -> None:

        self.root = root
        # self.cc_3m_dino_states_file_names = ['cc_3M_dino_v2_states_1of2.npy', 'cc_3M_dino_v2_states_2of2.npy']

        # self.local_narr_dino_states_file_name = 'local_narr_dino_v2_states.npy'

        self.cc_3m_caption_file_name = 'cc_3M_captions.json'

        self.local_narr_captions_file_name = 'local_narr_captions.json'

        self.dino_state_paths = ['cc_3M_dino_v2_states_1of2.npy', 'cc_3M_dino_v2_states_2of2.npy', 'local_narr_dino_v2_states.npy']

        self.dino_state_paths = [f'{root}/{path}' for path in self.dino_state_paths]

        self.caption_paths = ['cc_3M_captions.json', 'local_narr_captions.json']

        self.caption_paths = [f'{root}/{path}' for path in self.caption_paths]


        self.dino_embeddings_memmaps = [np.load(path, mmap_mode='r') for path in self.dino_state_paths]

        # from https://stackoverflow.com/questions/60127632/load-multiple-npy-files-size-10gb-in-pytorch

    
        self.dino_state_start_indices = [0] * len(self.dino_state_paths)

        self.dataset_size = dataset_size

        self.data_count = 0

        self.caption_start_indices = [0] * len(self.caption_paths)

        for index, memmap in enumerate(self.dino_embeddings_memmaps):
            self.dino_state_start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]

        self.captions = [json.loads(open(path).read()) for path in self.caption_paths]

        self.captions = sum(self.captions, [])


        if dataset_size > 0:
            self.data_count = dataset_size

        


    def __getitem__(self, index) -> Tuple[torch.Tensor, str]:

        memmap_index = bisect(self.dino_state_start_indices, index) - 1
        index_in_memmap = index - self.dino_state_start_indices[memmap_index]
        data: np.ndarray = self.dino_embeddings_memmaps[memmap_index][index_in_memmap]
        # target = self.target_memmaps[memmap_index][index_in_memmap]

        # shape of returned element = (768)

        return torch.from_numpy(data.copy()), self.captions[index]


    def __len__(self):
        return self.data_count
