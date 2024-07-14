'''
Abstract class for dataset processors
'''
from abc import ABC, abstractmethod
import torch
import clip
import random
import wandb
from sklearn.linear_model import LogisticRegression

import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# add sibling directory to path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))# def 




class DatasetProcessorParent(ABC):

    def __init__(self) -> None:
        self.val_dataset: torch.utils.data.Dataset = None
        self.classes: list = None
        
        self.device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        _, self.preprocess = clip.load(wandb.config['openai_clip_model'], device=self.device)
        # set seed
        torch.manual_seed(wandb.config['seed'])
        random.seed(wandb.config['seed'])
        # self.load_train_dataloader() # SKIPPING this for now to speed up validation
        self.load_val_dataloader()

        self.name: str = None
        self.keyname: str = None
        pass

    @abstractmethod
    def load_train_dataset(self) -> torch.utils.data.Dataset:
        pass

    @abstractmethod
    def load_val_dataset(self) -> torch.utils.data.Dataset:
        pass

    def load_val_dataloader(self) -> None:
        self.load_val_dataset()
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=wandb.config['batch_size'], shuffle=True, num_workers=wandb.config['zero_shot_acc_num_workers'], persistent_workers=True)

        pass

    def load_train_dataloader(self) -> None:
        self.load_train_dataset()
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=wandb.config['batch_size'], shuffle=True, num_workers=wandb.config['num_workers'])
        pass

    def print_dataset_stats(self):

        print(f'{self.name} dataset stats')
        print('num classes ', len(self.classes))
        print('classes ', self.classes)
        print('num val samples ', len(self.val_dataset))
        # print('num train samples ', len(self.train_dataset)) # SKIPPING this for now to speed up validation

    def get_num_batches(self) -> int:
        pass

    def get_accuracy(self, linear_classifier: LogisticRegression, all_val_features: torch.FloatTensor, all_val_labels: list) -> float: 
        '''
        Default Accuracy metric is top 1, so its implemented in dataset parent
        '''

        accuracy = linear_classifier.score(all_val_features, all_val_labels)

        return accuracy
        
        


