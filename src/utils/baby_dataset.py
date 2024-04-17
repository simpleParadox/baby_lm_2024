import os
import torch
from torch.utils.data import Dataset
from random import randrange
from pathlib import Path

class LlaMaDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = self._get_file_list()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        text = self._load_text(file_path)
        preprocessed_text = self._preprocess_text(text)
        tensor = self._text_to_tensor(preprocessed_text)
        return tensor

    def _get_file_list(self):
        file_list = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".txt"):
                    file_list.append(os.path.join(root, file))
        return file_list

    def _load_text(self, file_path):
        with open(file_path, "r") as f:
            text = f.read()
        return text

    def _preprocess_text(self, text):
        # Add your preprocessing steps here
        preprocessed_text = text.lower()
        return preprocessed_text

    def _text_to_tensor(self, text):
        # Convert text to tensor using LlaMa-specific method
        tensor = torch.tensor(text)
        return tensor


