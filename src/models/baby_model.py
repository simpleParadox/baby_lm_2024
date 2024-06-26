import torch
import numpy as np
import random
from transformers import get_linear_schedule_with_warmup
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig


import tqdm



class BabyModel:
    
    def __init__(self, args=None, use_cuda=True, cuda_device=-1, wandb_object=None):
        # Initialize the class attributes here
        torch.manual_seed(args['manual_seed'])
        
        self.args.update_from_dict(args) 
        self.wandb_object = wandb_object
        self.args = args
        self.model_name = self.args.model_name
        
        
        # Set seeds here.
        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            # Also setting deterministic behaviour for cudnn.
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            # torch.set_deterministic(True)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)
                

        if use_cuda:
            print("Inside use_cuda")
            if torch.cuda.is_available():
                print(f"GPU Available: {torch.cuda.is_available()}")
                if cuda_device == -1:
                    
                    self.device = torch.device("cuda")
                else:
                    print(f"On the gpu cuda:{cuda_device}")
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set `use_cuda=False`."
                )
        else:
            self.device = "cpu"
            
        self.min_val_loss = float('inf')
        
        # Train a tokenizer.
        # Image processor need not be trained because all it does is apply transformations to the image.
        
        
        
        
        # Load a randomly initialized model here.
        # Make sure that the format is of AutoModelForSequenceClassification or AutoModelForCausalLM
        try:
            # Load a randomly initialized model here.
            # Make sure that the format is of AutoModelForSequenceClassification or AutoModelForCausalLM
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # processor = AutoProcessor.from_pretrained("microsoft/git-base")
            # model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
            
        
        
        
    def train(self, train_dataloader, val_dataloader, method='random', pacing='gaussian', t_total=1000):
        pass