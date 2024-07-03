from transformers import AutoProcessor, AutoModelForCausalLM
import requests
from PIL import Image
import torch
import random
import numpy as np
from transformers.modeling_outputs import CausalLMOutputWithPast

from torch import nn



# processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
# model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

# url = "https://cdn.mos.cms.futurecdn.net/YMa7Wx2FyjQFUjEeqa72Rm-1200-80.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# pixel_values = processor(images=image, return_tensors="pt").pixel_values

# generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
# generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(generated_caption)


class BabyGitModel(nn.Module):
    
    def __init__(self, args=None, use_cuda=False, cuda_device=-1, wandb_object=None):

        super(BabyGitModel, self).__init__()
        # Initialize the class attributes here
        torch.manual_seed(22)
        
        # self.args.update_from_dict(args) 
        self.wandb_object = wandb_object
        self.args = args
        # self.model_name = self.args.model_name

        # Set seeds here.
        # if self.args.manual_seed:
        if False:
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


        self.processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")

        self.model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

        
        # Train a tokenizer.
        # Image processor need not be trained because all it does is apply transformations to the image.
        
        
        
        
        # Load a randomly initialized model here.
        # Make sure that the format is of AutoModelForSequenceClassification or AutoModelForCausalLM

    def forward(self, pixel_values, input_ids, attention_mask) -> CausalLMOutputWithPast:

        # CausalLMOutputWithPast is the direct output of the GitModel (which inherits from AutoModelForCausalLM, which is required by eval of babylm)

        # pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

        # convert images to pixel values in dataloader ig

        model_outputs: CausalLMOutputWithPast = self.model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids, attention_mask=attention_mask)

        return model_outputs










            
        
        
        
    # def train(self, train_dataloader, val_dataloader, method='random', pacing='gaussian', t_total=1000):
    #     pass