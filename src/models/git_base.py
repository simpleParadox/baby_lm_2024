from transformers import GitProcessor, CLIPImageProcessor, PreTrainedTokenizerFast
import torch
import random
import numpy as np
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig
from torch import nn
from transformers import PreTrainedModel
import transformers.models.git.modeling_git as modeling_git

from tokenizers import Tokenizer
from typing import List, Optional, Tuple, Union

from transformers.modeling_outputs import BaseModelOutput

from transformers import PretrainedConfig
import json


import sys
sys.path.append("../../git-2024")
sys.path.append('git-2024')

# add flamingo to the path
sys.path.append("../../flamingo-2024")
sys.path.append('flamingo-2024')

from configuration_flamingo import FlamingoConfig
from configuration_git import GitConfig
from modeling_git import GitForCausalLM, GitForSequenceClassification
from modeling_flamingo import FlamingoForCausalLM, FlamingoForSequenceClassification

# processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
# model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

# url = "https://cdn.mos.cms.futurecdn.net/YMa7Wx2FyjQFUjEeqa72Rm-1200-80.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# pixel_values = processor(images=image, return_tensors="pt").pixel_values

# generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
# generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(generated_caption)

class BabyFlamingoModel(nn.Module):
        
    def __init__(self,
    wandb_object=None,
    manual_seed=42,
    device=None,
    use_dino_embeds=False,
    baseline_causal_lm=False,
    baseline_sequence_classification=False,
    initialize_with_text=False,
    tokenizer_path='./src/tokenizer/multi_50m_and_captions_tokenizer_bert_wordpiece.json',
    text_init_model_path=None,
    load_optimizer=False,
    **kwargs):

        super(BabyFlamingoModel, self).__init__()
        
        self.device = device
        # print("Self device: ", self.device)
        # self.processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")

        # Load custom config.
        # print("Loading preprocessor from babylm config")
        processor_config_path = "/home/rsaha/projects/babylm/flamingo-2024/preprocessor_config.json"
        processor_config = json.load(open(processor_config_path, 'r'))
        self.clip_image_processor = CLIPImageProcessor(processor_config)
        # clip image processor should be ok because its just cropping and transforming the image without a learned network


        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

        # GIT needs predefined pad token
        self.processor = GitProcessor(self.clip_image_processor, self.tokenizer)


        # self.model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

        # print('-- INITIATING MODEL FROM SCRATCH -- ')

        # git_config = GitConfig()

        # Define pretrained_configs here
        # print("Loading from babylm config")
        flamingo_config_path = "/home/rsaha/projects/babylm/flamingo-2024/config.json"
        flamingo_config = FlamingoConfig.from_pretrained(flamingo_config_path)

        # Set the manual seed that will be later used to set the determinism in the modeling_git.py file.
        flamingo_config.manual_seed = manual_seed
        
        if initialize_with_text:
            print("Loading a model that was trained on text only.")
            # First make sure the model with the best min_val_loss is loaded.
            # TODO: Implement this feature.
            assert text_init_model_path is not None, "Please provide a text_init_model_path."
            if baseline_causal_lm and not baseline_sequence_classification:
                self.model_type = "causal_lm"
                self.model = FlamingoForCausalLM.from_pretrained(text_init_model_path, local_files_only=True)
            elif not baseline_causal_lm and baseline_sequence_classification:
                self.model_type = "sequence"
                self.model = FlamingoForSequenceClassification.from_pretrained(text_init_model_path, local_files_only=True)
            
            if load_optimizer:
                print("Loading the optimizer state dict.")
                self.optimizer_state_dict = torch.load(text_init_model_path + "/optim/optimizer.pth")
                
                
        else:
            print("Loading a randomly initialized model.")
            if baseline_causal_lm and not baseline_sequence_classification:
                self.model_type = "causal_lm"
                self.model = FlamingoForCausalLM(config=flamingo_config)
            elif not baseline_causal_lm and baseline_sequence_classification:
                self.model_type = "sequence"
                self.model = FlamingoForSequenceClassification(config=flamingo_config)
            else:
                raise ValueError("Please specify either baseline_git_causal_lm or baseline_git_sequence_classification as True (but not both)")
        

        if use_dino_embeds:
            self.model.git.image_encoder = IdentityVisionModel()
            self.model.git.encoder.layer[0].attention.self.image_patch_tokens = 1
        else:
            pass
    
    
    def forward(self, pixel_values=None, input_ids=None, attention_mask=None) -> CausalLMOutputWithPast:
        # CausalLMOutputWithPast is the direct output of the FlamingoModel (which inherits from AutoModelForCausalLM, which is required by eval of flamingo)
        if pixel_values == None:
            model_outputs = self.model(input_ids=input_ids, labels=input_ids, attention_mask=attention_mask)
        else:
            model_outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids, attention_mask=attention_mask)
        return model_outputs

    def save_model(self, optimizer=None, model_save_path=None):
        assert model_save_path is not None, "Please provide a model_save_path."
        self.model.save_pretrained(model_save_path)
        print(f"Model saved at {model_save_path}")
        
        if optimizer is not None:
            import os
            if os.path.exists(model_save_path + "optim") == False:
                os.makedirs(model_save_path + "optim")
            torch.save(optimizer.state_dict(), model_save_path + "/optim/optimizer.pth")
            print(f"Optimizer saved at {model_save_path}optim/optimizer.pth")
            
            
            

class BabyGitModel(nn.Module):
    
    # def __init__(self, args=None, use_cuda=False, device=None, wandb_object=None, manual_seed=22, use_dino_embeds=False):
    def __init__(self,
        wandb_object=None,
        manual_seed=42,
        device=None,
        use_dino_embeds=False,
        baseline_causal_lm=False,
        baseline_sequence_classification=False,
        initialize_with_text=False,
        tokenizer_path='./src/tokenizer/multi_50m_and_captions_tokenizer_bert_wordpiece.json',
        text_init_model_path=None,
        load_optimizer=False,
        **kwargs):

        super(BabyGitModel, self).__init__()
        # Initialize the class attributes here
        
        
        self.device = device
        # print("Self device: ", self.device)
        # self.processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")

        # Load custom config.
        # print("Loading preprocessor from babylm config")
        processor_config_path = "/home/rsaha/projects/babylm/git-2024/preprocessor_config.json"
        processor_config = json.load(open(processor_config_path, 'r'))
        self.clip_image_processor = CLIPImageProcessor(processor_config)
        # clip image processor should be ok because its just cropping and transforming the image without a learned network


        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

        # GIT needs predefined pad token
        self.processor = GitProcessor(self.clip_image_processor, self.tokenizer)


        # self.model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

        # print('-- INITIATING MODEL FROM SCRATCH -- ')

        # Define pretrained_configs here
        # print("Loading from babylm config")
        git_config_path = "/home/rsaha/projects/babylm/git-2024/config.json"
        git_config = GitConfig.from_pretrained(git_config_path)

        # Set the manual seed that will be later used to set the determinism in the modeling_git.py file.
        git_config.manual_seed = manual_seed
        
        if initialize_with_text:
            print("Loading a model that was trained on text only.")
            # First make sure the model with the best min_val_loss is loaded.
            # TODO: Implement this feature.
            assert text_init_model_path is not None, "Please provide a text_init_model_path."
            if baseline_causal_lm and not baseline_sequence_classification:
                self.model_type = "causal_lm"
                self.model = GitForCausalLM.from_pretrained(text_init_model_path, local_files_only=True)
            elif not baseline_causal_lm and baseline_sequence_classification:
                self.model_type = "sequence"
                self.model = GitForSequenceClassification.from_pretrained(text_init_model_path, local_files_only=True)
            else:
                raise ValueError("Please specify either baseline_git_causal_lm or baseline_git_sequence_classification as True (but not both)")
        
            if load_optimizer:
                print("Loading the optimizer state dict.")
                self.optimizer_state_dict = torch.load(text_init_model_path + "/optim/optimizer.pth")
                
                
        else:
            print("Loading a randomly initialized model.")
            if baseline_causal_lm and not baseline_sequence_classification:
                self.model_type = "causal_lm"
                self.model = GitForCausalLM(config=git_config)
            elif not baseline_causal_lm and baseline_sequence_classification:
                self.model_type = "sequence"
                self.model = GitForSequenceClassification(config=git_config)
            else:
                raise ValueError("Please specify either baseline_git_causal_lm or baseline_git_sequence_classification as True (but not both)")
        


        if use_dino_embeds:
            self.model.git.image_encoder = IdentityVisionModel()
            self.model.git.encoder.layer[0].attention.self.image_patch_tokens = 1
        else:
            pass
            # print("While initializing a model, the image encoder is set to facebook/dino-vitb16 by default.")


        
        self.model.to(self.device)
        # Train a tokenizer.
        # Image processor need not be trained because all it does is apply transformations to the image.
        
        
        # Load a randomly initialized model here.
        # Make sure that the format is of AutoModelForSequenceClassification or AutoModelForCausalLM

    

    
    def forward(self, pixel_values=None, input_ids=None, attention_mask=None) -> CausalLMOutputWithPast:

        # CausalLMOutputWithPast is the direct output of the GitModel (which inherits from AutoModelForCausalLM, which is required by eval of babylm)

        # pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

        # convert images to pixel values in dataloader ig
        
        # The following code block is redundant because in modeling_git.py the forward method supposedly handles this internally.
        # This is because pixel_values has a default value of None.
        # If pixel_values are none, then it won't be conditioned on.
        if self.model_type == "sequence":
            if pixel_values == None:
                model_outputs = self.model(input_ids=input_ids, labels=input_ids, attention_mask=attention_mask)
            else:
                model_outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids, attention_mask=attention_mask)
        
        elif self.model_type == "causal_lm":
            if pixel_values == None:
                model_outputs: CausalLMOutputWithPast = self.model(input_ids=input_ids, labels=input_ids, attention_mask=attention_mask)
            else:
                model_outputs: CausalLMOutputWithPast = self.model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids, attention_mask=attention_mask)

        return model_outputs

        
    def save_model(self, model_save_path):
        self.model.save_pretrained(model_save_path)
        # print(f"Model saved at {model_save_path}")



class IdentityVisionModel(nn.Module):
    def __init__(self):
        super(IdentityVisionModel, self).__init__()



    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:

        '''
        Identity forward
        '''

        # pixel_values is of rank 4. 
        # dino embeddings input in pixel_values would be of shape (1, 1, batch_size, 768)

        outputs = pixel_values[0, 0, :, :]

        # need outputs shape to be (batch_size, 1, 768)

        outputs = outputs.unsqueeze(1)

        # print('outputs shape ', outputs.shape)
        return BaseModelOutput(
            last_hidden_state=outputs,
            hidden_states=None,
            attentions=None,
        )






        
