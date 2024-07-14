from transformers import AutoProcessor, AutoModelForCausalLM
import requests
from PIL import Image

import torch

from models.git_base import BabyGitModel

import numpy as np

from torch import Tensor


baby_git_model = BabyGitModel()


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

baby_git_model.to(device)


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)



pixel_values = baby_git_model.processor(images=image, return_tensors="pt").pixel_values.to(device)

captions = ['I am a ']

tokenized_captions = baby_git_model.tokenizer(captions, padding=True, truncation=True, return_tensors="pt").to(device)






# generated_ids = baby_git_model.model.generate(pixel_values=pixel_values, max_length=50)

print('pixel values shape ', pixel_values.shape)

generated_ids = baby_git_model.model.generate(input_ids=tokenized_captions['input_ids'], attention_mask=tokenized_captions['attention_mask'], pixel_values=pixel_values)

generated_caption = baby_git_model.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_caption)



