import sys
sys.path.append('../git-2024') # NOTE: Might need to change this according to the user but usually it should be fine.
sys.path.append('../src/datasets')
sys.path.append('/home/rsaha/projects/babylm/src/datasets')
sys.path.append('/home/rsaha/projects/babylm/git-2024')
import torch

from multimodal_dataset_processor import MultiModalDatasetProcessor

from models.git_base import BabyGitModel

from PIL import Image
import numpy as np

from torch import Tensor

from torch.utils.data import DataLoader
import random
import datetime
from tqdm import tqdm
import os

import wandb

from modeling_git import GitForCausalLM as BaselineGitForCausalLM # Make sure git-2024 is clone 'inside' the root directory of the project.
from modeling_git import GitForSequenceClassification as BaselineGitForSequenceClassification


import argparse  # This is necessary for wandb sweeps.



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dataset_size', type=int, default=-1)
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--n_workers', type=int, default=24)
parser.add_argument('--min_save_every', type=int, default=200)
parser.add_argument('--seed', type=int, default=22)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--optimizer', help="adamw or adam", type=str, default='adam')
parser.add_argument('--do_curriculum', type=bool, default=False)  # If this is False, then do standard fine-tuning.
parser.add_argument('--model_type', help="causal or sequence. Case sensitive.", type=str, default='causal_lm')


args = parser.parse_args()

batch_size = args.batch_size
dataset_size = args.dataset_size # negative for full dataset
n_epochs=args.n_epochs
n_workers = args.n_workers
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
min_save_every = args.min_save_every # saving best model only if last save was > 200 steps ago
seed = args.seed
lr = args.lr


if args.model_type == 'causal_lm':
    baseline_git_casual_lm = True
    baseline_git_sequence_classification = False
elif args.model_type == 'sequence':
    baseline_git_casual_lm = False
    baseline_git_sequence_classification = True
else:
    raise ValueError('model_type should be either causal_lm or sequence.')

# Initialize wandb.
wandb.init(project='babylm_2024')

# Create dict from args.
args_dict = vars(args)
wandb.log(args_dict)


# Set the output directory with a timestamp.
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
random_dir = np.random.randint(0, 100000)

root_level_path = '../'
model_save_path = root_level_path + 'saved_models/'

if args.do_curriculum:
    model_save_path += f'standard/{args.model_type}/seed_{seed}/'
else:
    model_save_path += f'curriculum/{args.model_type}/seed_{seed}/'


model_save_path += f'{timestamp}_{random_dir}/'


# Previously there were the seeds and the deterministic settings here. Now they are in the modeling_git.py file.

baby_git_model = BabyGitModel(use_dino_embeds=False, manual_seed=seed, device=device, 
                              baseline_git_causal_lm=baseline_git_casual_lm, 
                              baseline_git_sequence_classification=baseline_git_sequence_classification)

def unnormalize_image_for_display(image: torch.Tensor) -> Image.Image:
    '''
    can do img.show() on returned output
    '''

    MEAN = np.array([123.675, 116.280, 103.530]) / 255
    STD = np.array([58.395, 57.120, 57.375]) / 255

   

    unnormalized_image = (image.numpy() * np.array(STD)[:, None, None]) + np.array(MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    img = Image.fromarray(unnormalized_image)

    return img


def evaluate_model(model: BabyGitModel, preprocessed_images: torch.Tensor, test_captions: list[str]):


    tokenized_captions = model.tokenizer(test_captions, padding=True, truncation=True, return_tensors="pt", max_length=50).to(device)
    img = unnormalize_image_for_display(preprocessed_images[0])
    img.save(f'test_image_eval.jpg')
    preprocessed_images = preprocessed_images.to(device)

    
    model.eval()
    # generated_ids = model.model.generate(pixel_values=dino_embeds, input_ids=tokenized_captions['input_ids'], attention_mask=tokenized_captions['attention_mask'], max_length=50)
    generated_ids = model.model.generate(pixel_values=preprocessed_images, max_length=50) 
    generated_caption = model.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print('generated caption: ', generated_caption)
    print('true caption ', test_captions[0])



optimizer = torch.optim.AdamW(baby_git_model.parameters(), lr=lr)
baby_git_model.to(device).train()
multimodal_dataset_processor = MultiModalDatasetProcessor(batch_size=batch_size, dataset_size=dataset_size, n_workers=n_workers)

lowest_loss = 9999999

last_saved = -1

step = 0  # Global step.

test_images = None
test_captions = None

print("-- training -- ")
for epoch in range(n_epochs):
    for preprocessed_images, captions in tqdm(multimodal_dataset_processor.train_dataloader):
        if test_images == None: # choosing first batch as test data
            test_images = preprocessed_images
            test_captions = captions
        # print('captions ', captions)
        tokenized_captions = baby_git_model.tokenizer(captions, padding=True, truncation=True, return_tensors="pt", max_length=50).to(device) # TODO: Check if max length is alright.
        input_ids = tokenized_captions['input_ids'].to(device)
        attention_mask = tokenized_captions['attention_mask'].to(device)
        # print(f'caps ({step}): ', captions[0])
        # print("preprocessed_image[0] shape ", preprocessed_images[0].shape)
        # image = unnormalize_image_for_display(preprocessed_images[0])
        # image.save(f'test_image_{step}.jpg')
        preprocessed_images = preprocessed_images.to(device)
        model_outputs = baby_git_model(pixel_values=preprocessed_images, input_ids=input_ids, attention_mask=attention_mask)
        loss = model_outputs.loss
        print(f'epoch: {epoch} (step: {step}): loss ', loss)
        if loss.item() < lowest_loss and step - last_saved > min_save_every:
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(baby_git_model.state_dict(), model_save_path)
            last_saved = step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1


print('-- EVALUATING GIT MODEL --- ')
baby_git_model.eval()
evaluate_model(model=baby_git_model, preprocessed_images=test_images, test_captions=test_captions)
