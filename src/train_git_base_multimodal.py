import sys
sys.path.append('../git-2024') # NOTE: Might need to change this according to the user but usually it should be fine.
sys.path.append('../src/datasets')
sys.path.append('/home/rsaha/projects/babylm/src/datasets')
sys.path.append('/home/rsaha/projects/babylm/git-2024')
import torch

from multimodal_dataset_processor import MultiModalDatasetProcessor

from accelerate import Accelerator

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
parser.add_argument('--batch_size', type=int, required=False, default=32)
parser.add_argument('--dataset_size', type=int, required=False, default=-1)
parser.add_argument('--n_epochs', type=int, required=False, default=5)
parser.add_argument('--n_workers', type=int, required=False, default=1)
parser.add_argument('--min_save_every', type=int, required=False, default=1)
parser.add_argument('--seed', type=int, required=False, default=42)
parser.add_argument('--lr', type=float, required=False, default=5e-5)
parser.add_argument('--optimizer', help="adamw, adam or sgd", type=str, required=False, default='adamw')
parser.add_argument('--do_curriculum', type=bool, default=False)  # If this is False, then do standard fine-tuning.
parser.add_argument('--model_type', help="causal or sequence. Case sensitive.", type=str, default='causal_lm')
parser.add_argument('--use_accelerate', type=bool, default=False)  # Whether to use accelerate or not.
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)  # This is only used if use_accelerate is True.
parser.add_argument('--max_token_length', type=int, default=50)
parser.add_argument('--initialize_with_text', type=bool, default=False)


args = parser.parse_args()

batch_size = args.batch_size
dataset_size = args.dataset_size # negative for full dataset
n_epochs=args.n_epochs
n_workers = args.n_workers
# Create accelerator.
if args.use_accelerate:
    accelerator = Accelerator(gradient_accumulation_steps=2)
    device = accelerator.device
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
min_save_every = args.min_save_every # saving best model every fixed number of epochs.
seed = args.seed
lr = args.lr

print("Device: ", device)


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
wandb.log(args_dict) # Log the args.




# Set the output directory with a timestamp.
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
random_dir = np.random.randint(0, 100000)

root_level_path = os.getcwd() + '/'
print("root_level_path: ", root_level_path)
model_save_path = root_level_path + 'saved_models/'

if args.initialize_with_text:
    model_save_path += 'initialize_with_text/'

if args.do_curriculum:
    model_save_path += f'standard/{args.model_type}/seed_{seed}/'
else:
    model_save_path += f'curriculum/{args.model_type}/seed_{seed}/'



model_save_path += f'{timestamp}_{random_dir}/best_model.pth'

print(f'model_save_path: {model_save_path}')
wandb.log({'model_save_path': model_save_path})


# Previously there were the seeds and the deterministic settings here. Now they are in the modeling_git.py file.

baby_git_model = BabyGitModel(use_dino_embeds=False, manual_seed=seed, device=device, 
                              baseline_git_causal_lm=baseline_git_casual_lm, 
                              baseline_git_sequence_classification=baseline_git_sequence_classification, initialize_with_text=args.initialize_with_text)

# baby_git_model = torch.compile(baby_git_model)

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

    tokenized_captions = model.tokenizer(test_captions, padding=True, truncation=True, return_tensors="pt", max_length=args.max_token_length).to(device)
    img = unnormalize_image_for_display(preprocessed_images[0])
    img.save(f'test_image_eval.jpg')
    preprocessed_images = preprocessed_images.to(device)

    
    model.eval()
    results = []
    # generated_ids = model.model.generate(pixel_values=dino_embeds, input_ids=tokenized_captions['input_ids'], attention_mask=tokenized_captions['attention_mask'], max_length=50)
    generated_ids = model.model.generate(pixel_values=preprocessed_images, max_length=args.max_length) 
    generated_caption = model.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption, test_captions


print("-- initializing -- ")
print("Loading optimizer")
optimizer = None
if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(baby_git_model.parameters(), lr=lr)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(baby_git_model.parameters(), lr=lr)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(baby_git_model.parameters(), lr=lr)


baby_git_model.to(device).train()
# print("Model loaded")
# print(baby_git_model)


print("Loading dataset processor")
multimodal_dataset_processor = MultiModalDatasetProcessor(batch_size=batch_size, dataset_size=dataset_size, n_workers=n_workers, device=device)

best_loss = np.inf
last_saved = -1
step = 0  # Global step.
test_images = None
test_captions = None


# full_dataset_length = len(multimodal_dataset_processor.train_dataloader)

# Use the prepare method from accelerator to prepare the pytorch objects.
if args.use_accelerate:
    baby_git_model, optimizer, training_dataloader = accelerator.prepare(
        baby_git_model, optimizer, multimodal_dataset_processor.train_dataloader
    )
else:
    training_dataloader = multimodal_dataset_processor.train_dataloader
    val_dataloader = multimodal_dataset_processor.val_dataloader
    test_dataloader = multimodal_dataset_processor.test_dataloader

# TODO: As of now, no early stopping is implemented. Implement it if needed.
print("-- training -- ")
num_batches = multimodal_dataset_processor.get_num_batches_train()
print("num_batches: ", num_batches)


# @torch.compile
def train_step(baby_git_model, preprocessed_images, optimizer, input_ids, attention_mask):
    model_outputs = baby_git_model(pixel_values=preprocessed_images, input_ids=input_ids, attention_mask=attention_mask)
    loss = model_outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return model_outputs

# train_step = torch.compile(train_step)

val_iterator = tqdm(val_dataloader)
running_loss = 0
epoch_iterator = tqdm(range(n_epochs))
for epoch in epoch_iterator:
    if epoch % 2 == 0:
        print("Validating")
        # evaluate_model(model=baby_git_model, preprocessed_images=preprocessed_images, test_captions=captions)
        baby_git_model.eval()
        print("Eval mode")
        val_loss = 0
        for preprocessed_images, captions in val_iterator:
            tokenized_captions = baby_git_model.tokenizer(captions, padding=True, truncation=True, return_tensors="pt", max_length=args.max_token_length).to(device)
            preprocessed_images = preprocessed_images.to(device)
            model_outputs = train_step(baby_git_model, preprocessed_images, optimizer=optimizer, input_ids=tokenized_captions['input_ids'], attention_mask=tokenized_captions['attention_mask'])
            # model_outputs = baby_git_model(pixel_values=preprocessed_images, input_ids=tokenized_captions['input_ids'], attention_mask=tokenized_captions['attention_mask'])
            loss = model_outputs.loss
            val_loss += loss.item()
            val_iterator.update(1)
        wandb.log({'val_loss': val_loss / len(val_dataloader)})
        print("Validation done.")
        baby_git_model.train()
        print("Train mode")

    
    epoch_loss = 0
    batch_steps = 0
    batch_iterator = tqdm(training_dataloader, disable=False, desc=f'epoch: {epoch}')
    for preprocessed_images, captions in batch_iterator:
        # if test_images == None: # choosing first batch as test data
        #     test_images = preprocessed_images
        #     test_captions = captions
        # print('captions ', captions)

        # with accelerator.accumulate(baby_git_model):
        print("Batch size: ", preprocessed_images.shape[0])
        tokenized_captions = baby_git_model.tokenizer(captions, padding=True, truncation=True, return_tensors="pt", max_length=args.max_token_length).to(device) # TODO: Check if max length is alright.


        input_ids = tokenized_captions['input_ids'].to(device)
        attention_mask = tokenized_captions['attention_mask'].to(device)

        
        # print(f'caps ({step}): ', captions[0])
        # print("preprocessed_image[0] shape ", preprocessed_images[0].shape)
        # image = unnormalize_image_for_display(preprocessed_images[0])
        # image.save(f'test_image_{step}.jpg')
        preprocessed_images = preprocessed_images.to(device)
        # model_outputs = baby_git_model(pixel_values=preprocessed_images, input_ids=input_ids, attention_mask=attention_mask)

        model_outputs = train_step(baby_git_model, preprocessed_images, optimizer=optimizer, input_ids=input_ids, attention_mask=attention_mask)
        loss = model_outputs.loss
        # optimizer = train_step(loss, optimizer)

        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        
        epoch_loss += loss.item()
        running_loss += loss.item()

        # print(f'epoch: {epoch} (step: {step}): loss ', loss)

        batch_iterator.set_description(f'epoch: {epoch} loss: {loss.item()},')
        batch_iterator.update(1)
        # if args.use_accelerate:
        #     accelerator.backward(loss)


        step += 1
        batch_steps += 1

        # Log the loss every 50 steps.
        if step % 50 == 0:
            wandb.log({'step': step, 'loss': epoch_loss / batch_steps})
            wandb.log({'step': step,'running_loss': running_loss / step})

    # Validate two epochs.

    epoch_loss /= batch_steps

    if epoch_loss < best_loss and step - last_saved > min_save_every:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        baby_git_model.save_model(model_save_path)
        last_saved = epoch
        best_loss = epoch_loss

    
    # Print average loss at the end of the epoch.
    epoch_iterator.set_description(f'epoch: {epoch} per_epoch_loss: {epoch_loss / batch_steps}')
    # Log the average loss.
    wandb.log({'epoch': epoch, 'per_epoch_loss': epoch_loss / batch_steps})


# Test model
baby_git_model.eval()
print("Testing")
all_generated_captions = []
print("Test indices: ", multimodal_dataset_processor.test_indices)
for preprocessed_images, captions in test_dataloader:
    generated_caption, true_captions = evaluate_model(model=baby_git_model, preprocessed_images=preprocessed_images, test_captions=captions)
    all_generated_captions.append([generated_caption, true_captions])

# Save in a dataframe.
import pandas as pd
df = pd.DataFrame(all_generated_captions, columns=['generated_caption', 'true_captions'])

# Save as a wandb Table.
wandb.log({'test_results_table': wandb.Table(dataframe=df)})

