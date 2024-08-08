import sys
sys.path.append('../git-2024') # NOTE: Might need to change this according to the user but usually it should be fine.
sys.path.append('../src/datasets')
sys.path.append('/home/rsaha/projects/babylm/src/datasets')
sys.path.append('/home/rsaha/projects/babylm/git-2024')
import torch
from functions import find_best_model_path
from multimodal_dataset_processor import MultiModalDatasetProcessor
from accelerate import Accelerator
# Load the custom models from the BabyLM challenge.
from models.git_base import BabyGitModel, BabyFlamingoModel
from PIL import Image
import numpy as np
import json
from torch import Tensor
from torch.utils.data import DataLoader
import random
import datetime
from tqdm import tqdm
import os
import wandb
import argparse  # This is necessary for wandb sweeps.

# torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=False, default=32)
parser.add_argument('--dataset_size', type=int, required=False, default=-1)
parser.add_argument('--n_epochs', type=int, required=False, default=1)
parser.add_argument('--n_workers', type=int, required=False, default=28)
parser.add_argument('--min_save_every', type=int, required=False, default=1)
parser.add_argument('--seed', type=int, required=False, default=42)
parser.add_argument('--lr', type=float, required=False, default=1e-5)
parser.add_argument('--optimizer', help="adamw, adam or sgd", type=str, required=False, default='adam')
parser.add_argument('--do_curriculum', type=str, default=False)  # If this is False, then do standard fine-tuning.
parser.add_argument('--model_type', help="causal or sequence. Case sensitive.", type=str, default='causal_lm')
parser.add_argument('--model_name', type=str, default='flamingo')
parser.add_argument('--use_accelerate', type=str, default=False)  # Whether to use accelerate or not.
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)  # This is only used if use_accelerate is True.
parser.add_argument('--max_token_length', type=int, default=50)
parser.add_argument('--initialize_with_text', type=str, default=False)
parser.add_argument('--fp16', type=str, default=True)
parser.add_argument('--tokenizer_path', type=str, default='./src/tokenizer/hf_wordpiece_tokenizer_from_git/')
parser.add_argument('--text_init_model_path', type=str, default=None)
parser.add_argument('--load_optimizer', type=str, default=False)

args = parser.parse_args()

if args.load_optimizer == False or args.load_optimizer == 'False':
    args.load_optimizer = False
else:
    args.load_optimizer = True

if args.do_curriculum == False or args.do_curriculum == 'False':
    args.do_curriculum = False
else:
    args.do_curriculum = True

if args.use_accelerate == False or args.use_accelerate == 'False':
    args.use_accelerate = False
else:
    args.use_accelerate = True

if args.initialize_with_text == False or args.initialize_with_text == 'False':
    args.initialize_with_text = False
else:
    args.initialize_with_text = True

if args.fp16 == False or args.fp16 == 'False':
    args.fp16 = False
else:
    args.fp16 = True

batch_size = args.batch_size
dataset_size = args.dataset_size # negative for full dataset
n_epochs=args.n_epochs
n_workers = args.n_workers
# Create accelerator.
if args.use_accelerate:
    accelerator = Accelerator(gradient_accumulation_steps=2)
    device = accelerator.device
else:
    print("Not using accelerate")
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
min_save_every = args.min_save_every # saving best model every fixed number of epochs.
seed = args.seed
lr = args.lr

print("Device: ", device)


if args.model_type == 'causal_lm':
    baseline_causal_lm = True
    baseline_sequence_classification = False
elif args.model_type == 'sequence':
    baseline_causal_lm = False
    baseline_sequence_classification = True
else:
    raise ValueError('model_type should be either causal_lm or sequence.')

# Initialize wandb.
# wandb.init(project='babylm_2024')
wandb.init(project='babylm_2024', mode='disabled')

# Create dict from args.
args_dict = vars(args)
wandb.log(args_dict) # Log the args.




# Set the output directory with a timestamp.
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
random_dir = np.random.randint(0, 100000)

root_level_path = os.getcwd() + '/'
print("root_level_path: ", root_level_path)
model_save_path = root_level_path + 'saved_models/'

# NOTE: best_text_init_root_path must be assigned first.
if args.do_curriculum:
    if args.initialize_with_text:
        best_text_init_root_path = model_save_path + f'text_only/standard/{args.model_type}/seed_{seed}/'
    model_save_path += f'standard/{args.model_type}/seed_{seed}/'
else:
    if args.initialize_with_text:
        best_text_init_root_path = model_save_path + f'text_only/curriculum/{args.model_type}/seed_{seed}/'
    model_save_path += f'curriculum/{args.model_type}/seed_{seed}/'


if args.initialize_with_text:
    # Find best model path from the best_text_init_root_path.
    args.text_init_model_path = find_best_model_path(best_text_init_root_path)
    
model_save_path += f'{timestamp}_{random_dir}/'

print(f'model_save_path: {model_save_path}')
wandb.log({'model_save_path': model_save_path})




# Previously there were the seeds and the deterministic settings here. Now they are in the modeling_git.py file.
print("Loading model")
if args.model_name == 'git':
    print("Loading a version of Git model")
    baby_model = BabyGitModel(use_dino_embeds=False, manual_seed=seed, device=device, 
                                baseline_causal_lm=baseline_causal_lm, 
                                baseline_sequence_classification=baseline_sequence_classification,
                                initialize_with_text=args.initialize_with_text,
                                tokenizer_path=args.tokenizer_path,
                                text_init_model_path=args.text_init_model_path,
                                load_optimizer=args.load_optimizer)
elif args.model_name == 'flamingo':
    baby_model = BabyFlamingoModel(use_dino_embeds=False, manual_seed=seed, device=device, 
                                baseline_causal_lm=baseline_causal_lm, 
                                baseline_sequence_classification=baseline_sequence_classification,
                                initialize_with_text=args.initialize_with_text,
                                tokenizer_path=args.tokenizer_path,
                                text_init_model_path=args.text_init_model_path,
                                load_optimizer=args.load_optimizer)
else:
    raise ValueError('Other models not supported yet.')

# baby_model = torch.compile(baby_model)



print("-- initializing -- ")
print("Loading optimizer")
optimizer = None


if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(baby_model.parameters(), lr=lr)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(baby_model.parameters(), lr=lr)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(baby_model.parameters(), lr=lr)

if args.load_optimizer:
    optimizer.load_state_dict(baby_model.optimizer_state_dict)

baby_model.to(device).train()
# print("Model loaded")
# print(baby_model)


print("Loading dataset processor")
multimodal_dataset_processor = MultiModalDatasetProcessor(batch_size=batch_size, dataset_size=dataset_size, 
                                                          n_workers=n_workers, device=device,
                                                          processor=baby_model.processor, manual_seed=seed)

best_loss = np.inf
last_saved = -1
step = 0  # Global step.
test_images = None
test_captions = None


# full_dataset_length = len(multimodal_dataset_processor.train_dataloader)

# Use the prepare method from accelerator to prepare the pytorch objects.
if args.use_accelerate:
    baby_model, optimizer, training_dataloader = accelerator.prepare(
        baby_model, optimizer, multimodal_dataset_processor.train_dataloader
    )
else:
    training_dataloader = multimodal_dataset_processor.train_dataloader
    val_dataloader = multimodal_dataset_processor.val_dataloader

# TODO: As of now, no early stopping is implemented. Implement it if needed.

num_batches = multimodal_dataset_processor.get_num_batches_train()
print("num_batches train: ", num_batches)
running_loss = 0

if args.fp16:
    print("Using fp16")
    scaler = torch.cuda.amp.GradScaler(enabled=True)
else:
    scaler = None
# @torch.compile
# def train_step(baby_model, preprocessed_images, optimizer, input_ids, attention_mask):
#     model_outputs = baby_model(pixel_values=preprocessed_images, input_ids=input_ids, attention_mask=attention_mask)
#     loss = model_outputs.loss
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     return model_outputs

# train_step = torch.compile(train_step)
minimum_val_loss = np.inf
epoch_iterator = tqdm(range(n_epochs))
device_autocast = 'cuda' if torch.cuda.is_available() else 'cpu'
train = True
print(f"Train = {train}")
if train:
    print("Training")
    for epoch in epoch_iterator:
        

        
        epoch_loss = 0
        batch_steps = 0
        batch_iterator = tqdm(total=num_batches+1, disable=False, desc=f'epoch: {epoch}')
        for preprocessed_images, captions in training_dataloader:

            optimizer.zero_grad()

            try:
                tokenized_captions = baby_model.tokenizer(text=captions, padding=True, truncation=True, return_tensors="pt", max_length=args.max_token_length).to(device) # TODO: Check if max length is alright.
            except:
                print("Error in tokenizing captions: ", captions)
                continue
            input_ids = tokenized_captions['input_ids'].to(device)
            attention_mask = tokenized_captions['attention_mask'].to(device)

            preprocessed_images = preprocessed_images.to(device)
            
            if args.fp16:
                with torch.autocast(device_type=device_autocast, dtype=torch.float16):
                    model_outputs = baby_model(pixel_values=preprocessed_images, input_ids=input_ids, attention_mask=attention_mask)
                    loss = model_outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                model_outputs = baby_model(pixel_values=preprocessed_images, input_ids=input_ids, attention_mask=attention_mask)
                loss = model_outputs.loss
                loss.backward()
                optimizer.step()

            
            epoch_loss += loss.item()
            running_loss += loss.item()

            batch_iterator.set_description(f'epoch: {epoch} loss: {loss.item()}')
            batch_iterator.update(1)
            step += 1
            batch_steps += 1

            # Log the loss every 50 steps.
            if step % 50 == 0:
                wandb.log({'step': step, 'loss': epoch_loss / batch_steps})
                wandb.log({'step': step,'running_loss': running_loss / step})


        epoch_loss /= batch_steps


        
        # Print average loss at the end of the epoch.
        epoch_iterator.set_description(f'epoch: {epoch} per_epoch_loss: {epoch_loss / batch_steps}')
        # Log the average loss.
        wandb.log({'epoch': epoch, 'per_epoch_loss': epoch_loss / batch_steps})

        if epoch % 1 == 0:
            num_batches_val = multimodal_dataset_processor.get_num_batches_val()
            print("Num batches val: ", num_batches_val)
            val_iterator = tqdm(total=num_batches_val, desc='Validation')
            print("Validating")
            # evaluate_model(model=baby_model, preprocessed_images=preprocessed_images, test_captions=captions)
            baby_model.eval()
            print("Eval mode")
            val_loss = 0
            val_step = 0  # Using this as a divisor to get the average loss.
            for preprocessed_images, captions in val_dataloader:
                try:
                    tokenized_captions = baby_model.tokenizer(captions, padding=True, truncation=True, return_tensors="pt", max_length=args.max_token_length).to(device)
                except:
                    print("Error in tokenizing captions: ", captions)
                    print("Continuing...")
                    continue
                input_ids = tokenized_captions['input_ids'].to(device)
                attention_mask = tokenized_captions['attention_mask'].to(device)
                preprocessed_images = preprocessed_images.to(device)

                if args.fp16:
                    with torch.autocast(device_type=device_autocast, dtype=torch.float16):
                        model_outputs = baby_model(pixel_values=preprocessed_images, input_ids=input_ids, attention_mask=attention_mask)
                        loss = model_outputs.loss
                else:
                    model_outputs = baby_model(pixel_values=preprocessed_images, input_ids=input_ids, attention_mask=attention_mask)
                    loss = model_outputs.loss

                val_loss += loss.item()
                val_iterator.update(1)
                val_step += 1
            current_val_loss = val_loss / val_step
            wandb.log({'val_loss': current_val_loss})
            
            if current_val_loss <= minimum_val_loss:
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                baby_model.save_model(model_save_path)
                print("Model saved.")
                minimum_val_loss = current_val_loss
                wandb.log({'min_val_loss': minimum_val_loss})
                args_dict['min_val_loss'] = current_val_loss
                # Save the args_dict in the same directory as json.
                with open(model_save_path + 'args.json', 'w') as f:
                    json.dump(args_dict, f)
                    print("Args saved.")
            print("Validation done.")
            baby_model.train()
            print("Train mode")


# NOTE: The following blocks is commented out for now because the 
# generate method doesn't really work.
# Test model
# baby_model.eval()
# print("Testing")
# all_generated_captions = []
# # print("Test indices: ", multimodal_dataset_processor.test_indices)
# for preprocessed_images, captions in test_dataloader:
#     generated_caption, true_captions = evaluate_model_image_caption(model=baby_model, preprocessed_images=preprocessed_images, test_captions=captions)
#     all_generated_captions.append([generated_caption, true_captions])

# Save in a dataframe.
# import pandas as pd
# df = pd.DataFrame(all_generated_captions, columns=['generated_caption', 'true_captions'])

# # Save as a wandb Table.
# wandb.log({'test_results_table': wandb.Table(dataframe=df)})