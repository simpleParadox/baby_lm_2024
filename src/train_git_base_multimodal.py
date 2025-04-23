import sys
sys.path.append('../git-2024') # NOTE: Might need to change this according to the user but usually it should be fine.
sys.path.append('../src/datasets')
sys.path.append('/home/rsaha/projects/babylm/src/datasets')
sys.path.append('/home/rsaha/projects/babylm/git-2024')

sys.path.append('/home/rsaha/projects/baby_lm_2024/src/datasets')
sys.path.append('/home/rsaha/projects/baby_lm_2024/git-2024')


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




parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=False, default=32)
parser.add_argument('--dataset_size', type=int, required=False, default=-1)
parser.add_argument('--n_epochs', type=int, required=False, default=8)
parser.add_argument('--n_workers', type=int, required=False, default=5)
parser.add_argument('--min_save_every', type=int, required=False, default=1)
parser.add_argument('--seed', type=int, required=False, default=0)
parser.add_argument('--lr', type=float, required=False, default=1e-5)
parser.add_argument('--optimizer', help="adamw, adam or sgd", type=str, required=False, default='adam')
parser.add_argument('--do_curriculum', type=str, default=False)  # If this is False, then do standard training.
parser.add_argument('--model_type', help="causal_lm or sequence. Case sensitive.", type=str, default='causal_lm')
parser.add_argument('--model_name', type=str, default='git')
parser.add_argument('--use_accelerate', type=str, default=False)  # Whether to use accelerate or not.
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)  # This is only used if use_accelerate is True.
parser.add_argument('--max_token_length', type=int, default=50)
parser.add_argument('--initialize_with_text', type=str, default=False)
parser.add_argument('--fp16', type=str, default=True)
parser.add_argument('--tokenizer_path', type=str, default='./src/tokenizer/hf_wordpiece_tokenizer_from_bert-base-uncased/')
parser.add_argument('--text_init_model_path', type=str, default=None)
parser.add_argument('--load_optimizer', type=str, default=False)
parser.add_argument('--train_on_full_data', type=str, default=True, help="Whether to train on the full data or not. If provided, the model will be trained on the full data.") # Default value is False.
parser.add_argument('--do_val', type=str, default=True, help="Whether to do validation or not.")
parser.add_argument('--wandb_mode', type=str, default='disabled', required=False)
parser.add_argument('--unfreeze_vision_encoder', type=str, default=False, help="Whether to unfreeze the vision encoder for flamingo.") # Default value is False.

args = parser.parse_args()

if args.do_val == False or args.do_val == 'False':
    args.do_val = False
else:
    args.do_val = True

if args.train_on_full_data == False or args.train_on_full_data == 'False':
    args.train_on_full_data = False
else:
    args.train_on_full_data = True

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



if args.unfreeze_vision_encoder == False or args.unfreeze_vision_encoder == 'False':
    args.unfreeze_vision_encoder = False
else:
    args.unfreeze_vision_encoder = True


# Must use this tokenizer for both GIT and Flamingo.
assert args.tokenizer_path == './src/tokenizer/hf_wordpiece_tokenizer_from_bert-base-uncased/'

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

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Initialize wandb.
wandb.init(project='babylm_2024', mode=args.wandb_mode)

# Create dict from args.
args_dict = vars(args)
wandb.log(args_dict) # Log the args.




# Set the output directory with a timestamp.
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
random_dir = np.random.randint(0, 100000)

root_level_path = os.getcwd() + '/'
print("root_level_path: ", root_level_path)
model_save_path = root_level_path + 'saved_models/'


if args.model_name == 'flamingo':
    model_save_path += 'flamingo/'
elif args.model_name == 'git':
    model_save_path += 'git/'


if args.train_on_full_data:
    model_save_path += 'full_data/'
else:
    model_save_path += 'not_full_data/'




# NOTE: best_text_init_root_path must be assigned first. The ordering is important.

if args.initialize_with_text:
    best_text_init_root_path = model_save_path + f'text_only/standard/causal_lm/seed_{seed}/' # NOTE: For both CausalLM and Sequence classification, it must be causal_lm because for seq class you're loading the causal_lm model.
    model_save_path += 'initialize_with_text/'
    args.text_init_model_path = find_best_model_path(best_text_init_root_path)

model_save_path += 'image_caption/'

if not args.do_curriculum:
    model_save_path += f'standard/{args.model_type}/seed_{seed}/'
else:
    model_save_path += f'curriculum/{args.model_type}/seed_{seed}/'


model_save_path += f'{timestamp}_{random_dir}/'

print(f'model_save_path: {model_save_path}')
wandb.log({'model_save_path': model_save_path})


args_dict = vars(args)
wandb.log(args_dict) # Log the args.



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
                                load_optimizer=args.load_optimizer, unfreeze_vision_encoder=args.unfreeze_vision_encoder)
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


# Calculate and print model parameters
total_params = sum(p.numel() for p in baby_model.parameters())
trainable_params = sum(p.numel() for p in baby_model.parameters() if p.requires_grad)
non_trainable_params = total_params - trainable_params

print(f"--- Model Summary ---")
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Non-Trainable Parameters: {non_trainable_params:,}")
print(f"---------------------")
import pdb; pdb.set_trace()
# print("Model loaded")
# print(baby_model)

do_val = args.do_val

print("Loading dataset processor")
multimodal_dataset_processor = MultiModalDatasetProcessor(batch_size=batch_size, dataset_size=dataset_size, 
                                                          n_workers=n_workers, device=device,
                                                          processor=baby_model.processor, manual_seed=seed, do_val=do_val, do_curriculum=args.do_curriculum)

best_loss = np.inf
last_saved = -1


# full_dataset_length = len(multimodal_dataset_processor.train_dataloader)

# Use the prepare method from accelerator to prepare the pytorch objects.
if not args.do_curriculum:
    if args.use_accelerate:
        baby_model, optimizer, training_dataloader = accelerator.prepare(
            baby_model, optimizer, multimodal_dataset_processor.train_dataloader
        )
    else:
        training_dataloader = multimodal_dataset_processor.train_dataloader
        if do_val:
            print("Loading val dataloader")
            val_dataloader = multimodal_dataset_processor.val_dataloader
else:
    curriculum_dataloaders = multimodal_dataset_processor.curriculum_dataloaders  # This is a list.
    if do_val:
        print("Loading val dataloader")
        val_dataloader = multimodal_dataset_processor.val_dataloader

if not args.do_curriculum:
    num_batches = multimodal_dataset_processor.get_num_batches_train()
    print("num_batches train: ", num_batches)
else:
    curriculum_num_batches = multimodal_dataset_processor.get_num_batches_train()
    print("curriculum_num_batches: ", curriculum_num_batches)
running_loss = 0

if args.fp16:
    print("Using fp16")
    scaler = torch.cuda.amp.GradScaler(enabled=True)
else:
    scaler = None


# train_step = torch.compile(train_step)
min_val_loss = np.inf
global_step = 0
epoch_iterator = tqdm(range(n_epochs))
device_autocast = 'cuda' if torch.cuda.is_available() else 'cpu'
train = True
print(f"Train = {train}")
print("Training")
if not args.do_curriculum:
    print("Standard training.")
    
    for epoch in epoch_iterator:
        # Training.
        running_loss = 0.0
        average_running_loss = 0.0
        batch_step = 0
        number_of_samples_per_epoch = 0 # This is set to zero every epoch.
        failed_batches = 0
        batch_iterator = tqdm(total=num_batches+1, disable=False, desc=f'epoch: {epoch}')
        for preprocessed_images, captions in training_dataloader:
            
            optimizer.zero_grad()
            try:
                tokenized_captions = baby_model.tokenizer(text=captions, padding=True, truncation=True, return_tensors="pt", max_length=args.max_token_length).to(device) # TODO: Check if max length is alright.
                number_of_samples_per_epoch += len(captions)
            except:
                print("Error in tokenizing captions: ", captions)
                failed_batches += 1
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

            running_loss += (loss.item() * input_ids.size(0)) # This will change based on the batch size, which could be less than 32.
            average_running_loss += loss.item()  # This is already divided by the batch size.

            batch_iterator.set_description(f'epoch: {epoch} loss: {loss.item()}')
            batch_iterator.update(1)
            global_step += 1

            # Log the loss every 100 steps.
            if batch_step % 100 == 0:
                average_train_loss_per_batch = average_running_loss / (batch_step + 1)  # This is correct - if in doubt, think about it. It's the average batch loss.
                wandb.log({"average_train_loss_per_batch": average_train_loss_per_batch, "epoch": epoch, "batch_step": batch_step})
            
            batch_step += 1

        # Epoch loss over the whole dataset.
        epoch_loss = running_loss / number_of_samples_per_epoch  # This is good because if the batch size changes during training, it will be accounted for here.
        wandb.log({"epoch_loss": epoch_loss, "epoch": epoch})
        
        # Log the failed batches.
        wandb.log({"failed_batches": failed_batches, "epoch": epoch})
        
        best_args = {f'epoch_{epoch}_loss': epoch_loss}
        # Save the args_dict in the same directory as json.
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        with open(model_save_path + 'best_args.json', 'w') as f:
            json.dump(best_args, f)
            print("Args saved.")
        
        if args.train_on_full_data:
            # Save the model every min_save_every epochs.
            if (epoch) % min_save_every == 0:
                epoch_path = model_save_path + f'epoch_{epoch}/'
                if not os.path.exists(epoch_path):
                    os.makedirs(epoch_path)
                baby_model.save_model(epoch_path)
                print("Model saved at epoch: ", epoch)
        
        
        if args.do_val:
            if epoch % 1 == 0:
                num_batches_val = multimodal_dataset_processor.get_num_batches_val()
                print("Num batches val: ", num_batches_val)
                val_iterator = tqdm(total=num_batches_val, desc='Validation')
                print("Validating")
                # evaluate_model(model=baby_model, preprocessed_images=preprocessed_images, test_captions=captions)
                baby_model.eval()
                print("Eval mode")
                eval_loss = 0.0
                failed_val_batches = 0
                number_of_samples_per_epoch_val = 0
                for preprocessed_images, captions in tqdm(val_dataloader):
                    try:
                        tokenized_captions = baby_model.tokenizer(captions, padding=True, truncation=True, return_tensors="pt", max_length=args.max_token_length).to(device)
                        number_of_samples_per_epoch_val += len(captions)
                    except:
                        print("Error in tokenizing captions: ", captions)
                        failed_val_batches += 1
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

                    eval_loss += (loss.item() * input_ids.size(0))
                    val_iterator.update(1)
                wandb.log({"failed_val_batches": failed_val_batches, "epoch": epoch})
                current_val_loss = eval_loss / number_of_samples_per_epoch_val
                wandb.log({'val_loss': current_val_loss})
                if current_val_loss <= min_val_loss:
                    print("Model saved.")
                    min_val_loss = current_val_loss
                    best_args['min_val_loss'] = current_val_loss
                    best_args['best_min_val_loss_epoch'] = epoch
                
                    # Save the best_args in the same directory as json.
                    if not os.path.exists(model_save_path):
                        os.makedirs(model_save_path)
                    with open(model_save_path + 'args.json', 'w') as f:
                        json.dump(best_args, f)
                        print("Args saved.")
                print("Validation done.")
                baby_model.train()
                print("Train mode")

else:
    print("Doing curriculum learning.")
    for epoch in epoch_iterator:
        # Training. The idea is that based on the epoch number, decide on the data loader to iterate over.
        running_loss = 0.0
        average_running_loss = 0.0
        batch_step = 0
        number_of_samples_per_epoch = 0 # This is set to zero every epoch.
        failed_batches = 0
        # Select the correct dataloader based on the epoch.
        if epoch < 2:
            training_dataloader = curriculum_dataloaders[0]
            batch_iterator = tqdm(total=curriculum_num_batches[0], disable=False, desc=f'epoch: {epoch}')
        elif epoch >= 2 and epoch < 4:
            training_dataloader = curriculum_dataloaders[1]
            batch_iterator = tqdm(total=curriculum_num_batches[1], disable=False, desc=f'epoch: {epoch}')
        elif epoch >= 4 and epoch < 6:
            training_dataloader = curriculum_dataloaders[2]
            batch_iterator = tqdm(total=curriculum_num_batches[2], disable=False, desc=f'epoch: {epoch}')
        else:
            training_dataloader = curriculum_dataloaders[3]
            batch_iterator = tqdm(total=curriculum_num_batches[3], disable=False, desc=f'epoch: {epoch}')

        for preprocessed_images, captions in training_dataloader:
            
            optimizer.zero_grad()
            try:
                tokenized_captions = baby_model.tokenizer(text=captions, padding=True, truncation=True, return_tensors="pt", max_length=args.max_token_length).to(device) # TODO: Check if max length is alright.
                number_of_samples_per_epoch += len(captions)
            except:
                print("Error in tokenizing captions: ", captions)
                failed_batches += 1
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

            running_loss += (loss.item() * input_ids.size(0))
            average_running_loss += loss.item()

            batch_iterator.set_description(f'epoch: {epoch} loss: {loss.item()}')
            batch_iterator.update(1)
            global_step += 1

            # Log the loss every 100 steps.
            if batch_step % 100 == 0:
                average_train_loss_per_batch = average_running_loss / (batch_step + 1)
                wandb.log({"average_train_loss_per_batch": average_train_loss_per_batch, "epoch": epoch, "batch_step": batch_step})
            
            batch_step += 1
            
        epoch_loss = running_loss / number_of_samples_per_epoch
        wandb.log({"epoch_loss": epoch_loss, "epoch": epoch})
        
        wandb.log({"failed_batches": failed_batches, "epoch": epoch})
        
        best_args = {f'epoch_{epoch}_loss': epoch_loss}
        # Save the args_dict in the same directory as json.
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        with open(model_save_path + 'best_args.json', 'w') as f:
            json.dump(best_args, f)
            print("Args saved.")
        
        if args.train_on_full_data:
            # Save the model every min_save_every epochs.
            if (epoch) % min_save_every == 0:
                epoch_path = model_save_path + f'epoch_{epoch}/'
                if not os.path.exists(epoch_path):
                    os.makedirs(epoch_path)
                baby_model.save_model(epoch_path)
                print("Model saved at epoch: ", epoch)
        if args.do_val:
            if epoch % 1 == 0:
                num_batches_val = multimodal_dataset_processor.get_num_batches_val()
                print("Num batches val: ", num_batches_val)
                val_iterator = tqdm(total=num_batches_val, desc='Validation')
                print("Validating")
                baby_model.eval()
                print("Eval mode")
                eval_loss = 0.0
                failed_val_batches = 0
                number_of_samples_per_epoch_val = 0
                for preprocessed_images, captions in tqdm(val_dataloader):
                    try:
                        tokenized_captions = baby_model.tokenizer(captions, padding=True, truncation=True, return_tensors="pt", max_length=args.max_token_length).to(device)
                        number_of_samples_per_epoch_val += len(captions)
                    except:
                        print("Error in tokenizing captions: ", captions)
                        failed_val_batches += 1
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

                    eval_loss += (loss.item() * input_ids.size(0))
                    val_iterator.update(1)
                wandb.log({"failed_val_batches": failed_val_batches, "epoch": epoch})
                current_val_loss = eval_loss / number_of_samples_per_epoch_val
                wandb.log({'val_loss': current_val_loss})
                if current_val_loss <= min_val_loss:
                    print("Model saved.")
                    min_val_loss = current_val_loss
                    best_args['min_val_loss'] = current_val_loss
                    best_args['best_min_val_loss_epoch'] = epoch
                
                    # Save the best_args in the same directory as json.
                    if not os.path.exists(model_save_path):
                        os.makedirs(model_save_path)
                    with open(model_save_path + 'args.json', 'w') as f:
                        json.dump(best_args, f)
                        print("Args saved.")
                print("Validation done.")
                baby_model.train()
                print("Train mode")

if args.train_on_full_data: # This is after th if-else block because we want it to happen for both cases.
    model_save_path = model_save_path + f'final_model/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    # Save the optimizer.
    torch.save(optimizer.state_dict(), f"{model_save_path}optimizer_after_training_{n_epochs}_epochs.pth")