import sys
sys.path.append('../git-2024') # NOTE: Might need to change this according to the user but usually it should be fine.
sys.path.append('../src/datasets')
sys.path.append('/home/rsaha/projects/babylm/src/datasets')
sys.path.append('/home/rsaha/projects/babylm/git-2024')

sys.path.append('/home/rsaha/projects/baby_lm_2024/src/datasets')
sys.path.append('/home/rsaha/projects/baby_lm_2025/git-2024')


from text_dataset_processor import TextDatasetProcessor
import torch
from models.git_base import BabyGitModel
import numpy as np
import torch
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
import json

# torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=False, default=256)
parser.add_argument('--dataset_size', type=int, required=False, default=-1)
parser.add_argument('--n_epochs', type=int, required=False, default=20)
parser.add_argument('--n_workers', type=int, required=False, default=20)
parser.add_argument('--min_save_every', type=int, required=False, default=1)
parser.add_argument('--seed', type=int, required=False, default=42)
parser.add_argument('--lr', type=float, required=False, default=1e-5)
parser.add_argument('--optimizer', help="adamw, adam or sgd", type=str, required=False, default='adam')
parser.add_argument('--do_curriculum', type=str, default=False)  # If this is False, then do standard fine-tuning.
parser.add_argument('--model_type', help="causal or sequence. Case sensitive.", type=str, default='causal_lm')
parser.add_argument('--use_accelerate', type=str, default=False)  # Whether to use accelerate or not.
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)  # This is only used if use_accelerate is True.
parser.add_argument('--max_token_length', type=int, default=50)
parser.add_argument('--initialize_with_text', type=str, default=False)
parser.add_argument('--model_name', type=str, default='flamingo')
parser.add_argument('--fp16', type=str, default=True)
parser.add_argument('--tokenizer_path', type=str, default='./src/tokenizer/hf_wordpiece_tokenizer_from_bert-base-uncased/')
parser.add_argument('--text_init_model_path', type=str, default=None)
parser.add_argument('--load_optimizer', type=str, default=False)
parser.add_argument('--train_on_full_data', type=str, default=True, help="Whether to train on the full data or not. If provided, the model will be trained on the full data.") # Default value is False.
parser.add_argument('--text_only_training', type=int, required=False, default=1)  # This is always True for this script.
parser.add_argument('--wandb_mode', type=str, default='online', required=False)
args = parser.parse_args()

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

# For training only with the text part, this must always be set to False.
if args.initialize_with_text == False or args.initialize_with_text == 'False':
    args.initialize_with_text = False
else:
    args.initialize_with_text = True
assert args.initialize_with_text == False, "initialize with text must be set to False for text_only training."

if args.fp16 == False or args.fp16 == 'False':
    args.fp16 = False
else:
    args.fp16 = True

# Check tokenizer path for correct model.
if args.model_name == 'git':
    assert args.tokenizer_path == './src/tokenizer/hf_wordpiece_tokenizer_from_bert-base-uncased/'
elif args.model_name == 'flamingo':
    assert args.tokenizer_path == './src/tokenizer/hf_wordpiece_tokenizer_from_bert-base-uncased/'


batch_size = args.batch_size
dataset_size = args.dataset_size # negative for full dataset
n_epochs=args.n_epochs
n_workers = args.n_workers
# Create accelerator.
if args.use_accelerate:
    accelerator = Accelerator(gradient_accumulation_steps=1)
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
# NOTE: This initialize_with_text condition is redundant because this is text_only_training.
# and the assertion is always False.
# if args.initialize_with_text:
#     model_save_path += 'initialize_with_text/'

if args.train_on_full_data:
    model_save_path += 'full_data/'
    
if not args.do_curriculum:
    # NOTE: This is not args.do_curriculum (which is standard i.i.d training).
    model_save_path += f'text_only/standard/{args.model_type}/seed_{seed}/'
else:
    model_save_path += f'text_only/curriculum/{args.model_type}/seed_{seed}/'

model_save_path += f'{timestamp}_{random_dir}/'  # Important because of hyperparameter tuning.




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


baby_model.to(device).train()
# print("Model loaded")
# print(baby_model)
do_val = True
if args.train_on_full_data:
    print("Because train_on_full_data is True, setting do_val to False.")
    do_val = False

text_dataset_processor = TextDatasetProcessor(batch_size=batch_size, dataset_size=dataset_size, 
                                                          n_workers=n_workers, device=device,
                                                          processor=baby_model.processor, manual_seed=seed,
                                                          do_val=do_val)

best_loss = np.inf
last_saved = -1
test_images = None
test_captions = None

training_dataloader = text_dataset_processor.train_dataloader

if do_val:
    val_dataloader = text_dataset_processor.val_dataloader

# TODO: As of now, no early stopping is implemented. Implement it if needed.

num_batches = text_dataset_processor.get_num_batches_train()
print("num_batches train: ", num_batches)

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
min_val_loss = np.inf
global_step = 0
epoch_iterator = tqdm(range(n_epochs))
device_autocast = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Training")
for epoch in epoch_iterator:
    # Training.
    running_loss = 0.0
    average_running_loss = 0.0
    batch_iterator = tqdm(total=num_batches+1, disable=False, desc=f'epoch: {epoch}')
    for batch_step, text_data in enumerate(training_dataloader):

        optimizer.zero_grad()

        try:
            tokenized_data = baby_model.tokenizer(text=text_data, padding=True, truncation=True, return_tensors="pt", max_length=args.max_token_length).to(device) # TODO: Check if max length is alright.
        except:
            print("Error in tokenizing text data: ", text_data)
            continue
        input_ids = tokenized_data['input_ids'].to(device)
        attention_mask = tokenized_data['attention_mask'].to(device)
        
        if args.fp16:
            with torch.autocast(device_type=device_autocast, dtype=torch.float16):
                model_outputs = baby_model(input_ids=input_ids, attention_mask=attention_mask)
                loss = model_outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            model_outputs = baby_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = model_outputs.loss
            loss.backward()
            optimizer.step()

        
        running_loss += (loss.item() * input_ids.size(0))
        average_running_loss += loss.item()

        batch_iterator.set_description(f'epoch: {epoch} loss: {loss.item()}')
        batch_iterator.update(1)
        global_step += 1

        # Log the loss every 50 steps.
        if batch_step % 100 == 0:
            average_train_loss_per_batch = average_running_loss / (batch_step + 1)
            wandb.log({"average_train_loss_per_batch": average_train_loss_per_batch, "epoch": epoch, "batch_step": batch_step})
            
    
    epoch_loss = running_loss / text_dataset_processor.get_dataset_length('train')
    wandb.log({"epoch_loss": epoch_loss, "epoch": epoch})

    best_args = {'epoch': epoch, 'epoch_loss': epoch_loss}
    # Save the args_dict in the same directory as json.
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    with open(model_save_path + 'best_args.json', 'w') as f:
        json.dump(best_args, f)
        print("Args saved.")
    # Print average loss at the end of the epoch.
    epoch_iterator.set_description(f'epoch: {epoch} per_epoch_loss: {epoch_loss}')
    # Log the average loss.
    
    if args.train_on_full_data:
        # Save the model every 5 epochs.
        if (epoch+1) % 5 == 0:
            epoch_path = model_save_path + f'epoch_{epoch+1}/'
            if not os.path.exists(epoch_path):
                os.makedirs(epoch_path)
            baby_model.save_model(epoch_path)
            print("Model saved at epoch: ", epoch)
        

    if not args.train_on_full_data:
        if epoch % min_save_every == 0:
            num_batches_val = text_dataset_processor.get_num_batches_val()
            print("Num batches val: ", num_batches_val)
            val_iterator = tqdm(total=num_batches_val, desc='Validation')
            print("Validating")
            baby_model.eval()
            print("Eval mode")
            eval_loss = 0.0
            for text_data in tqdm(val_dataloader):
                try:
                    tokenized_data = baby_model.tokenizer(text_data, padding=True, truncation=True, return_tensors="pt", max_length=args.max_token_length).to(device)
                except:
                    print("Error in tokenizing text data: ", text_data)
                    print("Continuing...")
                    continue
                input_ids = tokenized_data['input_ids'].to(device)
                attention_mask = tokenized_data['attention_mask'].to(device)

                if args.fp16:
                    with torch.autocast(device_type=device_autocast, dtype=torch.float16):
                        model_outputs = baby_model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = model_outputs.loss
                else:
                    model_outputs = baby_model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = model_outputs.loss
                eval_loss += (loss.item() * input_ids.size(0))
                val_iterator.update(1)
                
            current_val_loss = eval_loss / text_dataset_processor.get_dataset_length('val')
            wandb.log({'val_loss': current_val_loss})
            
            if current_val_loss <= min_val_loss:
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                baby_model.save_model(model_save_path)
                print("Model saved.")
                min_val_loss = current_val_loss
                wandb.log({'min_val_loss': min_val_loss})
                args_dict['min_val_loss'] = current_val_loss
                # Save the args_dict in the same directory as json.
                with open(model_save_path + 'best_args.json', 'w') as f:
                    json.dump(args_dict, f)
                    print("Args saved.")
            print("Validation done.")
            baby_model.train()
            print("Train mode")

if args.train_on_full_data:
    model_save_path = model_save_path + 'final_model/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    baby_model.save_model(model_save_path)
    print("Model saved.")