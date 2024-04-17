import torch
import numpy as np
import random
from transformers import get_linear_schedule_with_warmup
import tqdm



class BabyModel:
    
    def __init__(self, model, args=None, use_cuda=True, cuda_device=-1, wandb_object=None):
        # Initialize the class attributes here
        torch.manual_seed(args['manual_seed'])
        
        self.args.update_from_dict(args) 
        self.wandb_object = wandb_object
        self.model_args_dict = args
        self.model = model
        
        
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
        
        
        
    def train(self, train_dataloader, val_dataloader, method='random', pacing='gaussian', t_total=1000):
        
        args = self.args
        model = self.model
        
        model.to(self.device)
        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        # Define the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
        # Define the loss function
        loss_fn = torch.nn.CrossEntropyLoss()
        
        if method == 'random':
            # Do standard pretraining.
            global_step = 0
            for epoch in range(args.num_epochs):
                model.train()
                train_loss = 0.0
                for i, batch in enumerate(train_dataloader):
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    scheduler.step()
                    train_loss += loss.item()
                    
                    global_step += 1
                    
                train_loss /= len(train_dataloader)
                
                val_loss = self.evaluate(val_dataloader, loss_fn)
                
                if val_loss < self.min_val_loss:
                    self.min_val_loss = val_loss
                    torch.save(model.state_dict(), args.save_model_path)
                
                print(f"Epoch: {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}")
                
                if self.wandb_object:
                    self.wandb_object.log({'train_loss': train_loss, 'val_loss': val_loss})
        
        
        elif method == 'curriculum':
            # Implement the curriculum learning here.
            if pacing =='interleaved_curriculum':
                # Do interleaved curriculum.
                
                # First epoch is curriculum, next epoch is anti-curriculum.
                
                model.train()
                
                
                
                # First load the dataset with the scores.
                model.train()
                print("Training using gaussian pacing function")
                batch_iterator = tqdm(range(num_batches), disable=False, mininterval=0)

                for batch_number in batch_iterator:
                    batch_iterator.set_description(f"Batch {batch_number} of {num_batches}, Loss: {current_loss}")
                    
                    

                    batch_of_data = get_batch_from_data(df=train_dataset,batch_number=batch_number, batch_size=args.train_batch_size, total_batches=num_batches, number_of_blocks=num_pacing_blocks, anti=args.indices_ordering)  # Every batch, load a new batch of data using the probability levels.
                    model_train_dataset = self.dataset_class(self.tokenizer, self.args, batch_of_data, self.model_type, raw_data=self.args.load_raw_data)
                    
                    # It is possible that if there are different number of samples for each difficulty levels, then different number of samples from each difficulty level will in the batch.
                    train_dataloader = DataLoader(model_train_dataset,
                                                batch_size=args.train_batch_size,
                                                drop_last=False)

                    for _, batch in enumerate(train_dataloader):
                        inputs = self._get_inputs_dict(batch)
                        outputs = model(**inputs)
                        loss = outputs[0]
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps
                        current_loss = loss.item()
                        loss.backward()
                        tr_loss += loss.item()
                        # if (step + 1) % args.gradient_accumulation_steps == 0:
                        if batch_number % args.gradient_accumulation_steps == 0 or (batch_number == (num_batches - 1)):
                            optimizer.step()
                            if args.use_scheduler:
                                scheduler.step()
                            model.zero_grad()

                    self.wandb_object.log({'Training loss': current_loss, 
                                        'global_step': global_step})
                    global_step += 1
                    
                    if (batch_number % args.eval_after_n_batches == 0) or (batch_number == (num_batches - 1)):
                        if eval_data is not None: # NOTE: While retraining, eval_data is none, so this section isn't called.
                            model.eval()

                            eval_loss = self.evaluate(eval_data, val_dataset)
                            losses_dict = {'val_loss': eval_loss}
                            
                            # If you want to calculate the specific difficulty loss, then do it here.
                            # Values are being logged inside the function.
                            specific_difficulty_level_loss = self.competency_evaluate(eval_data, unique_difficulty_levels, reverse_competency=False, specific_difficulty_level=True, max_dataset_difficulty_level=highest_difficulty_value, specific_difficulty_data_loaders=specific_difficulty_data_loaders)  # This returns a dictionary. The 'anti' argument is not applicable here because the specific difficulty levels are being evaluated anyway.
                            
                            self.wandb_object.log(losses_dict)

                            if eval_loss < self.min_val_loss:
                                self.min_val_loss = eval_loss
                                # Log the min loss to wandb.
                                self.wandb_object.log({'min_val_loss': self.min_val_loss})
                                # Reset the counter because the current validation loss is less than the previous validation loss.
                                self.counter = 0

                                # Store model checkpoint here.
                                print("Output dir: ", output_dir)
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                    
                                model.save_pretrained(output_dir, from_pt=True)
                                
                                # Save the model args.
                                model_args = self.model_args_dict
                                model_args['output_dir'] = output_dir
                                model_args['min_loss'] = self.min_val_loss
                                with open(f'{output_dir}' + 'best_params.json', 'w') as args_file: 
                                    json.dump(model_args, args_file)

                            elif eval_loss > (self.min_val_loss + self.min_delta):
                                # This is the case where the current loss is larger than previous loss + delta.
                                print("Increasing early stopping counter")
                                self.counter += 1
                                if self.counter >= self.early_stopping_patience:
                                    # Stop the training procedure if the condition is satisfied.
                                    break # This is for the outer for loop.
                            # else:
                            #     print("eval_loss doesn't cross min_delta threshold but greater than min_val_loss")

                        model.train()
                        
                    # Set model to train mode just in case.
                    model.train()       
                
    # def train(self, train_dataloader, val_dataloader, method='random'):
        
        
                
            
    def evaluate(self, dataloader, loss_fn):
        model = self.model
        model.eval()
        loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = model(inputs)
                loss += loss_fn(outputs, labels).item()
        
        return loss / len(dataloader)
        
        

# Define your own dataset and dastaloader

dataset = YourCustomDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training progress
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), "trained_model.pth")
