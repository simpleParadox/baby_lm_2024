import torch

from datasets.multimodal_dataset_processor import MultiModalDatasetProcessor


from models.git_base import BabyGitModel

from PIL import Image
import numpy as np

from torch import Tensor

from torch.utils.data import DataLoader
import random

from tqdm import tqdm
import os


batch_size = 32

dataset_size = -1 # negative for full dataset

n_epochs=50

n_workers = 24
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

model_save_path = 'src/saved_models/best_model.pt'
min_save_every = 200 # saving best model only if last save was > 200 steps ago

manual_seed = 22


torch.manual_seed(22)

random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
# Also setting deterministic behaviour for cudnn.
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
# torch.set_deterministic(True)
torch.cuda.manual_seed_all(manual_seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

baby_git_model = BabyGitModel(use_dino_embeds=False, manual_seed=manual_seed, device=device)





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



optimizer = torch.optim.AdamW(baby_git_model.parameters(), lr=5e-5)

baby_git_model.to(device).train()

multimodal_dataset_processor = MultiModalDatasetProcessor(batch_size=batch_size, dataset_size=dataset_size, n_workers=n_workers, )



lowest_loss = 9999999 

last_saved = -1

step = 0

test_images = None
test_captions = None

print("-- training -- ")
for epoch in range(n_epochs):

    for preprocessed_images, captions in tqdm(multimodal_dataset_processor.train_dataloader):

        if test_images == None: # choosing first batch as test data
            test_images = preprocessed_images
            test_captions = captions

        # print('captions ', captions)
        tokenized_captions = baby_git_model.tokenizer(captions, padding=True, truncation=True, return_tensors="pt", max_length=50).to(device)

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
            torch.save(baby_git_model.state_dict(), model_save_path)
            last_saved = step

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        step += 1



print('-- EVALUATING GIT MODEL --- ')


baby_git_model.eval()

evaluate_model(model=baby_git_model, preprocessed_images=test_images, test_captions=test_captions)





