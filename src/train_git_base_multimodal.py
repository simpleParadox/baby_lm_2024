import torch

from datasets.multimodal_dataset_processor import MultiModalDatasetProcessor


from models.git_base import BabyGitModel

from PIL import Image
import numpy as np

from torch import Tensor

from torch.utils.data import DataLoader
import random

from tqdm import tqdm


batch_size = 32

dataset_size = -1


baby_git_model = BabyGitModel(use_dino_embeds=False)


n_epochs=500

n_workers = 24

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

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


def evaluate_model(model: BabyGitModel, dino_embeds: torch.Tensor, test_captions: list[str]):


    tokenized_captions = model.tokenizer(test_captions, padding=True, truncation=True, return_tensors="pt", max_length=50).to(device)

    
    model.eval()

    # take only first image
    dino_embeds = dino_embeds[0].unsqueeze(0) # shape: (1, 768)

    dino_embeds = dino_embeds.unsqueeze(0).unsqueeze(0).to(device) # shape: (1, 1, batch_size, 768)

    # generated_ids = model.model.generate(pixel_values=dino_embeds, input_ids=tokenized_captions['input_ids'], attention_mask=tokenized_captions['attention_mask'], max_length=50)
    generated_ids = model.model.generate(pixel_values=dino_embeds, max_length=50) 

    generated_caption = model.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]




    print('generated caption: ', generated_caption)

    print('true caption ', test_captions[0])



# NEED TO MAKE BABY_GIT INHERIT FROM NN.MODULE
optimizer = torch.optim.AdamW(baby_git_model.parameters(), lr=5e-5)

baby_git_model.to(device).train()

# print('--- CAP before training for "I am a " --- ' )

# captions = ['I am a ']

# evaluate_model(baby_git_model, captions)



multimodal_dataset_processor = MultiModalDatasetProcessor(batch_size=batch_size, dataset_size=dataset_size, n_workers=n_workers)



print("-- training -- ")
for epoch in range(n_epochs):

    step = 0

    print('stepping')

    for preprocessed_images, captions in tqdm(multimodal_dataset_processor.train_dataloader):

        # print("one step")

        # print('captions ', captions)
        tokenized_captions = baby_git_model.tokenizer(captions, padding=True, truncation=True, return_tensors="pt").to(device)

        input_ids = tokenized_captions['input_ids'].to(device)
        attention_mask = tokenized_captions['attention_mask'].to(device)
        

        # print('caps[0] ', captions[0])

        # print("preprocessed_image[0] shape ", preprocessed_images[0].shape)


        # image = unnormalize_image_for_display(preprocessed_images[0])

        # image.save('test_image.jpg')

        preprocessed_images = preprocessed_images.to(device)

        model_outputs = baby_git_model(pixel_values=preprocessed_images, input_ids=input_ids, attention_mask=attention_mask)

        loss = model_outputs.loss

        print(f'{step}: loss ', loss)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        step += 1

