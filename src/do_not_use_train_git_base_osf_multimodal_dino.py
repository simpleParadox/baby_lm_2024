import torch

from datasets.conceptual_captions_processor import ConceptualCaptionsProcessor

from datasets.text_dataset_processor import TextDatasetProcessor

from datasets.multimodal_dataset_processor import MultiModalDataset

from models.git_base import BabyGitModel

from PIL import Image
import numpy as np

from torch import Tensor

from torch.utils.data import DataLoader
import random


batch_size = 128

dataset_size = 10


baby_git_model = BabyGitModel(use_dino_embeds=True)


n_epochs=500

n_workers = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def unnormalize_image_for_display(image) -> Image:
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



def seed_dataloader_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)



# NEED TO MAKE BABY_GIT INHERIT FROM NN.MODULE
optimizer = torch.optim.AdamW(baby_git_model.parameters(), lr=5e-5)

baby_git_model.to(device).train()

# print('--- CAP before training for "I am a " --- ' )

# captions = ['I am a ']

# evaluate_model(baby_git_model, captions)



multimodal_dataset = MultiModalDataset(dataset_size=dataset_size)



multimodal_dataloader = DataLoader(multimodal_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False, worker_init_fn=seed_dataloader_worker, generator=torch.Generator().manual_seed(22))




state, cap = multimodal_dataset[0]
print(f'[0]: {state.shape}, {cap}')


print('-- CAP BEFORE TRAINING --')

test_embeds = None
test_captions = None

for dino_embeds, caps in multimodal_dataloader:

    test_embeds = dino_embeds

    test_captions = caps

    break

print('dino embeds shape ', dino_embeds.shape)

# print('test caps ', test_captions)


# evaluate_model(baby_git_model, dino_embeds, test_captions)



for epoch in range(n_epochs):

    step = 0

    for dino_states, captions in multimodal_dataloader:

        # print('captions ', captions)
        tokenized_captions = baby_git_model.tokenizer(captions, padding=True, truncation=True, return_tensors="pt").to(device)

        input_ids = tokenized_captions['input_ids'].to(device)
        attention_mask = tokenized_captions['attention_mask'].to(device)

        # dino_states shape now: (batch_size, 768)

        # change dino embeddings to rank 4 since Git expects it that way

        # Handling removal useless dims inside IdentityVisionModel

        dino_states = dino_states.unsqueeze(0).unsqueeze(0).to(device) # shape: (1, 1, batch_size, 768)


        model_outputs = baby_git_model(pixel_values=dino_states, input_ids=input_ids, attention_mask=attention_mask, )

        loss = model_outputs.loss

        print(f'{step}: loss ', loss)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        step += 1

    

print('-- CAP AFTER TRAINING --')

test_embeds = None
test_captions = None


for dino_embeds, caps in multimodal_dataloader:

    test_embeds = dino_embeds

    test_captions = caps

    break




evaluate_model(baby_git_model, dino_embeds, test_captions)