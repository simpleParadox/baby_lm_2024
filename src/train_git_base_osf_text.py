import torch

from datasets.conceptual_captions_processor import ConceptualCaptionsProcessor

from datasets.text_dataset_processor import TextDatasetProcessor

from models.git_base import BabyGitModel

from PIL import Image
import numpy as np

from torch import Tensor


batch_size = 128

dataset_size = 10

text_dataset_processor = TextDatasetProcessor(batch_size=batch_size)


baby_git_model = BabyGitModel()


n_epochs=100

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


def evaluate_model(model: BabyGitModel, captions: list[str]):


    tokenized_captions = model.tokenizer(captions, padding=True, truncation=True, return_tensors="pt").to(device)

    
    model.eval()

    generated_ids = model.model.generate(input_ids=tokenized_captions['input_ids'], attention_mask=tokenized_captions['attention_mask'], max_length=50)

    generated_caption = model.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


    print('generated caption: ', generated_caption)






# NEED TO MAKE BABY_GIT INHERIT FROM NN.MODULE
optimizer = torch.optim.AdamW(baby_git_model.parameters(), lr=5e-5)

baby_git_model.to(device).train()

print('--- CAP before training for "I am a " --- ' )

captions = ['I am a ']

evaluate_model(baby_git_model, captions)





for epoch in range(n_epochs):

    step = 0

    for captions in text_dataset_processor.train_dataloader:

        # print('captions ', captions)


        tokenized_captions = baby_git_model.tokenizer(captions, padding=True, truncation=True, return_tensors="pt").to(device)

        input_ids = tokenized_captions['input_ids'].to(device)
        attention_mask = tokenized_captions['attention_mask'].to(device)

        model_outputs = baby_git_model(input_ids=input_ids, attention_mask=attention_mask)

        loss = model_outputs.loss

        print(f'{step}: loss ', loss)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        step += 1

    

print('-- CAP AFTER TRAINING --')

evaluate_model(baby_git_model, captions)