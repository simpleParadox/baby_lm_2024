import torch

from datasets.conceptual_captions_processor import ConceptualCaptionsProcessor

from models.git_base import BabyGitModel

from PIL import Image
import numpy as np

from torch import Tensor


batch_size = 128

dataset_size = 10

dataset_processor = ConceptualCaptionsProcessor(batch_size=batch_size, dataset_size=dataset_size)

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


# NEED TO MAKE BABY_GIT INHERIT FROM NN.MODULE
optimizer = torch.optim.AdamW(baby_git_model.parameters(), lr=5e-5)

baby_git_model.to(device).train()

test_preprocessed_image: Tensor = None

test_caption: str = None

for (imgs, caps) in dataset_processor.train_dataloader:
    test_preprocessed_image = imgs[2]
    test_caption = caps[2]

print('--- CAP before training --- ' )

  
baby_git_model.eval()

generated_ids = baby_git_model.model.generate(pixel_values=test_preprocessed_image.unsqueeze(0).to(device), max_length=50)

generated_caption = baby_git_model.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


print('generated caption: ', generated_caption)

print("original caption: ", test_caption)

test_image_unnormalized = unnormalize_image_for_display(test_preprocessed_image)

test_image_unnormalized.save('test_image.jpg')



for epoch in range(n_epochs):

    step = 0

    for (imgs, captions) in dataset_processor.train_dataloader:

        if imgs == None:
            # happens when OSError in conceptual captions dataloader
            continue

        # imgs are preprocessed, captions are not

        tokenized_captions = baby_git_model.tokenizer(captions, padding=True, truncation=True, return_tensors="pt").to(device)

        imgs = imgs.to(device)


        

        input_ids = tokenized_captions['input_ids'].to(device)
        attention_mask = tokenized_captions['attention_mask'].to(device)

        model_outputs = baby_git_model(input_ids=input_ids, pixel_values=imgs, attention_mask=attention_mask)

        loss = model_outputs.loss

        print(f'{step}: loss ', loss)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        step += 1

    


# test inference
    
baby_git_model.eval()

generated_ids = baby_git_model.model.generate(pixel_values=test_preprocessed_image.unsqueeze(0).to(device), max_length=50)

generated_caption = baby_git_model.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


print('-- CAP AFTER TRAINING --')
print('generated caption ', generated_caption)