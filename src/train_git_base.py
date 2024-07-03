import torch



from datasets.conceptual_captions_processor import ConceptualCaptionsProcessor

from models.git_base import BabyGitModel
from transformers import AutoTokenizer


from PIL import Image
import numpy as np

from tqdm import tqdm


batch_size = 2

dataset_processor = ConceptualCaptionsProcessor(batch_size=batch_size)

baby_git_model = BabyGitModel()



n_epochs=2

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def unnormalize_image_for_display(image) -> Image:
    '''
    can do img.show() on returned output
    '''

    MEAN = np.array([123.675, 116.280, 103.530]) / 255
    STD = np.array([58.395, 57.120, 57.375]) / 255

   

    unnormalized_image = (imgs[0].numpy() * np.array(STD)[:, None, None]) + np.array(MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    img = Image.fromarray(unnormalized_image)

    return img


tokenizer = AutoTokenizer.from_pretrained("microsoft/git-base-coco")

# NEED TO MAKE BABY_GIT INHERIT FROM NN.MODULE
optimizer = torch.optim.AdamW(baby_git_model.parameters(), lr=5e-5)

baby_git_model.to(device).train()

for epoch in range(n_epochs):

    for (imgs, captions) in tqdm(dataset_processor.train_dataloader):

        if imgs == None:
            # happens when OSError in conceptual captions dataloader
            continue

        # imgs are preprocessed, captions are not

        tokenized_captions = tokenizer(captions, padding=True, truncation=True, return_tensors="pt").to(device)

        imgs = imgs.to(device)

        

        input_ids = tokenized_captions['input_ids'].to(device)
        attention_mask = tokenized_captions['attention_mask'].to(device)

        model_outputs = baby_git_model(input_ids=input_ids, pixel_values=imgs, attention_mask=attention_mask)

        loss = model_outputs.loss

        print('loss ', loss)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
