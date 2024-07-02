import torch



from datasets.conceptual_captions_processor import ConceptualCaptionsProcessor

from models.git_base import BabyGitModel
from transformers import AutoTokenizer


from PIL import Image
import numpy as np


dataset_processor = ConceptualCaptionsProcessor()

baby_git_model = BabyGitModel()

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

for (imgs, captions) in dataset_processor.train_dataloader:

    if imgs == None:
        # happens when OSError in conceptual captions dataloader
        continue

    # imgs are preprocessed, captions are not

    tokenized_captions = tokenizer(captions, padding=True, truncation=True, return_tensors="pt").to(device)

    input_ids = tokenized_captions['input_ids']
    attention_mask = tokenized_captions['attention_mask']

    model_outputs = baby_git_model(input_ids=input_ids, pixel_values=imgs, attention_mask=attention_mask)

    loss = model_outputs.loss

    print('loss ', loss)
