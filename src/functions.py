import torch
from PIL import Image
import numpy as np

def unnormalize_image_for_display(image: torch.Tensor) -> Image.Image:
    '''
    can do img.show() on returned output
    '''

    MEAN = np.array([123.675, 116.280, 103.530]) / 255
    STD = np.array([58.395, 57.120, 57.375]) / 255
    unnormalized_image = (image.cpu().numpy() * np.array(STD)[:, None, None]) + np.array(MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    # unnormalized_image = image.cpu().numpy().squeeze(axis=0)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    img = Image.fromarray(unnormalized_image)

    return img

# def evaluate_model_image_caption(model: BabyGitModel, preprocessed_images: torch.Tensor, test_captions: list[str]):

#     tokenized_captions = model.tokenizer(test_captions, padding=True, truncation=True, return_tensors="pt", max_length=args.max_token_length).to(device)
#     img = unnormalize_image_for_display(preprocessed_images[0])
#     img.save(f'test_image_eval.png')
#     preprocessed_images = preprocessed_images.to(device)

    
#     # model.eval()
#     model.model.eval()
#     results = []
#     generated_ids = model.model.generate(pixel_values=preprocessed_images, max_length=80)
#     # generated_ids = model.model.generate(pixel_values=preprocessed_images, max_length=args.max_token_length, input_ids=tokenized_captions['input_ids'], attention_mask=tokenized_captions['attention_mask'])
#     # generated_ids = model.model.generate(pixel_values=res, max_length=args.max_token_length)
#     generated_caption = model.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # return generated_caption, test_captions


def find_best_model_path(top_level_path):
    """
    In the top_level_path, there are multiple folders.
    Each folder has a json file with the validation data min_loss.
    Use this 'min_loss' value inside the json file to find the best model path.
    """
    
    import glob as glob
    import json
    import os
    import numpy as np
    
    min_loss = np.inf
    for folder in glob.glob(top_level_path + '/*/final_model'):
        for file in glob.glob(folder + '/best_args.json'):
            with open(file, 'r') as f:
                data = json.load(f)
                current_loss = data['epoch_loss']
                
                # If the current loss is less than the min_loss, then update the min_loss and the best_model_path.
                if current_loss < min_loss:
                    min_loss = current_loss
                    best_model_path = os.path.dirname(file)
    print("min_loss: ", min_loss)
    
    return best_model_path