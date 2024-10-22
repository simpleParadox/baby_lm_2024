from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import requests
from PIL import Image

processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

# Load the tokenizer.
tokenizer = AutoTokenizer.from_pretrained("microsoft/git-base-coco")

# Save the model in a directory.
model.save_pretrained("/home/rsaha/projects/babylm/pretrained_from_hf/pretrained_git_ms_model/")
tokenizer.save_pretrained("/home/rsaha/projects/babylm/pretrained_from_hf/pretrained_git_ms_model/")
processor.save_pretrained("/home/rsaha/projects/babylm/pretrained_from_hf/pretrained_git_ms_model/")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_caption)