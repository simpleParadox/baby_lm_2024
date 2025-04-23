import clip
import torch
import random
from torch.utils.data import DataLoader, Subset
import sys
sys.path.append('/home/rsaha/projects/baby_lm_2024/src/datasets')
sys.path.append('/home/rsaha/scratch/baby_lm_2024/src/datasets')
sys.path.append('/home/rsaha/projects/baby_lm_2024')
sys.path.append('/home/rsaha/scratch/baby_lm_2024')
from dataset_processor_parent import DatasetProcessorParent
import numpy as np

import torchdata.datapipes as dp

import aiohttp
from PIL import Image
import io
from typing import Optional
from torchdata.datapipes.iter import IterDataPipe
# import Generator
from typing import Generator, List, Tuple, Sequence
import asyncio

from torch import Tensor


import os

import json
profile = json.load(open('profile.json'))
print("Profile: ", profile)
if profile['cluster'] == 'cirrus':
    train_tsv_url = '/home/rsaha/projects/baby_lm_2024/src/datasets/multimodal_train/all_multimodal_all_concaps_uncompressed_dropped_first_col.tsv'
elif profile['cluster'] == 'vulcan':
    train_tsv_url = '/home/rsaha/scratch/baby_lm_2024/src/datasets/multimodal_train/all_multimodal_all_concaps_uncompressed_dropped_first_col.tsv'

TSV_URLS = {
    'train': 'src/datasets/multimodal_train/',
    'val': 'src/datasets/multimodal_train/',
    # 'val': 'src/datasets/Train-GCC-training.tsv'
    # 'train': 'https://storage.cloud.google.com/gcc-data/Train/GCC-training.tsv?_ga=2.191230122.-1896153081.1529438250'
}

CURRICULUM_TSV_URLS = {
    'train': 
        'src/datasets/multimodal_train/curriculum_tsvs_95_train_replaced/'
}

FORBIDDEN_URLS = [
    'sciencephoto'
]


"""
Create a text only dataloader. This part isn't a pipe because the data
is available on disk.
"""

# class GPT2Dataset(Dataset):

#   def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):

#     self.tokenizer = tokenizer
#     self.input_ids = []
#     self.attn_masks = []

#     for txt in txt_list:

#       encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")

#       self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
#       self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
#   def __len__(self):
#     return len(self.input_ids)

#   def __getitem__(self, idx):
#     return self.input_ids[idx], self.attn_masks[idx] 

class MultiModalDatasetProcessor(DatasetProcessorParent):

    

    def __init__(self, device='cuda:0', batch_size=64, dataset_size=-1, n_workers=3, manual_seed=42, processor=None, do_val=True,
                 do_curriculum=False) -> None:

        self.train_data_pipe: IterDataPipe = None
        self.val_data_pipe: IterDataPipe = None

        self.train_dataloader = None
        self.val_dataloader = None

        self.dataset_size = dataset_size
        
        self.batch_size = batch_size
        self.val_batch_size = 8

        self.n_workers = n_workers
        self.val_n_workers = 5
        self.manual_seed = manual_seed
        self.do_curriculum = do_curriculum
        self.train_indices = None
        self.val_indices = None
        # Create ranges for train / val / test splits.
        self.full_range = np.arange(0, 2891030)  # 2851072 is the number of samples in the all_multimodal.tsv file.
        # if dataset_size > 0:
        #     self.full_range = np.arange(0, dataset_size)

        if do_val:
            # Set the split ratios
            train_ratio = 0.9

            # Calculate the split indices
            num_samples = len(self.full_range)
            train_end = int(train_ratio * num_samples)

            # Shuffle the indices to ensure randomness
            np.random.seed(manual_seed)
            np.random.shuffle(self.full_range)

            # Split the indices into train, val, and test sets
            self.train_indices = self.full_range[:train_end]
            self.val_indices = self.full_range[train_end:]

            # Convert the indices to lists (optional, as they are already arrays)
            self.train_indices = self.train_indices.tolist()
            self.val_indices = self.val_indices.tolist()

            # Calculate the overlap between the indices. Use sets and find set intersection. the length of the intersection should be 0.
            indices_train_set = set(self.train_indices)
            indices_val_set = set(self.val_indices)

            assert len(indices_train_set.intersection(indices_val_set)) == 0
            print("No overlap between the indices of the train and val sets.")
        else:
            self.train_indices = self.full_range[:]
            self.val_indices = []
            self.val_dataset = None


        self.device = device
        assert processor is not None
        self.image_preprocessor = processor  # I replaced the line below with this line.
        # _, self.image_preprocessor = clip.load('ViT-B/32', device=self.device)


        # always need to first load train then load val dataset. Fix this confusing requirement later
        if not do_curriculum:
            self.train_standard_n_samples = np.arange(0, 2891030)
            self.load_train_dataset()
        else:
            # These are hardcoded so have to be changed if the dataset changes.
            # These values are updated for the train split.
            self.train_q1_n_samples = np.arange(0, 841907)
            self.train_q2_n_samples = np.arange(0, 1510443)
            self.train_q3_n_samples = np.arange(0, 2283866)
            self.train_q4_n_samples = np.arange(0, 2891030)

            self.load_curriculum_datasets()  # This will load multiple dataloaders based on the predefined split.
        if do_val:
            print("Doing validation")
            self.load_val_dataset()
            self.val_standard_n_samples = np.arange(0, 152160)
        
    def collate_fn(self, batch) -> tuple[Tensor, list[str]]:
        '''
        batch is a list of tuples?
        each tuple is of the form (image, caption)
        image is a jpeg image
        caption is a tuple of strings
        '''


        imgs: list
        og_captions: list

        imgs, og_captions = zip(*batch)
        
        
        """
        The following block was the one before I made all the changes. This is a slight modification of the list comprehension.
        """
        
        # imgs_2 = tuple(self.image_preprocessor(img.convert("RGBA"), return_tensors='pt').pixel_values for img in imgs)
        
        captions = []
        imgs_converted = []
        
        for img, cap in zip(imgs, og_captions):
            try:
                img = img.convert('RGBA')
                # The following lines execute only if the above line doesn't throw an exception.
                imgs_converted.append(img)
                captions.append(cap)
            except Exception as e:
                print("Cannot convert image to RGBA. Trying conversion to RGB.", e)
                try:
                    img = img.convert('RGB')
                    imgs_converted.append(img)
                    captions.append(cap)
                except Exception as e:
                    print("Cannot convert image to RGBA or  RGB. Skipping this image.", e)
        
        # Now convert the images to tensors.                    
        try:
            imgs_2 = self.image_preprocessor(images=imgs_converted, return_tensors='pt').pixel_values
        except Exception as e:
            print("Cannot process one or more images.")
            print("Figuring out which of the images is causing the problem.")
            good_images = []
            good_captions = []
            for img_index, (img, caption) in enumerate(zip(imgs_converted, captions)):
                try:
                    temp = self.image_preprocessor(images=img, return_tensors='pt').pixel_values
                    good_images.append(img)
                    good_captions.append(caption)
                except Exception as e:
                    print("Cannot process image at index: ", img_index)
            
            imgs_2 = self.image_preprocessor(images=good_images, return_tensors='pt').pixel_values
            captions = good_captions  # Newly assign captions.
            # Now remove the faulty images from the list and the captions

        
        # NOTE: The following is the old code.
        # try:


        #     # imgs_1 = tuple(self.image_preprocessor(images=img.convert('RGBA'), return_tensors='pt') for img in imgs)
        #     # imgs_2 = tuple(self.image_preprocessor(images=img, return_tensors='pt').pixel_values for img in imgs)
        #     # Convert to 'RGBA' mode before passing to the image preprocessor.
        #     imgs = [img.convert('RGBA') for img in imgs]
        #     imgs_2 = self.image_preprocessor(images=imgs, return_tensors='pt').pixel_values
        #     # imgs = tuple(self.image_preprocessor(images=img, return_tensors='pt') for img in imgs)

        # except Exception as e:

        #     print('Exception in collate_fn: ', e)

        #     return (None, None)

        # captions = og_captions
        # preprocessed_images = imgs_2 # torch.stack(imgs_2)

        
        
        # flag = 0
        # imgs_results = []
        # for img in imgs:
        #     try:
        #         img = img.convert('RGBA')
        #     except Exception as e:
        #         print("Exception in image conversion. Cannot convert to RGBA.", e)
        #         print("Trying conversion to RGB.")
        #         try:
        #             img = img.convert('RGB')
        #             print("Conversion to RGB successful")
        #         except Exception as e:
        #             print("Exception in image conversion. Cannot convert to RGB.", e)
        #             import pdb
        #             pdb.set_trace()
        #             print("Cannot convert image to RGB. Skipping this image.")
        #             flag = 1
        #             continue
        #     imgs_results.append(img)
        # flag = 0
        # try:

        #     imgs_2 = self.image_preprocessor(images=imgs_results, return_tensors='pt').pixel_values
        # except Exception as ex:
        #     # print('Exception in image_processing: ', ex)
        #     # Print the images, and also the shape of each image in the batch.
        #     print("Batch size: ", len(imgs_results))
        #     print("Imgs results: ", imgs_results)
            
        #     for img in imgs_results:
        #         print("Image shape: ", img.size)

        #     import pdb
        #     pdb.set_trace()
        #     print('Exception in image_processing: ', ex)
        #     flag = 1
        
        # if flag == 1:    
        #     return (None, None)
    
        preprocessed_images = imgs_2 # torch.stack(imgs_2)

        outputs1 = preprocessed_images


    

        # outputs2 = [caption[1] for caption in og_captions]
        outputs2 = captions

        return (outputs1, outputs2)


    def get_num_batches_train(self) -> int:
        # NOTE: It must be noted tha even if this function returns the total number of possible batches,
        # there are cases, where some samples are just failed to be processed. In that case, the number of samples
        # in a batch will be less than the declared batch size.
        # Regardless, in the train_git_base_multimodal.py file, this is accounted for, and I actually count the
        # number of batches, so that the correct value is used to calculate the epoch loss.
        if not self.do_curriculum:
            return len(self.train_standard_n_samples) // self.batch_size  # Changed to the number of samples in the all_multimodal.tsv file.
        else:
            quartile_1_batches = len(self.train_q1_n_samples) // self.batch_size
            quartile_2_batches = len(self.train_q2_n_samples) // self.batch_size
            quartile_3_batches = len(self.train_q3_n_samples) // self.batch_size
            quartile_4_batches = len(self.train_q4_n_samples) // self.batch_size

            return [quartile_1_batches, quartile_2_batches, quartile_3_batches, quartile_4_batches]

    def get_num_batches_val(self) -> int:
        return len(self.val_standard_n_samples) // self.val_batch_size
    
    def get_num_batches_test(self) -> int:
        raise NotImplementedError  # Currently unimplemented.
    
    
    @staticmethod
    def seed_dataloader_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    def load_curriculum_datasets(self):
        """
        Load multiple datasets for each epoch.
        The first dataset will contain samples of difficulty level in the first quartile.
        the second dataset will contain samples of difficulty level second quartile,
        the third dataset will contain samples of difficulty level third quartile,
        the fourth dataset will contain all the samples.
        """

        self.curriculum_dataloaders = []
        seed = self.manual_seed
        # Load the datasets for that seed.
        train_datapipe_quartile_1 = multimodal_dataset_pipe_curriculum(split="train", buffer_size=256, 
                                                                        dataset_size=self.dataset_size,
                                                                        tsv_url=f"{CURRICULUM_TSV_URLS['train']}quartile_1_seed_0_assigned_max_replaced_train.tsv")
        train_datapipe_quartile_2 = multimodal_dataset_pipe_curriculum(split="train", buffer_size=256,
                                                                        dataset_size=self.dataset_size,
                                                                        tsv_url=f"{CURRICULUM_TSV_URLS['train']}quartile_2_seed_0_assigned_max_replaced_train.tsv")
        train_datapipe_quartile_3 = multimodal_dataset_pipe_curriculum(split="train", buffer_size=256,
                                                                        dataset_size=self.dataset_size,
                                                                        tsv_url=f"{CURRICULUM_TSV_URLS['train']}quartile_3_seed_0_assigned_max_replaced_train.tsv")
        
        # The fourth quartile dataset contains all the rows.
        train_datapipe_quartile_4 = multimodal_dataset_pipe_curriculum(split="train", buffer_size=256,
                                                                            dataset_size=self.dataset_size,
                                                                            tsv_url=f"{TSV_URLS['train']}all_multimodal_all_concaps_with_pos_tags_with_noun_counts_assigned_max_replaced_train_seed_0.tsv")
        batch_size = self.batch_size

        # Now for each datapipe, create a dataloader.
        self.curriculum_dataloaders.append(DataLoader(train_datapipe_quartile_1, batch_size=batch_size, collate_fn=self.collate_fn, num_workers=self.n_workers, persistent_workers=True, worker_init_fn=self.seed_dataloader_worker, generator=torch.Generator().manual_seed(self.manual_seed), pin_memory=True))
        self.curriculum_dataloaders.append(DataLoader(train_datapipe_quartile_2, batch_size=batch_size, collate_fn=self.collate_fn, num_workers=self.n_workers, persistent_workers=True, worker_init_fn=self.seed_dataloader_worker, generator=torch.Generator().manual_seed(self.manual_seed), pin_memory=True))
        self.curriculum_dataloaders.append(DataLoader(train_datapipe_quartile_3, batch_size=batch_size, collate_fn=self.collate_fn, num_workers=self.n_workers, persistent_workers=True, worker_init_fn=self.seed_dataloader_worker, generator=torch.Generator().manual_seed(self.manual_seed), pin_memory=True))
        self.curriculum_dataloaders.append(DataLoader(train_datapipe_quartile_4, batch_size=batch_size, collate_fn=self.collate_fn, num_workers=self.n_workers, persistent_workers=True, worker_init_fn=self.seed_dataloader_worker, generator=torch.Generator().manual_seed(self.manual_seed), pin_memory=True))

        
    def load_train_dataset(self):
        seed = self.manual_seed
        # self.train_data_pipe = multimodal_dataset_pipe(split="train", buffer_size=256, dataset_size=self.dataset_size, indices=self.train_indices, tsv_url=f"{TSV_URLS['train']}all_multimodal_all_concaps_with_pos_tags_with_noun_counts_assigned_max_replaced_train_seed_{seed}.tsv")
        
        self.train_data_pipe = multimodal_dataset_pipe(split="train", buffer_size=256, dataset_size=self.dataset_size, indices=self.train_indices, tsv_url=train_tsv_url)
        batch_size = self.batch_size
        print("Using random number generator.")
        self.train_dataloader = DataLoader(self.train_data_pipe, batch_size=batch_size, collate_fn=self.collate_fn, num_workers=self.n_workers, persistent_workers=True, worker_init_fn=self.seed_dataloader_worker, generator=torch.Generator().manual_seed(self.manual_seed), pin_memory=True)


    def load_val_dataset(self):
        seed = self.manual_seed
        self.val_data_pipe = multimodal_dataset_pipe(split="val", buffer_size=256, dataset_size=self.dataset_size, indices=self.val_indices, tsv_url=f"{TSV_URLS['val']}all_multimodal_all_concaps_with_pos_tags_with_noun_counts_assigned_max_replaced_val_seed_0.tsv")  # This is for both curriculum and non-curriculum dataset (because only the training set changes).
        batch_size = self.val_batch_size
        self.val_dataloader = DataLoader(self.val_data_pipe, batch_size=batch_size, collate_fn=self.collate_fn, num_workers=self.val_n_workers, persistent_workers=True, worker_init_fn=self.seed_dataloader_worker, generator=torch.Generator().manual_seed(self.manual_seed))
    
    # def get_dataset_length(self, split='train', curriculum=False, quartile=0) -> int:
    #     if not curriculum:
    #         if split == 'train':
    #             return len(self.train_indices)
    #         elif split == 'val':
    #             return len(self.val_indices)
    #     else:
    #         # Curriculum
    #         assert quartile in [1, 2, 3, 4]
    #         if quartile == 1:
    #             return len(self.train_q1_n_samples)
            
            

    def print_dataset_stats(self):

        print()
        print('--- TRAIN DATASET STATS ---')
        print()

        print('no of train samples: ', self.train_standard_n_samples)

        print()
        print('--- VAL DATASET STATS ---')
        print()


        print('no of val samples: ', len(self.val_standard_n_samples))



async def async_get_image(
    session: aiohttp.ClientSession, url: str
) -> Optional[Image.Image]:
    try:

        for forbidden_url in FORBIDDEN_URLS:
            if forbidden_url in url:
                print(f'FORBIDDEN URL FOUND: {url}')
                print(' -- SKIPPING THIS -- ')
                return None
            
        
        resp = await session.get(url)
        image_bytes = await resp.read()
        return Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        # print(f"Error downloading {url}: {e}")
        # If an exception occurs, such as a timeout, invalid URL, etc, just
        # return None, and the caller can handle skipping this
        return None
    
async def async_batch_get_images(
    urls: Sequence[str], 
    timeout: float = 1.0
) -> List[Optional[Image.Image]]:
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        return await asyncio.gather(*[async_get_image(session, url) for url in urls])

def package_images_captions(batch):
    # The batch is a list of tuples, where the first element is the
    # caption, and the second element is the URL of the image.

    # print('batch: ', batch)
    captions = [x[0] for x in batch]
    image_urls = [x[1] for x in batch]


    images = asyncio.run(async_batch_get_images(image_urls))

    for image, caption in zip(images, captions):
        if image is not None:
            yield image, caption

def filter_rows_curriculum(rows):
    return [row for row in rows if (row[0] != '' and row[0] != 'Unnamed: 0')]


def filter_rows(rows, indices=None):
    # Return the row if the index in the tsv is in the indices list. 
    # return [(row[0] != '') and (int(row[0]) in indices) for row in rows]
    # return [row for row in rows if (row[0] != '' and int(row[0]) in indices)]
    return [row for row in rows if (row[0] != '' and row[0] != 'Unnamed: 0')]
    # res = []
    # for row in rows:
    #     if row[0] != '' and int(row[0]) in indices:
    #         res.append(row)
    # return [element for i, element in enumerate(row) if i in indices]

def _datapipe_from_tsv_url(
    tsv_url: str, buffer_size: int = 256, dataset_size=-1, indices=None, split='train', do_curriculum=False
) -> IterDataPipe[Tuple[Image.Image, str]]:

    datapipe =  dp.iter.FileOpener([tsv_url], mode='r').readlines(return_path=False)
    
        

    if dataset_size > 0:
        print('applying header')
        datapipe = (datapipe
        .header(dataset_size)
        # NO SHUFFLING
        )

    # else:
    #     if indices is None:
    #         # Return the whole datapipe.
    #         datapipe = (
    #             datapipe
    #             # NO HEADER
    #             .shuffle() # Load all the data.
                
    #         )
    #     else:
    #         # The indices will actually depend on the supplied ones. For each split, the indices will be different.
    #         # print(f"Indices for split: {split}: ", indices)
    #         print("Length of indices: ", len(indices))
    #         print("Type of indices: ", type(indices))
    #         # datapipe = datapipe.slice(indices).readlines()
    #         # from torchdata.datapipes.iter import LineReader
    #         # datapipe = LineReader(datapipe.shuffle())
    #         # print("Data pipe sliced: ", datapipe)


    datapipe: dp = datapipe.sharding_filter().map(lambda line: line.split("\t")).batch(buffer_size)

    if not do_curriculum:
        # datapipe = (datapipe.map(lambda elements: filter_rows(elements, indices)))
        datapipe = (datapipe.map(lambda elements: filter_rows(elements, None)))
    else:
        datapipe = datapipe.map(lambda elements: filter_rows_curriculum(elements))

    



        

    return ParallelSampleLoader(datapipe)
    # return datapipe


def multimodal_dataset_pipe_curriculum(
    split: str = "train", buffer_size: int = 8, dataset_size=-1, indices=None, tsv_url=None
) -> IterDataPipe[Tuple[Image.Image, str]]:
    return _datapipe_from_tsv_url(tsv_url=tsv_url, buffer_size=buffer_size, dataset_size=dataset_size, indices=indices, split=split, do_curriculum=True)

def multimodal_dataset_pipe(
    split: str = "train", buffer_size: int = 8, dataset_size=-1, indices=None, tsv_url=None
) -> IterDataPipe[Tuple[Image.Image, str]]:
    return _datapipe_from_tsv_url(tsv_url=tsv_url, buffer_size=buffer_size, dataset_size=dataset_size, indices=indices, split=split)




class ParallelSampleLoader(IterDataPipe):
    def __init__(
        self, dp: IterDataPipe[Tuple[str, str]]
    ) -> None:
        super().__init__()
        self.dp = dp

    def __iter__(self) -> Generator[Tuple[Image.Image, str], None, None]:
        # pipe: IterDataPipe[List[Tuple[str, str]]] = self.dp.batch(self.buffer_size)
        pipe = self.dp
        for batch in pipe:
            # The batch is a list of tuples, where the first element is the
            # caption, and the second element is the URL of the image.

            # print('batch: ', batch)

            try:
                captions = [x[4] for x in batch]
                image_urls = [x[1] for x in batch]
                images = asyncio.run(async_batch_get_images(image_urls))

            except:
                print("Batch: ", batch)
                print('--- FAILED --- ')
                continue
                



            # for image, caption in zip(images, captions):
            for image, caption in zip(images, captions):
                if image is not None and caption is not None:
                    yield image, caption