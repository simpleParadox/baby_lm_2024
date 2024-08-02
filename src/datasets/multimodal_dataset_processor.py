import clip
import torch
import random
from torch.utils.data import DataLoader, Subset
import sys
sys.path.append('/home/rsaha/projects/babylm/src/datasets')
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

TSV_URLS = {
    'train': 'src/datasets/multimodal_train/all_multimodal.tsv',
    'val': 'src/datasets/multimodal_train/all_multimodal.tsv',
    'test': 'src/datasets/multimodal_train/all_multimodal.tsv'
    # 'val': 'src/datasets/Train-GCC-training.tsv'
    # 'train': 'https://storage.cloud.google.com/gcc-data/Train/GCC-training.tsv?_ga=2.191230122.-1896153081.1529438250'
}

FORBIDDEN_URLS = [
    'sciencephoto'
]

class MultiModalDatasetProcessor(DatasetProcessorParent):

    

    def __init__(self, device='cuda:0', batch_size=64, dataset_size=-1, n_workers=3, manual_seed=22, processor=None) -> None:

        self.train_data_pipe: IterDataPipe = None
        self.val_data_pipe: IterDataPipe = None

        self.train_dataloader = None
        self.val_dataloader = None

        self.dataset_size = dataset_size
        
        self.batch_size = batch_size
        self.val_batch_size = 8
        self.test_batch_size = 8

        self.n_workers = n_workers
        self.val_n_workers = 28
        self.manual_seed = manual_seed

        # Create ranges for train / val / test splits.
        self.full_range = np.arange(0, 2851072)  # 2851072 is the number of samples in the all_multimodal.tsv file.


        # Set the split ratios
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1

        # Calculate the split indices
        num_samples = len(self.full_range)
        train_end = int(train_ratio * num_samples)
        val_end = train_end + int(val_ratio * num_samples)

        # Shuffle the indices to ensure randomness
        np.random.seed(manual_seed)
        np.random.shuffle(self.full_range)

        # Split the indices into train, val, and test sets
        self.train_indices = self.full_range[:train_end]
        self.val_indices = self.full_range[train_end:val_end]
        self.test_indices = self.full_range[val_end:]

        # Convert the indices to lists (optional, as they are already arrays)
        self.train_indices = self.train_indices.tolist()
        self.val_indices = self.val_indices.tolist()
        self.test_indices = self.test_indices.tolist()


        # Calculate the overlap between the indices. Use sets and find set intersection. the length of the intersection should be 0.
        indices_train_set = set(self.train_indices)
        indices_val_set = set(self.val_indices)
        indices_test_set = set(self.test_indices)

        assert len(indices_train_set.intersection(indices_val_set, indices_test_set)) == 0
        print("No overlap between the indices of the train, val, and test sets.")




        



        self.device = device
        assert processor is not None
        self.image_preprocessor = processor
        # _, self.image_preprocessor = clip.load('ViT-B/32', device=self.device)


        # always need to first load train then load val dataset. Fix this confusing requirement later
        self.load_train_dataset()
        self.load_val_dataset()
        self.load_test_dataset()
        
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

        # return imgs, og_captions

        try:


            # imgs_1 = tuple(self.image_preprocessor(images=img.convert('RGBA'), return_tensors='pt') for img in imgs)
            # imgs_2 = tuple(self.image_preprocessor(images=img, return_tensors='pt').pixel_values for img in imgs)
            imgs_2 = self.image_preprocessor(images=imgs, return_tensors='pt').pixel_values
            # imgs = tuple(self.image_preprocessor(images=img, return_tensors='pt') for img in imgs)

        except Exception as e:

            print('Exception in collate_fn: ', e)

            return (None, None)

        captions = og_captions
        preprocessed_images = imgs_2 # torch.stack(imgs_2)

        outputs1 = preprocessed_images


    

        # outputs2 = [caption[1] for caption in og_captions]
        outputs2 = captions

        return (outputs1, outputs2)


    def get_num_batches_train(self) -> int:

        return len(self.train_indices) // self.batch_size  # Changed to the number of samples in the all_multimodal.tsv file.
        # return len(self.train_dataloader)
    
    def get_num_batches_test(self) -> int:
        return None  # Currently unimplemented.
    
    def get_num_batches_val(self) -> int:
        return len(self.val_indices) // self.val_batch_size
    
    @staticmethod
    def seed_dataloader_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)




    def load_train_dataset(self):

        self.train_data_pipe = multimodal_dataset_pipe(split="train", buffer_size=256, dataset_size=self.dataset_size, indices=self.train_indices)

        batch_size = self.batch_size
        self.train_dataloader = DataLoader(self.train_data_pipe, batch_size=batch_size, collate_fn=self.collate_fn, num_workers=self.n_workers, persistent_workers=True, worker_init_fn=self.seed_dataloader_worker, generator=torch.Generator().manual_seed(self.manual_seed))

    def load_val_dataset(self):

        # self.val_data_pipe = multimodal_dataset_pipe(split="val")
        # for now, same as train
        self.val_data_pipe = multimodal_dataset_pipe(split="val", indices=self.val_indices)
        batch_size = self.val_batch_size
        self.val_dataloader = DataLoader(self.val_data_pipe, batch_size=batch_size, collate_fn=self.collate_fn, num_workers=self.val_n_workers, persistent_workers=True, worker_init_fn=self.seed_dataloader_worker, generator=torch.Generator().manual_seed(self.manual_seed))
    
    def load_test_dataset(self):

        self.test_data_pipe = multimodal_dataset_pipe(split="test", indices=self.test_indices)
        batch_size = self.test_batch_size
        self.test_dataloader = DataLoader(self.test_data_pipe, batch_size=batch_size, collate_fn=self.collate_fn, num_workers=self.val_n_workers, persistent_workers=True, worker_init_fn=self.seed_dataloader_worker, generator=torch.Generator().manual_seed(self.manual_seed))



    def print_dataset_stats(self):

        print()
        print('--- TRAIN DATASET STATS ---')
        print()

        print('no of train samples: ', 3318333)

        print()
        print('--- VAL DATASET STATS ---')
        print()


        print('no of val samples: ', 15840)



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



def filter_rows(rows, indices):
    # Return the row if the index in the tsv is in the indices list. 
    # return [(row[0] != '') and (int(row[0]) in indices) for row in rows]
    return [row for row in rows if (row[0] != '' and int(row[0]) in indices)]
    # res = []
    # for row in rows:
    #     if row[0] != '' and int(row[0]) in indices:
    #         res.append(row)
    # return [element for i, element in enumerate(row) if i in indices]

def _datapipe_from_tsv_url(
    tsv_url: str, buffer_size: int = 256, dataset_size=-1, indices=None, split='train'
) -> IterDataPipe[Tuple[Image.Image, str]]:

    datapipe =  dp.iter.FileOpener([tsv_url], mode='r').readlines(return_path=False)
    
        

    if dataset_size > 0:
        print('applying header')
        datapipe = (datapipe
        .header(dataset_size)
        # NO SHUFFLING
        )

    else:
        if indices is None:
            # Return the whole datapipe.
            datapipe = (
                datapipe
                # NO HEADER
                .shuffle() # Load all the data.
                
            )
        else:
            # The indices will actually depend on the supplied ones. For each split, the indices will be different.
            # print(f"Indices for split: {split}: ", indices)
            print("Length of indices: ", len(indices))
            print("Type of indices: ", type(indices))
            # datapipe = datapipe.slice(indices).readlines()
            # from torchdata.datapipes.iter import LineReader
            # datapipe = LineReader(datapipe.shuffle())
            # print("Data pipe sliced: ", datapipe)


    datapipe: dp = datapipe.sharding_filter().map(lambda line: line.split("\t")).batch(buffer_size)


    datapipe = (datapipe.map(lambda elements: filter_rows(elements, indices)))

    



        

    return ParallelSampleLoader(datapipe)
    # return datapipe

def multimodal_dataset_pipe(
    split: str = "train", buffer_size: int = 8, dataset_size=-1, indices=None
) -> IterDataPipe[Tuple[Image.Image, str]]:
    return _datapipe_from_tsv_url(tsv_url=TSV_URLS[split], buffer_size=buffer_size, dataset_size=dataset_size, indices=indices, split=split)




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
                if image is not None:
                    yield image, caption