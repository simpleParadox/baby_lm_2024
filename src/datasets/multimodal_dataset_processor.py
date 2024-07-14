import clip
import torch
import random
from torch.utils.data import DataLoader, Subset
from datasets.dataset_processor_parent import DatasetProcessorParent
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
    # 'val': 'src/datasets/Train-GCC-training.tsv'
    # 'train': 'https://storage.cloud.google.com/gcc-data/Train/GCC-training.tsv?_ga=2.191230122.-1896153081.1529438250'
}

FORBIDDEN_URLS = [
    'sciencephoto'
]

class MultiModalDatasetProcessor(DatasetProcessorParent):

    

    def __init__(self, device='cuda:0', batch_size=64, dataset_size=-1, n_workers=3, manual_seed=22) -> None:

        self.train_data_pipe: IterDataPipe = None
        self.val_data_pipe: IterDataPipe = None

        self.train_dataloader = None
        self.val_dataloader = None

        self.dataset_size = dataset_size
        
        self.batch_size = batch_size

        self.n_workers = n_workers

        self.manual_seed = manual_seed

        



        self.device = device

      
        _, self.image_preprocessor = clip.load('ViT-B/32', device=self.device)


        # always need to first load train then load val dataset. Fix this confusing requirement later
        self.load_train_dataset()
        self.load_val_dataset()


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


            imgs = tuple(self.image_preprocessor(img.convert("RGBA")) for img in imgs)
            # imgs = tuple(self.image_preprocessor(img) for img in imgs)

        except Exception as e:

            print('Exception in collate_fn: ', e)

            return (None, None)

        captions = og_captions
        preprocessed_images = torch.stack(imgs)

        outputs1 = preprocessed_images


    

        # outputs2 = [caption[1] for caption in og_captions]
        outputs2 = captions

        return (outputs1, outputs2)


    def get_num_batches(self) -> int:

        return 3318333 // self.batch_size
        # return len(self.train_dataloader)
    
    @staticmethod
    def seed_dataloader_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)




    def load_train_dataset(self):

        self.train_data_pipe = multimodal_dataset_pipe(split="train", buffer_size=256, dataset_size=self.dataset_size)

        batch_size = self.batch_size
        self.train_dataloader = DataLoader(self.train_data_pipe, batch_size=batch_size, collate_fn=self.collate_fn, num_workers=self.n_workers, persistent_workers=True, worker_init_fn=self.seed_dataloader_worker, generator=torch.Generator().manual_seed(self.manual_seed))

    def load_val_dataset(self):

        # self.val_data_pipe = multimodal_dataset_pipe(split="val")
        # for now, same as train
        self.val_data_pipe = multimodal_dataset_pipe(split="val")


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


            
def _datapipe_from_tsv_url(
    tsv_url: str, buffer_size: int = 256, dataset_size=-1
) -> IterDataPipe[Tuple[Image.Image, str]]:

    datapipe =  (dp.iter.FileOpener([tsv_url], mode='r')
        .readlines(return_path=False)
        
    )

    if dataset_size > 0:
        print('applying header')
        datapipe = (datapipe
        .header(dataset_size)
        # NO SHUFFLING
        )

    else:
        datapipe = (
            datapipe
            # NO HEADER
            .shuffle()
            
        )



    datapipe: dp = (
        datapipe
        .sharding_filter()
        .map(lambda line: line.split("\t"))
        .batch(buffer_size)
    )

        

    return ParallelSampleLoader(datapipe)
    # return datapipe

def multimodal_dataset_pipe(
    split: str = "train", buffer_size: int = 8, dataset_size=-1
) -> IterDataPipe[Tuple[Image.Image, str]]:
    return _datapipe_from_tsv_url(tsv_url=TSV_URLS[split], buffer_size=buffer_size, dataset_size=dataset_size)




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

                print('--- FAILED --- ')
                continue
                



            # for image, caption in zip(images, captions):
            for image, caption in zip(images, captions):
                if image is not None:
                    yield image, caption