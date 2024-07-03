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
    'train': 'src/datasets/Train-GCC-training.tsv',
    'val': 'src/datasets/Train-GCC-training.tsv'
    # 'train': 'https://storage.cloud.google.com/gcc-data/Train/GCC-training.tsv?_ga=2.191230122.-1896153081.1529438250'
}

FORBIDDEN_URLS = [
    'sciencephoto'
]

class ConceptualCaptionsProcessor(DatasetProcessorParent):

    

    def __init__(self, return_org_imgs_collate_fn=False, return_only_captions=False, cuda_device='cuda:0', batch_size=64) -> None:

        self.train_data_pipe: IterDataPipe = None
        self.val_data_pipe: IterDataPipe = None

        self.train_dataset = None
        self.train_dataset = None
        self.train_dataloader = None
        self.val_dataset = None
        self.val_dataloader = None
        
        self.batch_size = batch_size



        self.device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")

      
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
        return len(self.train_dataloader)
    
    @staticmethod
    def seed_dataloader_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)




    def load_train_dataset(self):

        self.train_data_pipe = conceptual_captions_3m(split="train", buffer_size=256)

        batch_size = self.batch_size


        self.train_dataloader = DataLoader(self.train_data_pipe, batch_size=batch_size, collate_fn=self.collate_fn, num_workers=0, worker_init_fn=self.seed_dataloader_worker, generator=torch.Generator().manual_seed(22))

    def load_val_dataset(self):

        self.val_data_pipe = conceptual_captions_3m(split="val")

        # val_indices = torch.randint(0, 15840 , (wandb.config['validation_dataset_size'],))


        # subsetting probably doesnt work with data pipes, CHECK LATER IF NEEDED
        val_indices = torch.arange(0, 1024)
        val_data_subset = Subset(self.val_data_pipe, val_indices)
        



        # no need val dataloader as I'm creating it in do_validation in utils

        # val_dataloader = DataLoader(val_data_subset, batch_size=wandb.config['validation_batch_size'], shuffle=True, collate_fn=self.collate_fn, num_workers=wandb.config['num_workers'], worker_init_fn=self.seed_dataloader_worker)


        # set class variables
        self.val_dataset = val_data_subset
        # self.val_dataloader = val_dataloader

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
    tsv_url: str, buffer_size: int = 256
) -> IterDataPipe[Tuple[Image.Image, str]]:
    # pipe = HttpReader([tsv_url])
    # pipe = LineReader(pipe, return_path=False)

    # source_dp = IterableWrapper([(tsv_url, io.StringIO(text1)), ("file2", io.StringIO(text2))])

    datapipe = (
        dp.iter.FileOpener([tsv_url], mode='r')
        .readlines(return_path=False)
        .shuffle()
        .sharding_filter()
        .map(lambda line: line.split("\t"))
        .batch(buffer_size)
        
        
        # .map(lambda x: package_images_captions(x))
    )

    # pipe = pipe.sharding_filter()

    # pipe = LineReader(pipe, return_path=False)
    # # # use pipe to read from local file
    # # pipe = LineReader(pipe, return_path=True)
    # # LineReader downloads raw bytes.  Decode them to strings, then split.

    
    # pipe = pipe.map(lambda line: line.split("\t"))

    return ParallelSampleLoader(datapipe)
    # return datapipe

def conceptual_captions_3m(
    split: str = "train", buffer_size: int = 256
) -> IterDataPipe[Tuple[Image.Image, str]]:
    return _datapipe_from_tsv_url(tsv_url=TSV_URLS[split], buffer_size=buffer_size)




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
            captions = [x[0] for x in batch]
            image_urls = [x[1] for x in batch]
            images = asyncio.run(async_batch_get_images(image_urls))

            for image, caption in zip(images, captions):
                if image is not None:
                    yield image, caption