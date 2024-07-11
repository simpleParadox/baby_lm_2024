# concat urls of openimages with cc3m

# create new tsv file containing all

import pandas as pd
from tqdm import tqdm




def create_open_images_tsv():

    data: dict = {
        'image_url': [],
        'folder': [],
        'exists': [],
        'caption': [],
        'source_dataset': []
    }

    



    split = "train"

    open_images_captions_file_path = '/home/afahim2/tmp/babylm/baby_lm_2024/src/datasets/osf/multimodal_data/open_images_train_v6_captions.jsonl'


    open_images_caption_df = pd.read_json(open_images_captions_file_path, lines=True)

    print('--- UNIQUE ---')

    print(open_images_caption_df.count())

    tsv_file_path = 'src/datasets/osf/multimodal_data/open_images_train.tsv'


    # with open(tsv_file_path, 'a+') as tsv_file:

    #     tsv_file.write('image_url\tfolder\texists\tcaption\n')


    # write_batch: str = ""


    # gather open images image ids in localized narratives dataset, then store the urls and captions of those images in tsv file 

    # batch_size = 20000

    # i = 0

    # with open(tsv_file_path, 'a+') as tsv_file:
    for index, row in tqdm(open_images_caption_df.iterrows()):

        # print(f"{row['image_id']}: {row['caption']}")

        image_url = f"https://s3.amazonaws.com/open-images-dataset/{split}/{row['image_id']}.jpg"

        caption = row['caption']

        # generated_row = f"{image_url}\ttraining\t1\t{caption}\n"

        data['image_url'].append(image_url)

        data['caption'].append(caption)

        data['exists'].append(1)

        data['folder'].append('training')

        data['source_dataset'].append('open_images')

    new_df = pd.DataFrame(data=data)

    new_df.to_csv(tsv_file_path, sep='\t')




            # if i > batch_size:
            #     # write batch to file
            #     print('write batch size ', len(write_batch.split('\n')))
                
                
            #     tsv_file.write(write_batch)
            #     tsv_file.flush()

            #     write_batch = ""

            #     i = 0

            # else:
            #     write_batch += generated_row
            #     # print('write batch size ', len(write_batch.split('\n')))
            #     i += 1



def create_mscoco_tsv():

    data: dict = {
        'image_url': [],
        'folder': [],
        'exists': [],
        'caption': [],
        'source_dataset': []
    }

    split="train2014"

    mscoco_captions_file_path = '/home/afahim2/tmp/babylm/baby_lm_2024/src/datasets/osf/multimodal_data/coco_train_captions.jsonl'

    mscoco_captions_df = pd.read_json(mscoco_captions_file_path, lines=True)

    tsv_file_path = 'src/datasets/osf/multimodal_data/mscoco_train.tsv'

    write_batch:str = ""

    with open(tsv_file_path, 'a+') as tsv_file:

        tsv_file.write('image_url\tfolder\texists\tcaption\n')

        


    # gather open images image ids in localized narratives dataset, then store the urls and captions of those images in tsv file 

    batch_size = 50000

    i = 0
    for index, row in tqdm(mscoco_captions_df.iterrows()):

        # print(f"{row['image_id']}: {row['caption']}")

        image_url = f"http://images.cocodataset.org/{split}/COCO_{split}_{int(row['image_id']):012d}.jpg"

        caption = row['caption']

        data['image_url'].append(image_url)

        data['caption'].append(caption)

        data['exists'].append(1)

        data['folder'].append('training')

        data['source_dataset'].append('mscoco')

    new_df = pd.DataFrame(data=data)
    new_df.to_csv(tsv_file_path, sep='\t')



# create_open_images_tsv()
create_mscoco_tsv()
# file_path = 'src/datasets/cc_3m_training_exists_concatenated_with_captions_reduced.tsv'

# new_file = 'src/datasets/cc_3m_training_exists_concatenated_with_captions_reduced_unzip.tsv'

# df = pd.read_csv(file_path, sep='\t', compression='gzip')

# df.to_csv(new_file, sep='\t')


# open_images_tsv = 'src/datasets/osf/multimodal_data/open_images_train.tsv'

# mscoco_tsv = 'src/datasets/osf/multimodal_data/mscoco_train.tsv'

# cc_3m_tsv = 'src/datasets/osf/multimodal_data/cc_3m_training_exists_concatenated_with_captions_reduced_unzip.tsv'

# oi_df = pd.read_csv(open_images_tsv, sep='\t')

# ms_df = pd.read_csv(mscoco_tsv, sep='\t')

# cc_3m_df = pd.read_csv(cc_3m_tsv, sep='\t', index_col=0)

# print(oi_df.head())
# print(ms_df.head())
# print(cc_3m_df.head())

# oi_df.insert(4, 'source_dataset', ['open_images'] * len(oi_df), allow_duplicates=True)
# ms_df.insert(4, 'source_dataset', ['mscoco'] * len(ms_df), allow_duplicates=True)
# cc_3m_df.insert(4, 'source_dataset', ['cc_3m'] * len(cc_3m_df), allow_duplicates=True)


# oi_df[["exists"]] = oi_df[['exists']].astype(int)
# ms_df[["exists"]] = ms_df[['exists']].astype(int)





# print(oi_df.head())

# print(ms_df.head())
# print(cc_3m_df.head())

