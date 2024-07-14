from datasets import load_dataset_builder

ds_builder = load_dataset_builder("rotten_tomatoes")


print(ds_builder.info.description)