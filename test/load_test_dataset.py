import os
from datasets import load_dataset, load_from_disk

def load_test_dataset(path: str):
    """
    If dataset exists on disk, load it from disk.
    Otherwise, download from HF and save to disk.
    """
    if os.path.exists(path):
        print("Loading dataset from disk:", path)
        ds = load_from_disk(path)
    else:
        print("Downloading dataset and saving to:", path)
        ds = load_dataset("stanfordnlp/imdb")
        ds.save_to_disk(path)
    return ds
