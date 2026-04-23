import os
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler

class config:
    CSV_PATH = "/kaggle/input/datasets/awsaf49/brats2020-training-data/BraTS20 Training Metadata.csv"
    TRAIN_DIR = "/kaggle/input/datasets/awsaf49/brats2020-training-data/BraTS2020_training_data"
    TUMOR_IDX_PATH = "/kaggle/input/datasets/karthikeyaa6274/indices/indices/tumor_indices.pkl"
    EMPTY_IDX_PATH = "/kaggle/input/datasets/karthikeyaa6274/indices/indices/empty_indices.pkl"

def create_df(config=config):
    return pd.read_csv(config.CSV_PATH)

def load_h5(df, idx, config=config):
    row = df.iloc[idx]
    rel_path = row["slice_path"]
    rel_path = rel_path.replace("../input/brats2020-training-data/BraTS2020_training_data/", "")
    full_path = os.path.join(config.TRAIN_DIR, rel_path)
    with h5py.File(full_path, "r") as f:
        #print(list(f.keys())) 
        image = f["image"][:]
        mask  = f["mask"][:]
    return image, mask
    
def normalize_image(img):
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img
    
def plot_sample(image, mask):
    import numpy as np
    import matplotlib.pyplot as plt

    if image.ndim == 3:
        image = image[..., 0]
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=-1)

    image = normalize_image(image)

    fig, ax = plt.subplots(1,3, figsize=(15,5))

    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("MRI")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Mask")
    ax[1].axis("off")

    ax[2].imshow(image, cmap="gray")
    ax[2].imshow(mask, cmap="jet", alpha=0.4)
    ax[2].set_title("Overlay")
    ax[2].axis("off")

    plt.show()

class BraTSDataset(Dataset):
    def __init__(self, df, config, transform=None, filter_empty=False):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.transform = transform
        self.filter_empty = filter_empty
        if self.filter_empty:
            self.valid_indices = self._filter_non_empty()
        else:
            self.valid_indices = list(range(len(self.df)))

    def _resolve_path(self, rel_path):
        rel_path = rel_path.replace("../input/brats2020-training-data/", "")
        return os.path.join(self.config.TRAIN_DIR, rel_path)

    def _filter_non_empty(self):
        valid = []
        for i in range(len(self.df)):
            path = self._resolve_path(self.df.iloc[i]["slice_path"])

            try:
                with h5py.File(path, "r") as f:
                    mask = f["mask"][:]
                    if mask.ndim == 3:
                        mask = np.argmax(mask, axis=-1)

                    if mask.sum() > 0:
                        valid.append(i)
            except:
                continue
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]

        path = self._resolve_path(row["slice_path"])

        with h5py.File(path, "r") as f:
            image = f["image"][:]
            mask = f["mask"][:]

        # -------------------------
        # IMAGE processing
        # -------------------------
        if image.ndim == 3:
            image = image[..., 0]  # take first modality

        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # -------------------------
        # MASK processing
        # -------------------------
        if mask.ndim == 3:
            mask = np.argmax(mask, axis=-1)

        mask = mask.astype(np.int64)

        # -------------------------
        # TO TENSOR
        # -------------------------
        image = torch.tensor(image).unsqueeze(0)  # (1, H, W)
        mask = torch.tensor(mask)                 # (H, W)

        # optional transforms
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

class BalancedBatchSampler(Sampler):
    def __init__(self, tumor_indices, empty_indices, batch_size):
        self.tumor = np.array(
            pickle.load(open(config.TUMOR_IDX_PATH, "rb"))
        )
        self.empty = np.array(
            pickle.load(open(config.EMPTY_IDX_PATH, "rb"))
        )
        self.batch_size = batch_size

        assert batch_size % 2 == 0, "Batch size must be even for 50-50 split"

    def __iter__(self):
        tumor_half = self.batch_size // 2
        empty_half = self.batch_size // 2

        tumor_shuffled = np.random.permutation(self.tumor)
        empty_shuffled = np.random.permutation(self.empty)

        # repeat empty if needed
        if len(empty_shuffled) < len(tumor_shuffled):
            empty_shuffled = np.random.choice(
                self.empty,
                size=len(tumor_shuffled),
                replace=True
            )

        batches = []

        for i in range(0, len(tumor_shuffled), tumor_half):
            tumor_batch = tumor_shuffled[i:i + tumor_half]

            empty_batch = np.random.choice(
                self.empty,
                size=len(tumor_batch),
                replace=False
            )

            batch = np.concatenate([tumor_batch, empty_batch])
            np.random.shuffle(batch)

            batches.append(batch)

        return iter(batches)

    def __len__(self):
        return len(self.tumor) // (self.batch_size // 2)
