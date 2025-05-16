import jax
import flax
import optax
import orbax.checkpoint as ocp

print("Jax version:", jax.__version__)
print("Flax version:", flax.__version__)
print("Optax version:", optax.__version__)
print("Orbax version:", ocp.__version__)

# -------------------------------------------------------------------
from typing import Any
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class OxfordPetsDataset:
    def __init__(self, path: Path):
        assert path.exists(), path
        self.path: Path = path
        self.images = sorted((self.path / "images").glob("*.jpg"))
        self.masks = [
            self.path / "annotations" / "trimaps" / path.with_suffix(".png").name
            for path in self.images
        ]
        assert len(self.images) == len(self.masks), (len(self.images), len(self.masks))

    def __len__(self) -> int:
        return len(self.images)

    def read_image_opencv(self, path: Path):
        img = cv2.imread(str(path))
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            None

    def read_image_pillow(self, path: Path):
        img = Image.open(str(path))
        img = img.convert("RGB")
        return np.asarray(img)

    def read_mask(self, path: Path):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        # mask has values: 1, 2, 3
        # 1 - object mask
        # 2 - background
        # 3 - boundary
        # Define mask as 0-based int values
        mask = mask - 1
        return mask.astype("uint8")

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        img_path, mask_path = self.images[index], self.masks[index]
        img = self.read_image_opencv(img_path)
        if img is None:
            # Fallback to Pillow if OpenCV fails to read an image
            img = self.read_image_pillow(img_path)
        mask = self.read_mask(mask_path)
        return {
            "image": img,
            "mask": mask,
        }


class SubsetDataset:
    def __init__(self, dataset, indices: list[int]):
        # Check input indices values:
        for i in indices:
            assert 0 <= i < len(dataset)
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        i = self.indices[index]
        return self.dataset[i]


# -------------------------------------------------------------------
seed = 12
train_split = 0.7
dataset_path = Path("/data/oxford_pets")

dataset = OxfordPetsDataset(dataset_path)

rng = np.random.default_rng(seed=seed)
le = len(dataset)
data_indices = list(range(le))

# Let's remove few indices corresponding to corrupted images
# to avoid libjpeg warnings during the data loading
corrupted_data_indices = [3017, 3425]
for index in corrupted_data_indices:
    data_indices.remove(index)

random_indices = rng.permutation(data_indices)

train_val_split_index = int(train_split * le)
train_indices = random_indices[:train_val_split_index]
val_indices = random_indices[train_val_split_index:]

# Ensure there is no overlapping
assert len(set(train_indices) & set(val_indices)) == 0

train_dataset = SubsetDataset(dataset, indices=train_indices)
val_dataset = SubsetDataset(dataset, indices=val_indices)

print("Training dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))
# -------------------------------------------------------------------

import matplotlib.pyplot as plt


def display_datapoint(datapoint, label=""):
    img, mask = datapoint["image"], datapoint["mask"]
    if img.dtype in (np.float32,):
        img = ((img - img.min()) / (img.max() - img.min()) * 255.0).astype(np.uint8)
    fig, axs = plt.subplots(1, 3, figsize=(10, 10))
    axs[0].set_title(f"Image{label}")
    axs[0].imshow(img)
    axs[1].set_title(f"Mask{label}")
    axs[1].imshow(mask)
    axs[2].set_title("Image + Mask")
    axs[2].imshow(img)
    axs[2].imshow(mask, alpha=0.5)


display_datapoint(train_dataset[0], label=" (train set)")
display_datapoint(val_dataset[0], label=" (val set)")

# -------------------------------------------------------------------
import albumentations as A


img_size = 256

train_transforms = A.Compose(
    [
        A.Affine(
            rotate=(-35, 35), cval_mask=1, p=0.3
        ),  # Random rotations -35 to 35 degrees
        A.RandomResizedCrop(
            width=img_size, height=img_size, scale=(0.7, 1.0)
        ),  # Crop a random part of the input and rescale it to a specified size
        A.HorizontalFlip(p=0.5),  # Horizontal random flip
        A.RandomBrightnessContrast(
            p=0.4
        ),  # Randomly changes the brightness and contrast
        A.Normalize(),  # Normalize the image and cast to float
    ]
)


val_transforms = A.Compose(
    [
        A.Resize(width=img_size, height=img_size),
        A.Normalize(),  # Normalize the image and cast to float
    ]
)
# -------------------------------------------------------------------
output = train_transforms(**train_dataset[0])
img, mask = output["image"], output["mask"]
print("Image array info:", img.dtype, img.shape, img.min(), img.mean(), img.max())
print("Mask array info:", mask.dtype, mask.shape, mask.min(), mask.max())

# -------------------------------------------------------------------
output = val_transforms(**val_dataset[0])
img, mask = output["image"], output["mask"]
print("Image array info:", img.dtype, img.shape, img.min(), img.mean(), img.max())
print("Mask array info:", mask.dtype, mask.shape, mask.min(), mask.max())
# -------------------------------------------------------------------
from typing import Any, Callable

import grain.python as grain


class DataAugs(grain.MapTransform):
    def __init__(self, transforms: Callable):
        self.albu_transforms = transforms

    def map(self, data):
        output = self.albu_transforms(**data)
        return output
