

#%%
import os
import tempfile
from typing import Dict, Optional
import kornia.augmentation as K
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datasets import EuroSAT100
from torchgeo.transforms import AugmentationSequential, indices


# %%
class MinMaxNormalize(K.IntensityAugmentationBase2D):
    """Normalize channels to the range [0, 1] using min/max values

    Args:
        K (_type_): _description_
    """
 
    def __init__(self, mins: Tensor, maxs: Tensor) -> None:
        super().__init__(p=1)
        self.flags = {"mins": mins.view(1, -1, 1, 1), "maxs": maxs.view(1, -1, 1, 1)}
        
    def apply_transform(self, input: Tensor, params: Dict[str, int],
                        flags: Dict[str, int],
                        transform: Optional[Tensor] = None,
                        ) -> Tensor:
        return (input - flags["mins"]) / (flags["maxs"] - flags["mins"] + 1e-10)
    
#%%
mins = torch.tensor([1013.0,
                    676.0,
                    448.0,
                    247.0,
                    269.0,
                    253.0,
                    243.0,
                    189.0,
                    61.0,
                    4.0,
                    33.0,
                    11.0,
                    186.0,
                    ]
                )

maxs = torch.tensor([2309.0,
                    4543.05,
                    4720.2,
                    5293.05,
                    3902.05,
                    4473.0,
                    5447.0,
                    5948.05,
                    1829.0,
                    23.0,
                    4894.05,
                    4076.05,
                    5846.0,
                    ]
                )

bands = {"B01": "Coastal Aerosol",
    "B02": "Blue",
    "B03": "Green",
    "B04": "Red",
    "B05": "Vegetation Red Edge 1",
    "B06": "Vegetation Red Edge 2",
    "B07": "Vegetation Red Edge 3",
    "B08": "NIR 1",
    "B8A": "NIR 2",
    "B09": "Water Vapour",
    "B10": "SWIR 1",
    "B11": "SWIR 2",
    "B12": "SWIR 3",
}

#%%
batch_size = 4
num_workers = 2

#%% load EuroSAT MS dataset and dataloader
root = os.path.join(tempfile.gettempdir(), "eurosat100")
dataset = EuroSAT100(root, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers
                        )

dataloader = iter(dataloader)
print(f"Number of images in dataset: {len(dataset)}")
print(f"Dataset Classes: {dataset.classes}")

#%%  ####   load a sample and batch of images and labels
sample = dataset[0]
x, y = sample["image"], sample["label"]
print(x.shape, x.dtype, x.min(), x.max())
print(y, dataset.classes[y])

# %%  test dataloader for loading batches of images
# image is o shape (4, 13, 64, 64) -> (batch_num, channels(bands), height, width)
batch = next(dataloader)
x, y = batch["image"], batch["label"]
print(x.shape, x.dtype, x.min(), x.max())
print(y, [dataset.classes[i] for i in y])
# %% transforms usage
transform = MinMaxNormalize(mins, maxs)
x = transform(x)
print(x.shape)
print(x.dtype, x.min(), x.max())


# %% compute indices and append as additional channel
transform = indices.AppendNDVI(index_nir=7, index_red=3)
batch = next(dataloader)
x = batch["image"]
print(x.shape)
x = transform(x)
print(x.shape)

# %% adding several indices to the channels
transforms = nn.Sequential(MinMaxNormalize(mins, maxs),
                            indices.AppendNDBI(index_swir=11, index_nir=7),
                            indices.AppendNDSI(index_green=3, index_swir=11),
                            indices.AppendNDVI(index_nir=7, index_red=3),
                            indices.AppendNDWI(index_green=2, index_nir=7)
                        )


batch = next(dataloader)
x = batch["image"]
print(x.shape)
x = transforms(x)
print(x.shape)

#%% chain indicies with augmentations form Kornia
transforms = AugmentationSequential(
    MinMaxNormalize(mins, maxs),
    indices.AppendNDBI(index_swir=11, index_nir=7),
    indices.AppendNDSI(index_green=3, index_swir=11),
    indices.AppendNDVI(index_nir=7, index_red=3),
    indices.AppendNDWI(index_green=2, index_nir=7),
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    data_keys=["image"],
)

batch = next(dataloader)

print(batch["image"].shape)
batch = transforms(batch)
print(batch["image"].shape)



#%% transformations are nn.Module hence can be sent to GPU for faster speed on large data

device = "cuda" if torch.cuda.is_available() else "cpu"

transforms = AugmentationSequential(
    MinMaxNormalize(mins, maxs),
    indices.AppendNDBI(index_swir=11, index_nir=7),
    indices.AppendNDSI(index_green=3, index_swir=11),
    indices.AppendNDVI(index_nir=7, index_red=3),
    indices.AppendNDWI(index_green=2, index_nir=7),
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomAffine(degrees=(0.90), p=0.25),
    K.RandomGaussianBlur(kernel_size=(3,3), sigma=(0.1, 2.0), p=0.25),
    K.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=0.25),
    data_keys=["image"],
)

transforms_gpu = AugmentationSequential(
                    MinMaxNormalize(mins.to(device), maxs.to(device)),
                    indices.AppendNDBI(index_swir=11, index_nir=7),
                    indices.AppendNDSI(index_green=3, index_swir=11),
                    indices.AppendNDVI(index_nir=7, index_red=3),
                    indices.AppendNDWI(index_green=2, index_nir=7),
                    K.RandomHorizontalFlip(p=0.5),
                    K.RandomVerticalFlip(p=0.5),
                    K.RandomAffine(degrees=(0.90), p=0.25),
                    K.RandomGaussianBlur(kernel_size=(3,3), sigma=(0.1, 2.0), p=0.25),
                    K.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=0.25),
                    data_keys=["image"],
                ).to(device)

def get_batch_cpu():
    return dict(image=torch.randn(64, 13, 512, 512).to("cpu"))

def get_batch_gpu():
    return dict(image=torch.randn(64, 13, 512, 512).to(device))

#%%timeit -n 1 -r 5
_ = transforms(get_batch_cpu())


#%%timeit -n 1 -r 5
_ = transforms_gpu(get_batch_gpu())


#%% visualize images and labels
transforms = AugmentationSequential(MinMaxNormalize(mins, maxs), data_keys=["image"])
dataset = EuroSAT100(root, transforms=transforms)

#%%
idx = 21
sample = dataset[idx]
rgb = sample["image"][0, 1:4]
image = T.ToPILImage()(rgb)
print(f"Class label: {dataset.classes[sample['label']]}")
image.resize((256, 256), resample=Image.BILINEAR).show()

### index channels apex accuracy (ICAC)
# %%
