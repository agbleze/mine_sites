

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
    """Normalize channels to therange [0, 1] using min/max values

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



# %%
