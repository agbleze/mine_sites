

#%%
import os
import tempfile
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import planetary_computer
import pystac 
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import RandomGeoSampler
import rasterio
from rasterio import plot


#%%
root = os.path.join(tempfile.gettempdir(), "sentinel")

#%%
item_urls = [
    "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2B_MSIL2A_20220902T090559_R050_T40XDH_20220902T181115",
    "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/S2B_MSIL2A_20220718T084609_R107_T40XEJ_20220718T175008",
]

for item_url in item_urls:
    item = pystac.Item.from_file(item_url)
    signed_item = planetary_computer.sign(item)
    for band in ["B02", "B03", "B04", "B08"]:
        asset_href = signed_item.assets[band].href
        filename = urlparse(asset_href).path.split("/")[-1]
        download_url(asset_href, root, filename)

#%%
sorted(os.listdir(root))


#%%

class Sentinel2(RasterDataset):
    filename_glob = "T*_B02_10m.tif"
    filename_regex = r"^.{6}_(?P<date>\d{8}T\d{6})_(?P<band>B0[\d])"
    date_format = "%Y%m%dT%H%M%S"
    is_image = True
    separate_files = True
    all_bands = ["B02", "B03", "B04", "B08"]
    rgb_bands = ["B04", "B03", "B02"]
    
    def plot(self, sample):
        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.all_bands.index(band))
            
        # reoder and rescale image
        image = sample["image"][rgb_indices].permute(1,2,0)
        image = torch.clamp(image / 10000, min=0, max=1).numpy()
        
        # plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)
        
        return fig
    

#%%
dataset = Sentinel2(root)
print(dataset)

#%%
torch.manual_seed(1)

dataset = Sentinel2(root)
sampler = RandomGeoSampler(dataset, size=4096, length=3)
dataloader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)

for batch in dataloader:
    sample = unbind_samples(batch)[0]
    dataset.plot(sample)
    plt.axis("off")
    plt.show()

# %%
train_0_path = "/home/lin/codebase/mine_sites/solafune_find_mining_sites/train/train/train_0.tif"
img_1 = rasterio.open(train_0_path)

#%%


#%%
plot.show(img_1)

# %%

for i in next(img_1):
    print(i)

# %%  ## create a dataset object that loads images and labels
#  Apat to the EuroSAT format nad cater for our folder structure
#%%

from torchvision.datasets.folder import DatasetFolder
from torchvision.datasets.vision import VisionDataset
from typing import Union, Callable, Any, Optional, Tuple, List, Dict
from pathlib import Path
import os
import numpy as np
import pandas as pd

#%%
class MineSiteDataset(MineSiteImageFolder):
    def __init__():
        pass
    
    

#%%    
class MineSiteImageFolder(VisionDataset):
    def __init__(self, root: Union[str, Path],
                 loader: Callable[[str], Any],
                 extensions: Optional[Tuple[str, ...]] = None,
                 transform: Optional[Callable] = None,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 allow_empty: bool = False,
                 ) -> None:
        super().__init__(transform, )
        self.root = root
        samples = self.make_dataset()  ## pass params
        
        self.samples = samples
        
        
    def find_classes() -> Tuple[List[str], Dict[str, int]]:
        """This returns a hard-coded binary classes and class to index.
            It is hardcorded because the classes and not provided in the file 
            but deduced from their binary nature and task at hand
        """
        
        classes = ["not_mining_site", "mining_site"]
        class_to_idx = {"not_mining_site": 0, "mining_site": 1}
        return classes, class_to_idx
    def get_all_image_bands(self, root: Optional[Union[str, Path]], img_name: Optional[Union[str, Path]],
                            img_idx: Optional[int] = 1
                            ) -> np.ndarray:
        if not root:
            root = self.root
            
        if not img_name and not img_idx:
            raise ValueError(f"img_name or img_idx must be provided")
        
        if not img_name:
            img_name = os.listdir(root)[img_idx]
        
        img_path = os.path.join(root, img_name)
        with rasterio.open(img_path) as src:
            bands = [src.read(i) for i in range(1, 13)]
            
        return bands
            
    def get_rgb_img(rgb_bands: List[int]):
        pass
    
    def make_dataset(self, root, 
                     target_file_path: Union[str, Path],
                     class_to_idx: Optional[Dict],
                     img_name: Optional[str], 
                     img_idx: Optional[int], fetch_for_all_classes: bool = True,
                     target_file_has_header = False,
                     
                     ):
        """Generate a list of samples of each class of the form (path_to_sample, class)

        Args:
            root (_type_): _description_
            target_file_path (Union[str, Path]): Csv file with first column as file name and second column as target
                                                If file does not have a header then set target_file_has_header = False
            img_name (_type_): _description_
            img_idx (_type_): _description_
            target_file_has_header (bool): _description_ = False

        Raises:
            ValueError: _description_
        """
        
        if not root:
            root = self.root
            
        if not class_to_idx:
            classes, class_to_idx = self.find_classes()
            
        if not target_file_has_header:
            target_file_df = pd.read_csv(target_file_path, header=None)
        else:
            target_file_df = pd.read_csv(target_file_path, header=None)          
        
        
        instances = []
        if fetch_for_all_classes:
            cls_idx = class_to_idx.values()
            cls_target_df = target_file_df[target_file_df[1].isin(cls_idx)]
            img_names, img_target_idx = cls_target_df[0].values, cls_target_df[1]
            img_path = [os.path.join(root, img_nm) for img_nm in img_names]
            
            for data_sample in zip(img_path, img_target_idx):
                instances.append(data_sample)
        else:    
            if not img_name and not img_idx:
                raise ValueError(f"""img_name or img_idx must be provided if you want to 
                                    fetch sample for a particular image. Or set fetch_for_all_classes: bool = True
                                 
                                 """)
            
            if not img_name:
                img_name = os.listdir(root)[img_idx]
            
            img_path = os.path.join(root, img_name)
            img_target_idx = target_file_df[target_file_df[0]==img_name][1].values[0]
            instances.append((img_path, img_target_idx))
            
        return instances
        
        

# Select bands 4, 3, 2 for RGB visualization
rgb_bands = np.dstack([bands[3], bands[2], bands[1]])

# Normalize bands to 0-255
rgb_bands = ((rgb_bands - rgb_bands.min()) / (rgb_bands.max() - rgb_bands.min()) * 255).astype(np.uint8)

        
    def __getitem__():
        pass
    
    def __len__(self) -> int:
        return super().__len__()()





# %%
from torchvision import get_image_backend

get_image_backend()
# %%
ans_path = "/home/lin/codebase/mine_sites/solafune_find_mining_sites/train/answer.csv"
df = pd.read_csv(ans_path, header=None)
# %%
df.iloc[0]
# %%
l = [1,2,3]
k = ["a","b","c"]

for g in zip(l,k):
    print(g)
# %%
