

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
from typing import Union, Callable, Any, Optional, Tuple, List, Dict, cast
from pathlib import Path
import os
import numpy as np
import pandas as pd
import rasterio
from torch import Tensor
import torch
from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples
#%%
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
    
def get_tiff_img(path):
        with rasterio.open(path) as src:
            img_allbands = [src.read(band) for band in range(1, 13)]
        return img_allbands


class MineSiteImageFolder(VisionDataset):
    def __init__(self, root: Union[str, Path],
                 loader: Callable[[str], Any],
                 target_file_path: str,
                 extensions: Optional[Tuple[str, ...]] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 allow_empty: bool = False,
                 class_to_idx: Optional[Union[Dict, None]] = None,
                 fetch_for_all_classes = True, 
                 target_file_has_header = False,
                 img_name: Optional[Union[str, None]] = None,
                 img_idx: Optional[Union[int, None]] = None
                 ) -> None:
        super().__init__(transform)
        self.root = root
        self.target_file_path = target_file_path
        self.target_file_has_header = target_file_has_header
        self.fetch_for_all_classes = fetch_for_all_classes
        self.img_name = img_name
        self.img_idx = img_idx
        self.loader = loader
        self.target_transform = target_transform
        #self.is_valid_file = is_valid_file
        self.allow_empty = allow_empty
        
        if not class_to_idx:
            self.classes, self.class_to_idx = self.find_classes()
            class_to_idx = self.class_to_idx
        samples = self.make_dataset(root=self.root,
                                    target_file_path=self.target_file_path,
                                    class_to_idx=self.class_to_idx,
                                    img_name=self.img_name, img_idx=self.img_idx,
                                    target_file_has_header=self.target_file_has_header 
                                    )  ## pass params
        
        self.samples = samples
        
    @property    
    def find_classes() -> Tuple[List[str], Dict[str, int]]:
        """This returns a hard-coded binary classes and class to index.
            It is hardcorded because the classes and not provided in the file 
            but deduced from their binary nature and task at hand
        """
        
        classes = ["not_mining_site", "mining_site"]
        class_to_idx = {"not_mining_site": 0, "mining_site": 1}
        return classes, class_to_idx
    
    
    
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
    def _load_image(self, index) -> Tuple[Tensor, Tensor]:
        """Load a single image with its class label as tensor

        Args:
            index (_type_): _description_
            
        """
        img, label = self.__getitem__(index)
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).float()
        img_tensor = img_tensor.permute((2,0,1))
        label_tensor = torch.tensor(label).long()
        return img_tensor, label_tensor
        
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path=path)
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)
            
        return sample, target
    
    def __len__(self) -> int:
        return len(self.samples)
            
    
class MineSiteDataset(MineSiteImageFolder, RasterDataset):

    all_band_names = ("B01",
                        "B02",
                        "B03",
                        "B04",
                        "B05",
                        "B06",
                        "B07",
                        "B08",
                        "B8A",
                        "B09",
                        "B10",
                        "B11",
                        "B12",
                    )
    
    rgb_bands = ("B04", "B03", "B02")
    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}
    
    def __init__(self, root: Union[str, Path],
                 loader: Callable[[str], Any],
                 target_file_path: str,
                 extensions: Optional[Tuple[str, ...]] = None,
                 transforms: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 allow_empty: bool = False,
                 #class_to_idx: Optional[Union[Dict, None]] = None,
                 fetch_for_all_classes = True, 
                 target_file_has_header = False,
                 img_name: Optional[Union[str, None]] = None,
                 img_idx: Optional[Union[int, None]] = None,
                 bands = BAND_SETS["all"],
                 class_to_idx = None,
                ):
        self.root = root
        self.target_file_path = target_file_path
        self.target_file_has_header = target_file_has_header
        self.fetch_for_all_classes = fetch_for_all_classes
        self.img_name = img_name
        self.img_idx = img_idx
        self.loader = loader
        self.transforms = transforms
        self.bands = bands
        self.class_to_idx = class_to_idx
        self.band_indices = Tensor([self.all_band_names.index(b) for b in self.all_band_names]).long()
        
        super().__init__(root=self.root, target_file_path= self.target_file_path,
                         target_file_has_header=self.target_file_has_header,
                        fetch_for_all_classes=self.fetch_for_all_classes,
                        loader=self.loader, class_to_idx=self.class_to_idx
                        )
    
    
    
    def __getitem__(self, index):
        img, label = self._load_image(index)
        img = torch.index_select(img, dim=0, index=self.band_indices).float()
        sample = {"image": img, "label": label}
        if self.transforms:
            sample = self.transforms(sample)

        return sample
    
    def plot(self, sample, show_title):
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError(f"band {band} not found in existing bands which are {self.bands}")
            
        image = np.take(sample["image"].numpy(), indices=rgb_indices, axis=0)
        image = np.rollaxis(image, 0, 3)
        image = np.clip(image / 3000, 0, 1)   
        
        label = cast(int, sample["label"].item())
        label_class = self.classes[label]
        
        fig, ax = plt.subplot(figsize=(4,4))
        ax.imshow(image)
        ax.axis("off")
        
        if show_title:
            title = f"Label: {label_class}"
            ax.set_title(title)
            
        return fig

#%% load and plot data  

root = "/home/lin/codebase/mine_sites/solafune_find_mining_sites/train/train"   
target_file_path = "/home/lin/codebase/mine_sites/solafune_find_mining_sites/train/answer.csv"       
MineSiteDataset(root=root, target_file_path=target_file_path,
                target_file_has_header=False, loader=get_tiff_img,
                class_to_idx = {"not_mining_site": 0, "mining_site": 1}
                
                )


#%% Select bands 4, 3, 2 for RGB visualization
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
