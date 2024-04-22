

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
plot.show(img_1)


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
from torchgeo.datasets.geo import NonGeoClassificationDataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchgeo.transforms import AugmentationSequential, indices
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split
from glob import glob
import os
import shutil
from torchgeo.samplers import RandomGeoSampler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.optim import SGD, Adam
import torch
from torchsummary import summary
from torchgeo.models import ResNet18_Weights, ViTSmall16_Weights
import timm
from torch_snippets import Report
# %%
ans_path = "/home/lin/codebase/mine_sites/solafune_find_mining_sites/train/answer.csv"
df = pd.read_csv(ans_path, header=None)

#%%

df[1].value_counts()




#%%

X = df[0]
y = df[1]

#%%
X_train, X_eval, y_train, y_eval = train_test_split(X.to_frame(), y.to_frame(), 
                                                    test_size=.3, stratify=y.to_frame()[1],
                                                    random_state=2024
                                                    )

#%%

y_train.value_counts()

y_eval.value_counts()

#%%
X_val, X_test, y_val, y_test = train_test_split(X_eval, y_eval, test_size=.3,
                                                random_state=2024, stratify=y_eval[1]
                                                )

#%%

y_val.value_counts()

#%%

y_test.value_counts()


#%%
train_image_names = X_train[0].values.tolist()
val_image_names = X_val[0].values.tolist()
test_image_names = X_test[0].values.tolist()

#%%
df[df[0].isin(train_image_names)].to_csv("train_answers.csv", columns=[0, 1])
df[df[0].isin(val_image_names)].to_csv("val_answers.csv", columns=[0, 1])
df[df[0].isin(test_image_names)].to_csv("test_answers.csv", columns=[0, 1])
#for img in os.listdir(root)

#%%
train_img_dir = "/home/lin/codebase/mine_sites/solafune_find_mining_sites/model_train_images"
val_img_dir = "/home/lin/codebase/mine_sites/solafune_find_mining_sites/model_val_images"
test_img_dir = "/home/lin/codebase/mine_sites/solafune_find_mining_sites/model_test_images"
#root_glob = f"{root}/*"

train_img_dir, val_img_dir
#%%
for img_path in glob(root_glob):
    img_name = os.path.basename(img_path)
    if img_name in train_image_names:
        shutil.copy(img_path, train_img_dir)
    elif img_name in val_image_names:
        shutil.copy(img_path, val_img_dir)
    elif img_name in test_image_names:
        shutil.copy(img_path, test_img_dir)
    else:
        print(f"Image does not belong to a split: {img_path}")
    

#%%
# def get_all_image_bands(self, root: Optional[Union[str, Path]], img_name: Optional[Union[str, Path]],
#                             img_idx: Optional[int] = 1
#                             ) -> np.ndarray:
#         if not root:
#             root = self.root
            
#         if not img_name and not img_idx:
#             raise ValueError(f"img_name or img_idx must be provided")
        
#         if not img_name:
#             img_name = os.listdir(root)[img_idx]
        
#         img_path = os.path.join(root, img_name)
#         with rasterio.open(img_path) as src:
#             bands = [src.read(i) for i in range(1, 13)]
            
#         return bands
    
# def get_tiff_img(path):
#         with rasterio.open(path) as src:
#             img_allbands = [src.read(band) for band in range(1, 13)]
#         dstacked_bands = np.dstack([band for band in img_allbands])
#         return img_allbands

def get_tiff_img(path, return_all_bands, bands=("B01", "B03", "B02"),
                 normalize_bands=True
                ):
    all_band_names = ("B01","B02", "B03","B04","B05", "B06",
                      "B07","B08","B8A","B09","B11","B12"
                    )
    if return_all_bands:
        band_indexs = [all_band_names.index(band_name) for band_name in all_band_names]
    
    else:
        band_indexs = [all_band_names.index(band_name) for band_name in bands]
    #print(band_indexs)
    with rasterio.open(path) as src:
        img_bands = [src.read(band) for band in range(1,13)]
    dstacked_bands = np.dstack([img_bands[band_index] for band_index in band_indexs])
    #dstacked_bands = np.dstack([img_bands[3], img_bands[2], img_bands[1]])
    if normalize_bands:
        # Normalize bands to 0-255
        dstacked_bands = ((dstacked_bands - dstacked_bands.min()) / 
                          (dstacked_bands.max() - dstacked_bands.min()) * 255
                          ).astype(np.uint8)

    return dstacked_bands
#%%
class MineSiteImageFolder(Dataset):
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
                 img_idx: Optional[Union[int, None]] = None,
                 return_all_bands=False,
                 bands=("B04", "B03", "B02"),
                 normalize_bands=True
                 ) -> None:
        #super().__init__(transform)
        self.root = root
        self.target_file_path = target_file_path
        self.target_file_has_header = target_file_has_header
        self.fetch_for_all_classes = fetch_for_all_classes
        self.img_name = img_name
        self.img_idx = img_idx
        self.loader = loader
        self.target_transform = target_transform
        self.transform = transform
        #self.is_valid_file = is_valid_file
        self.allow_empty = allow_empty
        self.return_all_bands = return_all_bands
        self.bands = bands
        self.normalize_bands = normalize_bands
          
        if not class_to_idx:
            self.classes, self.class_to_idx = self.find_classes()
            class_to_idx = self.class_to_idx
        samples = self.make_dataset(root=self.root,
                                    target_file_path=self.target_file_path,
                                    class_to_idx=self.class_to_idx,
                                    img_name=self.img_name, img_idx=self.img_idx,
                                    target_file_has_header=self.target_file_has_header,
                                    )  ## pass params
        
        self.samples = samples
        
 
    def find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """This returns a hard-coded binary classes and class to index.
            It is hardcorded because the classes and not provided in the file 
            but deduced from their binary nature and task at hand
        """
        
        classes = ["not_mining_site", "mining_site"]
        class_to_idx = {"not_mining_site": 0, "mining_site": 1}
        return classes, class_to_idx
    
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
            target_file_df = pd.read_csv(target_file_path)          
        
        
        instances = []
        if fetch_for_all_classes:
            #print(f"in fetch_for_all_classes: {fetch_for_all_classes}")
            cls_idx = class_to_idx.values()
            #print(target_file_df)
            columns = target_file_df.columns.to_list()
            if len(columns) > 2:
                target_file_df = target_file_df.drop(columns[0], axis=1)
                columns = columns[1:]
            
            cls_target_df = target_file_df[target_file_df[columns[1]].isin(cls_idx)]
            #(f"cls_target_df: {cls_target_df}")
            img_names, img_target_idx = cls_target_df[columns[0]].values, cls_target_df[columns[1]].values
            img_path = [os.path.join(root, img_nm) for img_nm in img_names]
            
            for data_sample in zip(img_path, img_target_idx):
                instances.append(data_sample)
        else: 
            #print(f"Second Not in fetch_for_all_classes: {fetch_for_all_classes}")   
            if not img_name and not img_idx:
                raise ValueError(f"""img_name or img_idx must be provided if you want to 
                                    fetch sample for a particular image. Or set fetch_for_all_classes: bool = True
                                 
                                 """)
            
            if not img_name:
                img_name = os.listdir(root)[img_idx]
            
            img_path = os.path.join(root, img_name)
            img_target_idx = target_file_df[target_file_df[0]==img_name][1].values[0]
            instances.append((img_path, img_target_idx))
            
        return instances#[10]
        
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path=path,
                             return_all_bands=self.return_all_bands,
                             bands=self.bands,
                             normalize_bands=self.normalize_bands
                            )
        #sample_dstack = np.dstack([band for band in sample])
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)
            
        return sample, target
    
    def __len__(self) -> int:
        return len(self.samples)
#%%            
class NonGeoMineSiteClassificationDataset(MineSiteImageFolder):
    def __init__(self, root: str,
                 loader,
                 target_file_path: str,
                 is_valid_file = None,
                 transforms = None,
                 extensions: Optional[Tuple[str, ...]] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 allow_empty: bool = False,
                 class_to_idx: Optional[Union[Dict, None]] = None,
                 fetch_for_all_classes = True, 
                 target_file_has_header = False,
                 img_name: Optional[Union[str, None]] = None,
                 img_idx: Optional[Union[int, None]] = None,
                 return_all_bands=False,
                 bands=("B04", "B03", "B02"),
                 normalize_bands=True
                 ) -> None:
        self.root = root
        self.loader = loader
        self.target_file_path = target_file_path
        self.target_file_has_header = target_file_has_header
        self.transforms = transforms
        self.transform = transform
        self.return_all_bands = return_all_bands
        self.bands = bands
        self.normalize_bands = normalize_bands
        super().__init__(
                        root=root,
                        is_valid_file=is_valid_file,
                        loader=loader, class_to_idx=class_to_idx,
                        target_file_path=target_file_path,
                        target_file_has_header=target_file_has_header,
                        transform=transform,
                        fetch_for_all_classes=fetch_for_all_classes,
                        img_name=img_name, img_idx=img_idx,
                        target_transform=target_transform, allow_empty=allow_empty,
                        return_all_bands=self.return_all_bands, bands=self.bands,
                        normalize_bands=self.normalize_bands
                        
                    )
        
        self.mnfolder = MineSiteImageFolder(root=self.root, target_file_path=self.target_file_path,
                                            target_file_has_header=self.target_file_has_header,
                                            loader=self.loader,return_all_bands=self.return_all_bands, 
                                            bands=self.bands,
                                            normalize_bands=self.normalize_bands
                                            )
        #self.transforms = transforms
        
    def _load_image(self, index) -> Tuple[Tensor, Tensor]: # take-out, import NonGeoClassificationDataset and use to override there
        """Load a single image with its class label as tensor

        Args:
            index (_type_): _description_
            
        """
        self.img, self.label = self.mnfolder[index]
        #img_array = np.array(img)
        img_tensor = torch.tensor(self.img).permute(2,0,1).float()
        #img_tensor = img_tensor.permute(2,0,1)
        label_tensor = torch.tensor(self.label).float()
        return img_tensor, label_tensor
    
    #def __len__(self) -> int:
    #    return len(self.img)
    
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        image, label = self._load_image(index)
        sample = {"image": image, "label": label}
        
        if self.transforms:
            sample = self.transforms(sample)
        return sample
        
#%%    
class MineSiteDataset(NonGeoMineSiteClassificationDataset):

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
                        "B11",
                        "B12",
                    )
    
    rgb_bands = ("B04", "B03", "B02")
    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}
    separate_files = False
    
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
                 #bands = BAND_SETS["all"],
                 class_to_idx = None,
                 return_all_bands=False, 
                 bands=("B04", "B03", "B02"),
                normalize_bands=True,
                 #index=1
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
        self.return_all_bands = return_all_bands
        self.normalize_bands = normalize_bands
        #self.index=index
        super().__init__(root=self.root, target_file_path= self.target_file_path,
                         target_file_has_header=self.target_file_has_header,
                        fetch_for_all_classes=self.fetch_for_all_classes,
                        loader=self.loader, class_to_idx=self.class_to_idx,
                        is_valid_file=is_valid_file, transform=transforms,
                        return_all_bands=return_all_bands,
                        bands=bands,
                        normalize_bands=self.normalize_bands,
                        )
    
    
    
    def __getitem__(self,index):#, index=None):
        #if not index:
        #    index = self.index
        img, label = self._load_image(index)
        #img = torch.index_select(img, dim=0, index=self.band_indices).float()
        #img = np.dstack([band for band in img])
        self.sample = {"image": img, "label": label}
        if self.transforms:
            self.sample = self.transforms(self.sample)

        return self.sample
    def plot(self, index):
        self._load_image(index)
        plt.imshow(self.img)
        plt.show()
        
    
    def _plot(self, sample, show_title):
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError(f"band {band} not found in existing bands which are {self.bands}")
            
        image = np.take(sample["image"].numpy(), indices=rgb_indices, axis=0)
        #rgb_bands = np.dstack([image[0], image[1], image[2]])
        #image = ((rgb_bands - rgb_bands.min()) / (rgb_bands.max() - rgb_bands.min()) * 255).astype(np.uint8)
        image = np.rollaxis(image, 0, 3)
        #image = np.clip(image / 3000, 0, 1)   
        
        label = cast(int, sample["label"].item())
        label_class = self.classes[label]
        
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(image)
        ax.axis("off")
        
        if show_title:
            title = f"Label: {label_class}"
            ax.set_title(title)
            
        return fig

#%% load and plot data  

train_target_file_path = "/home/lin/codebase/mine_sites/train_answers.csv"
val_target_file_path = "/home/lin/codebase/mine_sites/val_answers.csv"
train_root = "/home/lin/codebase/mine_sites/solafune_find_mining_sites/model_train_images"   
target_file_path = "/home/lin/codebase/mine_sites/solafune_find_mining_sites/train/answer.csv"  

#%%
ex_df = pd.read_csv(val_target_file_path, header=None)

if 'Unnamed: 0' in ex_df.columns:
    ex_df_drp = ex_df.drop('Unnamed: 0', axis=1)
#%%
     
mnds = MineSiteDataset(root=train_root, target_file_path=train_target_file_path,
                target_file_has_header=True, loader=get_tiff_img,
                return_all_bands=True
                
                )

#%%
val_root = "/home/lin/codebase/mine_sites/solafune_find_mining_sites/model_val_images"
val_ans = "/home/lin/codebase/mine_sites/val_answers.csv"
val_mnds = MineSiteDataset(root=val_root, target_file_path=val_ans,
                target_file_has_header=True, loader=get_tiff_img,
                return_all_bands=True  
                )
#%%
batch_size = 4
num_workers = 2

#%% load EuroSAT MS dataset and dataloader
#root = os.path.join(tempfile.gettempdir(), "eurosat100")
#dataset = EuroSAT100(root, download=True)
dataloader = DataLoader(mnds, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers,
                        collate_fn=stack_samples
                        )

dataloader = iter(dataloader)
print(f"Number of images in dataset: {len(mnds)}")
print(f"Dataset Classes: {mnds.classes}")

#%%

batch = next(dataloader)
x, y = batch["image"], batch["label"]
print(x.shape, x.dtype, x.min(), x.max())

#x.to("cuda")
# %% compute indices and append as additional channel
transform = indices.AppendNDVI(index_nir=7, index_red=3)
batch = next(dataloader)
x = batch["image"]
print(x.shape)
x = transform(x)
print(x.shape)


#%%  ####   load a sample and batch of images and labels
mnds_sample = mnds[3]

#%%
#mnds.plot(index=3)
#mnds.plot(sample=mnds_sample, show_title="visualize")
#%%
x, y = mnds_sample["image"], mnds_sample["label"]
print(x.shape, x.dtype, x.min(), x.max())
#print(y, mnds.classes[y])

#%%
img_samp  = mnds_sample.get("image")

#%%
resnet18 = models.resnet18(pretrained=True)

#%%
weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
weights.get_state_dict().keys()

#%%
type(weights)
# %%

in_chans = weights.meta["in_chans"]
#%%
model = timm.create_model("resnet18", in_chans=in_chans, num_classes=2)
# %%
model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
# %%
import numpy as np
summary(model.to("cuda"), input_size=(13,512,512))

#%%
device = "cuda" if torch.cuda.is_available() else "cpu"
def conv_layer(in_chan, out_chan, kernel_size, stride=1):
    return nn.Sequential(
                    nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                              kernel_size=kernel_size, 
                              stride=stride
                              ),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=out_chan),
                    nn.MaxPool2d(kernel_size=2)
                )

def get_model():
    model = nn.Sequential(conv_layer(12, 64, 3),
                          conv_layer(64, 512, 3),
                          conv_layer(in_chan=512, out_chan=512, kernel_size=3),
                          conv_layer(in_chan=512, out_chan=512, kernel_size=3),
                          conv_layer(in_chan=512, out_chan=512, kernel_size=3),
                          conv_layer(512, 512, 3),
                          nn.Flatten(),
                          nn.Linear(18432, 1),
                          nn.Sigmoid()
                          ).to(device)
    
    loss_fn = nn.BCELoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer

#%%

model, loss_fn, optimizer = get_model()


#%%
summary(model, input_size=(12, 512, 512))


#%%
def train_batch(x, y, model, loss_fn, optimizer):
    model.train()
    prediction = model(x)
    #print(f"prediction: {prediction}")
    #print(f"prediction.shape: {prediction.shape}")
    #print(f"y.shape: {y.shape}")
    #print(f"y: {y}")
    batch_loss = loss_fn(prediction.squeeze(), y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

#%%
@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    is_correct = (prediction > 0.5).squeeze() 
    is_correct = (is_correct == y)
    return is_correct.cpu().numpy().tolist()

#%%
@torch.no_grad()
def val_loss(x, y, model):
    model.eval()
    prediction = model(x)
    #print(f"prediction.squeeze(): {prediction.squeeze()}")
    #print(f"y: {y}")
    prediction = prediction.squeeze()
    valid_loss = loss_fn(prediction, y)
    return valid_loss.item()

#%%
train_target_file_path = "/home/lin/codebase/mine_sites/train_answers.csv"
val_target_file_path = "/home/lin/codebase/mine_sites/val_answers.csv"
#%%
def get_data(train_img_dir, val_image_dir, 
             train_target_file_path=train_target_file_path,
             val_target_file_path=val_target_file_path,
             target_file_has_header=True, 
             loader=get_tiff_img,
             return_all_bands=True,
             batch_size=10, drop_last=True,
             num_workers=4
            ):
    train_dataset = MineSiteDataset(root=train_img_dir, 
                                    target_file_path=train_target_file_path,
                                    target_file_has_header=target_file_has_header, 
                                    loader=loader,
                                    return_all_bands=return_all_bands
                                    
                                    )
    val_dataset = MineSiteDataset(root=val_image_dir, 
                                  target_file_path=val_target_file_path,
                                    target_file_has_header=target_file_has_header, 
                                    loader=loader,
                                    return_all_bands=return_all_bands
                                    
                                    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  collate_fn=stack_samples, drop_last=drop_last
                                  )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers,
                                collate_fn=stack_samples, drop_last=drop_last
                                )
    
    return train_dataloader, val_dataloader
    

#%%
num_epochs=10
log = Report(n_epochs=num_epochs)
def trigger_training_process(train_dataload, val_dataload, model, loss_fn,
                             optimizer, num_epochs=num_epochs, device="cuda"
                             ):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    
    
    for epoch in range(num_epochs):
        train_epoch_losses, train_epoch_accuracies = [], []
        val_epoch_accuracies, val_epoch_losses = [], []
        #_n = len(train_dataload)
        for ix, batch in enumerate(iter(train_dataload)):
            x, y = batch["image"].to(device), batch["label"].to(device)
            batch_loss = train_batch(x, y, model, loss_fn, optimizer)
            train_epoch_losses.append(batch_loss)
        train_epoch_loss = np.array(train_epoch_losses).mean()
        
        for ix, batch in enumerate(iter(train_dataload)):
            x, y = batch["image"].to(device), batch["label"].to(device)
            is_correct = accuracy(x, y, model)
            train_epoch_accuracies.extend(is_correct)
        train_epoch_accuracy = np.mean(train_epoch_accuracies)
        
        for ix, batch in enumerate(iter(val_dataload)):
            x, y = batch["image"].to(device), batch["label"].to(device)
            val_is_correct = accuracy(x, y, model)
            val_epoch_accuracies.extend(val_is_correct)
            val_batch_loss = val_loss(x, y, model)
            val_epoch_losses.append(val_batch_loss)
            
        val_epoch_loss = np.mean(val_epoch_losses)
        val_epoch_accuracy = np.mean(val_epoch_accuracies)
        
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_accuracies.append(val_epoch_accuracy)
        val_losses.append(val_epoch_loss)
        
        log.record(pos=epoch+1, trn_loss=train_epoch_loss,
                   trn_acc=train_epoch_accuracy,
                   val_acc=val_epoch_accuracy,
                   val_loss=val_epoch_loss,
                   end="\r"
                   )
        log.report_avgs(epoch+1)
    return {"train_loss": train_losses,
            "train_accuracy": train_accuracies,
            "valid_loss": val_losses,
            "valid_accuracy": val_accuracies
            }



#%%
def plot_loss(train_metric, val_metric, title, num_epochs=10, ylabel="acc"):
    epochs = np.arange(num_epochs)+1
    plt.subplot(111)
    plt.plot(epochs, train_metric, "bo", label="train")
    plt.plot(epochs, val_metric, "r", label="valid")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel) 
    plt.title(title)
    plt.legend()
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
    plt.show()
    
    
#%%

train_dl, val_dl = get_data(train_img_dir=train_img_dir, 
                            val_image_dir=val_img_dir,
                            target_file_has_header=True, loader=get_tiff_img,
                            return_all_bands=True, batch_size=10
                            )

#%%


result = trigger_training_process(train_dataload=train_dl, val_dataload=val_dl,
                                  model=model, loss_fn=loss_fn,
                                  optimizer=optimizer, num_epochs=200
                                  )

#%%
log.plot_epochs("trn_acc, val_acc".split(","))


#%%

with open()
#%%
for varname in optimizer.state_dict():
    print(varname, "\t", optimizer.state_dict()[varname])


#%%
+233 244 638339
#%%
import pandas as pd

train_img_dir, val_img_dir
#%%

mnds[100].plot()
#%%
#image_array = np.transpose(mnds_sample.get("image"), (1, 2, 0))
plt.imshow(img_samp)
plt.show()
#%%

np.stack([n for n in mnds_sample["image"]]).shape

#%%

np.dstack(mnds_sample["image"]).size
#%
# %%  test dataloader for loading batches of images
# image is o shape (4, 13, 64, 64) -> (batch_num, channels(bands), height, width)
batch = next(dataloader)
x, y = batch["image"], batch["label"]
print(x.shape, x.dtype, x.min(), x.max())
print(y, [mnds.classes[i] for i in y])
#%%
#RandomGeoSampler(dataset=mnds, size=512, length=10)

#%%
mnds[0]
#%% Select bands 4, 3, 2 for RGB visualization
rgb_bands = np.dstack([bands[3], bands[2], bands[1]])

# Normalize bands to 0-255
rgb_bands = ((rgb_bands - rgb_bands.min()) / (rgb_bands.max() - rgb_bands.min()) * 255).astype(np.uint8)


# %%
from torchvision import get_image_backend

get_image_backend()


# %%
df.iloc[0]
# %%
l = [1,2,3]
k = ["a","b","c"]

for g in zip(l,k):
    print(g)
# %%
