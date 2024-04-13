

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

# %%
