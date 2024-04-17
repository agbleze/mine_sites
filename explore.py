

#%%
import torchvision
import os
import tempfile
from torch.utils.data import DataLoader
from torchgeo.datasets import NAIP, ChesapeakeDE, stack_samples
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import RandomGeoSampler 

# %%
naip_root = os.path.join(tempfile.gettempdir(), "naip")

#%%
naip_url = (
    "https://naipeuwest.blob.core.windows.net/naip/v002/de/2018/de_060cm_2018/38075/"
)

tiles = [
    "m_3807511_ne_18_060_20181104.tif",
    "m_3807511_se_18_060_20181104.tif",
    "m_3807512_nw_18_060_20180815.tif",
    "m_3807512_sw_18_060_20180815.tif",
]

for tile in tiles:
    download_url(naip_url + tile, naip_root)
    
#%%
naip = NAIP(naip_root)


# %%
chesapeake_root = os.path.join(tempfile.gettempdir(), "chesapeake")
os.makedirs(chesapeake_root, exist_ok=True)
chesapeake = ChesapeakeDE(chesapeake_root, crs=naip.crs, res=naip.res, download=True)

#%%
dataset = naip & chesapeake

#%%
sampler = RandomGeoSampler(dataset, size=1000, length=10)

#%% dataloader
dataloader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)

#%% training
for sample in dataloader:
    image = sample["image"]
    target = sample["mask"]
# %%
