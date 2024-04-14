

#%%
import rasterio
import matplotlib.pyplot as plt
import numpy as np

#%% Open the .tif file
train_0_path = "/home/lin/codebase/mine_sites/solafune_find_mining_sites/train/train/train_0.tif"
import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Open the .tif file
with rasterio.open(train_0_path) as src:
    # Read the 12 bands
    bands = [src.read(i) for i in range(1, 13)]

# Select bands 4, 3, 2 for RGB visualization
rgb_bands = np.dstack([bands[3], bands[2], bands[1]])

# Normalize bands to 0-255
rgb_bands = ((rgb_bands - rgb_bands.min()) / (rgb_bands.max() - rgb_bands.min()) * 255).astype(np.uint8)

# Plot the combined image
plt.imshow(rgb_bands)
plt.show()

# Plot each band
for i, band in enumerate(bands, start=1):
    plt.figure()
    plt.title(f'Band {i}')
    plt.imshow(band, cmap='gray')
    plt.show()

# %%
train_root = "/home/lin/codebase/mine_sites/solafune_find_mining_sites/train/train"


#%%
import os

os.listdir(train_root)[1]


# %%
