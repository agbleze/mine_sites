

#%%
import rasterio
import matplotlib.pyplot as plt
import numpy as np

#%% Open the .tif file
train_0_path = "/home/lin/codebase/mine_sites/solafune_find_mining_sites/train/train/train_3.tif"
import rasterio
import matplotlib.pyplot as plt
import numpy as np

#%% Open the .tif file
with rasterio.open(train_0_path) as src:
    # Read the 12 bands
    bands = [src.read(i) for i in range(1, 13)]
#%%
# Select bands 4, 3, 2 for RGB visualization
rgb_bands = np.dstack([bands[3], bands[2], bands[1]])

rgb_bands.shape
# Normalize bands to 0-255
rgb_bands = ((rgb_bands - rgb_bands.min()) / (rgb_bands.max() - rgb_bands.min()) * 255).astype(np.uint8)
#%%
# Plot the combined image
plt.imshow(rgb_bands)
plt.show()

#%%
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
    print(band_indexs)
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
default_rgb_bands = get_tiff_img(path=train_0_path, return_all_bands=False)

#%%
plt.imshow(default_rgb_bands)
plt.show()


#%%

np.dstack([band for band in bands]).shape


#%%
all_band_names[10,12]
#%%

np.stack(([bands[3], bands[2], bands[1]])).shape

#%%
# Normalize bands to 0-255
rgb_bands = ((rgb_bands - rgb_bands.min()) / (rgb_bands.max() - rgb_bands.min()) * 255).astype(np.uint8)
#%%
# Plot the combined image
plt.imshow(rgb_bands)
plt.show()

#%%
all_bands = np.dstack([band for band in bands])

# Plot each band
# for i, band in enumerate(bands, start=1):
#     plt.figure()
#     plt.title(f'Band {i}')
#     plt.imshow(band, cmap='gray')
#     plt.show()

# %%
train_root = "/home/lin/codebase/mine_sites/solafune_find_mining_sites/train/train"


#%%
import os

os.listdir(train_root)[1]

#%%

import numpy as np
# %%
np.array(rgb_bands).shape
# %%
