
#%%
import tensorboard
import os
import tempfile
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchgeo.datamodules import EuroSAT100DataModule
from torchgeo.trainers import ClassificationTask
from torchgeo.models import ResNet18_Weights, ViTSmall16_Weights
import timm
from torchsummary import summary

# %%
batch_size = 10
num_workers = 2
max_epochs = 50
fast_dev_run = False

root = os.path.join(tempfile.gettempdir(), "eurosat100")
datamodule = EuroSAT100DataModule(root=root, batch_size=batch_size, num_workers=num_workers, download=True)

#%%
task = ClassificationTask(loss="ce", model="resnet18", 
                          weights=ResNet18_Weights.SENTINEL2_ALL_MOCO,
                          in_channels=13,
                          num_classes=10,
                          lr=0.1,
                          patience=5
                          )



# %%
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
default_root_dir = os.path.join(tempfile.gettempdir(), "experiments")
checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=default_root_dir,
                                      save_top_k=1, save_last=True
                                      )
early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00,
                                        patience=10
                                        )
logger = TensorBoardLogger(save_dir=default_root_dir, name="tutorial_logs")


# %%
trainer = Trainer(accelerator=accelerator, callbacks=[checkpoint_callback, early_stopping_callback],
                  fast_dev_run=fast_dev_run, log_every_n_steps=1,
                  logger=logger, min_epochs=1, max_epochs=max_epochs
                  )

#%%
trainer.fit(model=task, datamodule=datamodule)

# %%
%tensorboard --logdir "$default_root_dir"

# %%
trainer.test(model=task, datamodule=datamodule)
# %%
weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
# %%
weights.get_state_dict().keys()
# %%

in_chans = weights.meta["in_chans"]
model = timm.create_model("resnet18", in_chans=in_chans, num_classes=10)
# %%
model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
# %%
import numpy as np
summary(model.to("cuda"), input_size=(13,512,512))
# %%
vit_wet = ViTSmall16_Weights.SENTINEL2_ALL_DINO
# %%
in_chans = vit_wet.meta["in_chans"]
model = timm.create_model("vit_small_patch16_224", in_chans=in_chans, num_classes=10, pretrained=True)
model.load_state_dict(vit_wet.get_state_dict(progress=True), strict=False)
# %%
