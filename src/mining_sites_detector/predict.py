from data_preprocessor import get_tiff_img
import torch
from pathlib import Path
from typing import Union
#from mining_sites_detector import logger

def load_torchscript_model(model_path: Union[str, Path], device="cuda"):
    model = torch.jit.load(model_path, map_location=device)
    return model


def predict_image(model, img, device="cuda"):
    img_tensor = torch.tensor(img).permute(2,0,1).float()
    img_tensor = img_tensor.to("cuda").unsqueeze(0)
        
    model = model.to("cuda")
    with torch.no_grad():
        prediction = model(img_tensor)
    return prediction



def make_prediction(torchscript_model_path: Union[str, Path], 
                    img_path: Union[str, Path], 
                    device="cuda",
                    return_all_bands=True,
                    ):
    print(f"Loading image from: {img_path}")
    img = get_tiff_img(img_path, return_all_bands=return_all_bands)
    print(f"Loading model from: {torchscript_model_path}")
    model = load_torchscript_model(model_path=torchscript_model_path, device=device)
    print(f"Making prediction on image...")
    predicted_class = predict_image(model=model, img=img, 
                                    device=device
                                    )
    return predicted_class

