import numpy as np
import torch
from torch_snippets import Report
import torch.nn as nn
import os
from copy import deepcopy
from tqdm import tqdm

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    is_correct = (prediction > 0.5).squeeze() 
    is_correct = (is_correct == y)
    return is_correct.cpu().numpy().tolist()


@torch.no_grad()
def val_loss(x, y, model, loss_fn):
    model.eval()
    prediction = model(x)
    prediction = prediction.squeeze()
    valid_loss = loss_fn(prediction, y)
    return valid_loss.item()


def train_batch(x, y, model, loss_fn, optimizer):
    optimizer.zero_grad()
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction.squeeze(), y)
    batch_loss.backward()
    optimizer.step()
    #optimizer.zero_grad()
    return batch_loss.item()


def trigger_training_process(train_dataload, val_dataload, model, loss_fn,
                             optimizer, num_epochs: int, device="cuda",
                             model_store_dir="model_store", 
                             model_name="mining_site_detector_model",
                             checkpoint_interval: int = 1
                             ):
    os.makedirs(model_store_dir, exist_ok=True)
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    log = Report(n_epochs=num_epochs)
    
    for epoch in range(num_epochs):
        train_epoch_losses, train_epoch_accuracies = [], []
        val_epoch_accuracies, val_epoch_losses = [], []
        for ix, batch in enumerate(iter(train_dataload)):
            x, y = batch["image"].to(device), batch["label"].to(device)
            model.to(device)
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
            val_batch_loss = val_loss(x, y, model, loss_fn=loss_fn)
            val_epoch_losses.append(val_batch_loss)
            
        val_epoch_loss = np.mean(val_epoch_losses)
        val_epoch_accuracy = np.mean(val_epoch_accuracies)
        
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_accuracies.append(val_epoch_accuracy)
        val_losses.append(val_epoch_loss)
        
        #if (epoch + 1) % 1 == 0:
        log.record(pos=epoch+1, trn_loss=train_epoch_loss,
                   trn_acc=train_epoch_accuracy,
                   val_acc=val_epoch_accuracy,
                   val_loss=val_epoch_loss,
                   end="\r"
                   )
        log.report_avgs(epoch+1)
        
        if (epoch +1) % checkpoint_interval == 0:
            model_path = os.path.join(model_store_dir, f'{model_name}_epoch_{epoch+1}.pth')
            torch.save(deepcopy(model.to("cpu").state_dict()), model_path)
            
            # save model in state for infernece / resuming training
            print("saving model as checkpoint")
            resume_model_path = os.path.join(model_store_dir, 
                                             f'{model_name}_resumable_epoch_{epoch+1}.pth'
                                             )
            torch.save({"epoch": epoch+1,
                        "model_state_dict": deepcopy(model.to("cpu").state_dict()),
                        "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                        "loss": deepcopy(val_epoch_loss),
                        },
                       resume_model_path
                       )
            
            # save model as torchscript file for easy loading
            print("Exporting to torchscript")
            torchscript_model_path = os.path.join(model_store_dir, 
                                             f'{model_name}_torchscript_epoch_{epoch+1}.pt'
                                             )
            model_scripted = torch.jit.script(deepcopy(model.to("cpu")))
            model_scripted.save(torchscript_model_path)       
        
    return {"train_loss": train_losses,
            "train_accuracy": train_accuracies,
            "valid_loss": val_losses,
            "valid_accuracy": val_accuracies
            }



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

