import numpy as np
import torch
from torch_snippets import Report


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
    #print(f"prediction.squeeze(): {prediction.squeeze()}")
    #print(f"y: {y}")
    prediction = prediction.squeeze()
    valid_loss = loss_fn(prediction, y)
    return valid_loss.item()


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



def trigger_training_process(train_dataload, val_dataload, model, loss_fn,
                             optimizer, num_epochs: int, device="cuda"
                             ):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    log = Report(n_epochs=num_epochs)
    
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




