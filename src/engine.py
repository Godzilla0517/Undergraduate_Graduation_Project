import torch


def train_step(model, dataloader, loss_fn, optimizer, device, is_RNN=False):
    train_loss, train_acc = 0, 0
    model.train()
    for X_train, y_train in dataloader:
        if is_RNN is True:
            X_train, y_train = X_train.reshape(-1, 44, 64).to(device), y_train.to(device)
        else:
            X_train, y_train = X_train.to(device), y_train.to(device)            
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        train_loss += loss.item()
        train_acc += torch.eq(y_pred.argmax(dim=1), y_train).sum().item() / len(y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc



def val_step(model, dataloader, loss_fn, device, is_RNN=False):
    model.eval()
    val_loss, val_acc = 0, 0
    with torch.inference_mode():
        for X_val, y_val in dataloader:
            if is_RNN is True:
                X_val, y_val = X_val.reshape(-1, 44, 64).to(device), y_val.to(device)
            else:
                X_val, y_val = X_val.to(device), y_val.to(device)
            val_pred = model(X_val)
            val_loss += loss_fn(val_pred, y_val).item()
            val_acc += torch.eq(val_pred.argmax(dim=1), y_val).sum().item() / len(val_pred)
        val_loss /= len(dataloader)
        val_acc /= len(dataloader)
    return val_loss, val_acc



def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, epochs, device, is_RNN):
    
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model=model, 
            dataloader=train_dataloader, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            device=device, 
            is_RNN=is_RNN)
        scheduler.step()
        val_loss, val_acc = val_step(
            model=model, 
            dataloader=val_dataloader, 
            loss_fn=loss_fn, 
            device=device, 
            is_RNN=is_RNN) 
        print(f"Epoch: {epoch + 1} | "
              f"train_loss: {train_loss: .4f} | train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc) 
    return results
