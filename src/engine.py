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



def test_step(model, dataloader, loss_fn, device, is_RNN=False):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X_test, y_test in dataloader:
            if is_RNN is True:
                X_test, y_test = X_test.reshape(-1, 44, 64).to(device), y_test.to(device)
            else:
                X_test, y_test = X_test.to(device), y_test.to(device)
            test_pred = model(X_test)
            test_loss += loss_fn(test_pred, y_test).item()
            test_acc += torch.eq(test_pred.argmax(dim=1), y_test).sum().item() / len(test_pred)
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
    return test_loss, test_acc



def train(model, train_dataloader, test_dataloader, loss_fn, optimizer, scheduler, epochs, device, is_RNN):
    
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model=model, 
            dataloader=train_dataloader, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            device=device, 
            is_RNN=is_RNN)
        scheduler.step()
        test_loss, test_acc = test_step(
            model=model, 
            dataloader=test_dataloader, 
            loss_fn=loss_fn, 
            device=device, 
            is_RNN=is_RNN) 
        print(f"Epoch: {epoch + 1} | "
              f"train_loss: {train_loss: .4f} | train_acc: {train_acc:.4f} | "
              f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc) 
    return results