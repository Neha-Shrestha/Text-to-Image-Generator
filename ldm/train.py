import torch
from tqdm import tqdm
from utils import accuracy_fn

def fit(epochs, model, train_dataloader, test_dataloader, loss_fn, optimizer, scheduler, device):
    model.to(device)
    
    train_count, test_count = 0, 0
    train_loss, train_acc, test_loss, test_acc = 0, 0, 0, 0
    
    for epoch in tqdm(range(epochs)):
        model.train()
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y) 
            train_loss += loss
            train_acc += accuracy_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                test_pred = model(X)
                test_loss += loss_fn(test_pred, y)
                test_acc += accuracy_fn(pred=test_pred, y=y)
        
        train_count = len(train_dataloader)*(epoch+1)
        test_count = len(test_dataloader)*(epoch+1)
        scheduler.step()
        
        print(f"| Epoch: {epoch} | Train loss: {train_loss / train_count } | Train accuracy: {train_acc / train_count} | Test loss: {test_loss / test_count} | Test accuracy: {test_acc / test_count} |")
    
    return (
        train_loss.item() / train_count,
        train_acc / train_count,
        test_loss.item() / test_count,
        test_acc / test_count
    )

def fit_2(epochs, model, train_dataloader, test_dataloader, loss_fn, optimizer, device):
    for epoch in range(epochs):
        model.to(device)
        model.train()
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            loss = loss_fn(model(X), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            tot_loss, tot_acc, count = 0.,0.,0
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                n = len(X)
                count += n
                tot_loss += loss_fn(pred, y).item()*n
                tot_acc += accuracy_fn(pred, y).item()*n
        print("--------------------")
        print(count)
        print(tot_loss)
        print(tot_acc)
        print("--------------------")
        print(epoch, tot_loss/count, tot_acc/count)
        print("--------------------")
    return tot_loss/count, tot_acc/count

def fit_3(epochs, model, train_dataloader, test_dataloader, loss_fn, optimizer, device):
    result = dict.fromkeys(["mode", "epoch", "loss", "accuracy"])
    
    for epoch in range(epochs):
        model.to(device)
        model.train()
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            loss = loss_fn(model(X), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            tot_loss, tot_acc, count = 0.,0.,0
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss_metric.update(loss_fn(model(X), y))
                accuracy_metric.update(pred, y)
            result["mode"] = "test"
            result["epoch"] = epoch
            result["loss"] = loss_metric.compute().item()
            result["accuracy"] = accuracy_metric.compute().item()
            loss_metric.reset()
            accuracy_metric.reset()
        
        print(result)

# insert in train 

def fit_4(epochs, model, train_dataloader, test_dataloader, loss_fn, optimizer, scheduler, device):
    result = dict.fromkeys(["mode", "epoch", "loss", "accuracy"])
    
    for epoch in range(epochs):
        model.to(device)
        model.train()
        print(f"before: {scheduler.get_last_lr()}")
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            loss = loss_fn(model(X), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            for X, y in test_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss_metric.update(loss_fn(model(X), y))
                accuracy_metric.update(pred, y)
            result["mode"] = "test"
            result["epoch"] = epoch
            result["loss"] = loss_metric.compute().item()
            result["accuracy"] = accuracy_metric.compute().item()
            loss_metric.reset()
            accuracy_metric.reset()

        scheduler.step()
        print(f"after: {scheduler.get_last_lr()}")
        
        print(result)