import torch
from tqdm import tqdm
from utils import accuracy_fn

def fit(epochs, model, train_dataloader, test_dataloader, loss_fn, optimizer, device):
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
        
        print(f"| Epoch: {epoch} | Train loss: {train_loss / train_count } | Train accuracy: {train_acc / train_count} | Test loss: {test_loss / test_count} | Test accuracy: {test_acc / test_count} |")
    
    return (
        train_loss.item() / train_count,
        train_acc / train_count,
        test_loss.item() / test_count,
        test_acc / test_count
    )