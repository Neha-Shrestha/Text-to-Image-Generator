import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def accuracy_fn(pred, y):
    return (pred.argmax(dim=1) == y).sum().item() / len(pred)

def show_images(dataset, class_names=None, rows=3, cols=3):
    fig = plt.figure(figsize=(9, 9))
    for i in range(1, rows * cols + 1):
        idx = torch.randint(0, len(dataset), size=[1]).item()
        fig.add_subplot(rows, cols, i)
        if class_names:
            image, label = dataset[idx]
            plt.title(class_names[label])
        else:
            image = dataset[idx]
        plt.imshow(image.squeeze(), cmap="gray")
        plt.axis(False)

def dataloader(train_dataset, test_dataset, batch_size):
    return (
        DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True), 
        DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False),
        train_dataset.classes
    )