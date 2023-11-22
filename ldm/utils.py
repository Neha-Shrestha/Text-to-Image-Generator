import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def init_attr(instance, locals):
    locals.pop('self', None)
    for k, v in locals.items():
        setattr(instance, k, v)

def dataloader(train_dataset, test_dataset, batch_size):
    return (
        DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True), 
        DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False),
        train_dataset.classes
    )

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

def plot_curves(results):
    epochs = range(len(results["train_loss"]))
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results["train_loss"], label="Train Loss")
    plt.plot(epochs, results["test_loss"], label="Test Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["train_acc"], label="Train Accuracy")
    plt.plot(epochs, results["test_acc"], label="Test Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()