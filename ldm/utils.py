import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from torch.utils.data import DataLoader

import torch
from PIL import Image

def get_class_tensor(class_name):
    class_mapping = {"barack obama": 0, "cristiano ronaldo": 1, "donald trump": 2}
    lower_class_name = class_name.lower()
    if lower_class_name in class_mapping:
        class_idx = class_mapping[lower_class_name]
        return torch.tensor(class_idx)
    return None

def latents_to_pil(vae, latents):     
    latents = (1 / 0.18215) * latents     
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def init_attr(instance, locals):
    locals.pop('self', None)
    for k, v in locals.items():
        setattr(instance, k, v)

def dataloader(train_dataset, test_dataset, batch_size):
    return (
        DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True), 
        DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False),
    )

def show_images_grid(batch, grid_size=5):
    batch = batch.cpu()
    grid_images = vutils.make_grid(batch[:grid_size], nrow=grid_size, normalize=True)
    grid_images_np = grid_images.numpy()
    grid_images_np = np.transpose(grid_images_np, (1, 2, 0))
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_images_np)
    plt.axis("off")
    plt.show()

def show_image(image, idx, msg=None):
    single_image = image[idx].detach().cpu().permute(1, 2, 0).numpy()
    plt.imshow(single_image)
    plt.title(f'{msg} Image')
    plt.axis('off')
    plt.show()

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