import os

import torch
import matplotlib.pyplot as plt
from PIL import Image

def get_class_tensor(class_name):
    class_mapping = {"barack obama": 0, "cristiano ronaldo": 1, "donald trump": 2}
    lower_class_name = class_name.lower()
    if lower_class_name in class_mapping:
        class_idx = class_mapping[lower_class_name]
        return torch.tensor(class_idx)
    return None

def get_class_name(class_idx):
    class_mapping = {0: "barack obama", 1: "cristiano ronaldo", 2: "donald trump"}
    if class_idx in class_mapping:
        return class_mapping[class_idx]
    return None

def latents_to_pil(vae, latents):
    latents = (1 / 0.18215) * latents     
    with torch.inference_mode():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def save_images(img_name, images, labels):
    if not os.path.exists("results"):
        os.makedirs("results")
    fig = plt.figure(figsize=(15, 3))
    for i in range(5):
        ax = fig.add_subplot(1, 5, i+1)
        ax.set_title(get_class_name(labels[i].item()))
        ax.imshow(images[i])
        ax.axis("off")
        plt.savefig(f"results/image_{img_name}.png")
    plt.close()

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

def plot_results(result, title):
    batches = range(1, len(result) + 1)
    plt.plot(batches, result, label=title)
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()