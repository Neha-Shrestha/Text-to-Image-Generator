import os
import re

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

def latents_to_pil(vae, latents, save_path=None):
    latents = (1 / 0.18215) * latents     
    with torch.inference_mode():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    if save_path: 
        pil_images[0].save(save_path)
    return pil_images

def plot_images_from_folder(folder_path):
    save_path = "./images/inference_img.png"
    _, axes = plt.subplots(2, 6, figsize=(20, 8))
    image_files = sorted(
        os.listdir(folder_path), 
        key=lambda x: int(re.findall(r'\d+', x)[0]), 
        reverse=True
    )
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)
        ax = axes[i // 6, i % 6]
        ax.imshow(image)
        ax.set_title(f"t={image_file}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return Image.open(save_path)

def save_images(img_name, images, labels):
    if not os.path.exists("results"):
        os.makedirs("results")
    fig = plt.figure(figsize=(15, 3))
    for i in range(5):
        ax = fig.add_subplot(1, 5, i+1)
        ax.set_title(get_class_name(labels[i].item()))
        ax.imshow(images[i])
        ax.axis("off")
        plt.savefig(f"./images/output.png")
    plt.close()

def plot_results(result, title):
    batches = range(1, len(result) + 1)
    plt.plot(batches, result, label=title)
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig(f"./images/error_train.png")
    plt.close()