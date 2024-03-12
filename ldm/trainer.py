import torch
from torch import nn

from pathlib import Path

import ldm_components
from inference import inference
from data_processing import data_preprocessing
from utils import plot_results
from dataloader import latent_dataloader

# train_images_folder = Path("./images/train/")
# test_images_folder = Path("./images/test/")

# train_latent_images_folder = Path("./latent_images/train/")
# test_latent_images_image_folder = Path("./latent_images/test/")

# train_dataloader = latent_dataloader(Path("./latent_images/train/"), batch_size=20, shuffle=True)
# test_dataloader = latent_dataloader(Path("./latent_images/test/"), batch_size=20, shuffle=True)

# data_to_latents(vae, "./images/train/", Path("./latent_images/train/"), transform=transform, batch_size=5)
# data_to_latents(vae, "./images/test/", Path("./latent_images/test/"), transform=transform, batch_size=5)

# def main(epochs, loss_fn, optimizer, device, preprocess=False, data_folder=None, latent_folder=None):
def train(epochs, preprocess=False, data_folder=None, latent_folder=None):
    lr = 1e-3
    ddpm_scheduler = ldm_components.ddpm_schedular()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = ldm_components.vae().to((device))
    unet = ldm_components.unet().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    dataloader = latent_dataloader(Path("./data/face/latents/"), batch_size=20, shuffle=True)

    if preprocess: data_preprocessing(vae, data_folder, latent_folder)

    losses = []

    for epoch in range(epochs):
        train_loss = 0

        unet.train()
        for latent_images, labels in dataloader:
            latent_images, labels = latent_images.to(device), labels.to(device)

            timesteps = torch.randint(0, ddpm_scheduler.config.num_train_timesteps, (latent_images.shape[0], )).to(device)
            noise = torch.randn(latent_images.shape).to(device)
            noisy_images = ddpm_scheduler.add_noise(latent_images, noise, timesteps)

            pred_noise = unet(noisy_images, timesteps, labels).sample
            loss = loss_fn(pred_noise, noise)
            train_loss += loss.item()
            losses.append(train_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0: 
            inference(unet, ddpm_scheduler, vae, device, num_images=5, img_name=epoch)

        print(f"| Epoch: {epoch} | Loss: {train_loss:.2f} |")

    plot_results(losses, "Training Loss")
        