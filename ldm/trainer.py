import torch
from torch import nn

import gradio as gr

from pathlib import Path

import ldm_components
from inference import inference
from data_processing import data_preprocessing
from utils import plot_results
from dataloader import latent_dataloader

continue_training = True

def train(
    epochs, 
    batch_size, 
    lr, 
    loss_fn, 
    optimizer, 
    latent_folder=None,
    preprocess=False, 
    data_folder=None,
    progress=gr.Progress()
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ddpm_scheduler = ldm_components.ddpm_schedular()
    vae = ldm_components.vae().to((device))
    unet = ldm_components.unet().to(device)

    loss_fn = nn.MSELoss() if loss_fn == "MSELoss" else nn.L1Loss()
    if optimizer == "Adam":
        optimizer = torch.optim.AdamW(unet.parameters(), lr=float(lr))
    else:
        optimizer = torch.optim.SGD(unet.parameters(), lr=float(lr), momentum=0.999, weight_decay=0.0001)
    
    if preprocess and data_folder: data_preprocessing(vae, data_folder, latent_folder)

    dataloader = latent_dataloader(Path("./data/face/latents/"), batch_size=int(batch_size), shuffle=True)

    losses = []

    best_loss = float("inf")
    best_model_path = "./saved/best_model.pth"

    print_loss = 0
    global continue_training
    for epoch in range(int(epochs)):
        train_loss = 0

        unet.train()
        for latent_images, labels in progress.tqdm(dataloader, desc=f"Training: Epoch: {epoch+1} |  Training Loss: {print_loss:.3f} |"):
            if not continue_training: break
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
        
        if not continue_training: break
        
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(unet.state_dict(), best_model_path)

        if epoch % 10 == 0:
            inference(unet, ddpm_scheduler, vae, device, num_images=5, img_name=epoch)

        print_loss = train_loss / len(dataloader)
        print(f"Epoch: {epoch} | Loss: {print_loss:.3f}")

    plot_results(losses, "Training Loss")
    return "Training Complete" if continue_training else "Training Cancelled"

def cancel():
    global continue_training
    continue_training = False