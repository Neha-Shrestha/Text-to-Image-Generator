import torch

def train(unet, ddpm_scheduler, loss_fn, optimizer, device, dataloader):
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()