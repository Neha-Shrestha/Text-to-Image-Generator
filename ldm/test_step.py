import torch

def test(unet, ddpm_scheduler, loss_fn, device, dataloader):
    test_loss = 0

    unet.eval()
    with torch.inference_mode():
        for latent_images, labels in dataloader:
            latent_images, labels = latent_images.to(device), labels.to(device)

            timesteps = torch.randint(0, ddpm_scheduler.config.num_train_timesteps, (latent_images.shape[0], )).to(device)
            noise = torch.randn(latent_images.shape).to(device)
            noisy_images = ddpm_scheduler.add_noise(latent_images, noise, timesteps)

            pred_noise = unet(noisy_images, timesteps, labels).sample
            loss = loss_fn(pred_noise, noise)
            test_loss += loss.item()
        
        return test_loss / len(dataloader)