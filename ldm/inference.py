import torch
from utils import get_class_tensor, latents_to_pil, save_images, plot_images_from_folder

def inference(unet, ddpm_scheduler, vae, device, num_images=1, text=None, img_name=None):
    noisy_images = torch.randn((num_images, 4, 32, 32)).to(device)
    if text:
        label = get_class_tensor(text)
        if label is None: 
            return "./images/error.jpg"
        labels = label.expand(num_images).to(device)
    else:
        labels = torch.randint(3, size=(noisy_images[:num_images].shape[0],)).to(device)

    for timestep in ddpm_scheduler.timesteps:
        unet.eval()
        img_checkpoints = [999, 700, 500, 200, 150, 120, 100, 80, 60, 40, 20, 0]
        with torch.inference_mode():
            noise = unet(noisy_images, timestep, labels).sample
        noisy_images = ddpm_scheduler.step(noise, timestep, noisy_images).prev_sample
        if timestep.item() in img_checkpoints and text:
            latents_to_pil(vae, noisy_images, save_path=f"./images/inference/{timestep}.png")

    images = latents_to_pil(vae, noisy_images)
    
    if text: 
        inference_img = plot_images_from_folder("./images/inference")
        return images[0], inference_img
    
    save_images(img_name, images, labels)