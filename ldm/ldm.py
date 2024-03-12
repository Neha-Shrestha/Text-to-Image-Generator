import torch

import ldm_components
from inference import inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_of_generation = 1

ddpm_scheduler = ldm_components.ddpm_schedular()
vae = ldm_components.vae().to(device)
unet = ldm_components.unet().to(device)

unet.load_state_dict(torch.load("./saved/face_unet.pth", map_location=device))

def generate_image(text):
    if text == "": return "./images/error.jpg"
    return inference(unet, ddpm_scheduler, vae, device, text=text)

# def generate_image(text):
#     noisy_images = torch.randn((num_of_generation, 4, 32, 32)).to(device)
#     label = get_class_tensor(text)
#     if label == None: 
#         return "./error.jpg"
#     labels = label.expand(num_of_generation).to(device)

#     for timestep in ddpm_scheduler.timesteps:
#         with torch.inference_mode():
#             noise = unet(noisy_images, timestep, labels).sample
#         noisy_images = ddpm_scheduler.step(noise, timestep, noisy_images).prev_sample

#     images = latents_to_pil(vae, noisy_images)
#     return images[0]