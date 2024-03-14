import torch

import ldm_components
from inference import inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ddpm_scheduler = ldm_components.ddpm_schedular()
vae = ldm_components.vae().to(device)
unet = ldm_components.unet().to(device)

unet.load_state_dict(torch.load("./saved/face_unet.pth", map_location=device))

def generate_image(text):
    if text == "": return "./images/error.jpg"
    return inference(unet, ddpm_scheduler, vae, device, text=text)