import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
num_of_generation = 1

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).requires_grad_(False)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

unet = UNet2DModel(
    sample_size=32,
    in_channels=4,
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(32, 64, 128, 256),
    down_block_types=(
        "DownBlock2D", 
        "AttnDownBlock2D", 
        "AttnDownBlock2D", 
        "AttnDownBlock2D"
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D", 
        "AttnUpBlock2D", 
        "UpBlock2D"
    ),
    class_embed_type="timestep"
).to(device)

unet.load_state_dict(torch.load("./models/face_unet.pth"))

def generate_image(text):
    unet.eval()
    noisy_images = torch.randn((num_of_generation, 4, 32, 32)).to(device)
    label = get_class_tensor(text)
    if label == None:
        return "./error.jpg"
    labels = label.expand(num_of_generation).to(device)

    for timestep in noise_scheduler.timesteps:
        with torch.inference_mode():
            noise = unet(noisy_images, timestep, labels).sample
        noisy_images = noise_scheduler.step(noise, timestep, noisy_images).prev_sample

    images = latents_to_pil(vae, noisy_images)
    return images[0]