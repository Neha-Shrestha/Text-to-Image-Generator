from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel

def vae(): 
    return AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").requires_grad_(False)

def ddpm_schedular(): 
    return DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02)

def unet():
    return UNet2DModel(
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
    )