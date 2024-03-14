import os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import CustomDataset

def data_to_latents(model, data_folder, latent_folder, transform, batch_size):
    data_set = CustomDataset(root_dir=data_folder, transform=transform)
    label = data_set.classes
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size)
    name = 0
    for X, y in data_loader:
        X_encode = model.encode(X.to(model.device)).latent_dist.sample() * 0.18215
        for X_encode_i, y_i in zip(X_encode, y):
            y_i_label = latent_folder / label[y_i.item()]
            y_i_label.mkdir(parents=True, exist_ok=True) 
            file_path = os.path.join(y_i_label, f"{name}.npy")
            name += 1
            np.save(file_path, X_encode_i.cpu().numpy()) 

def data_preprocessing(vae, data_folder, latent_folder):
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x * 2.0 - 1.0
    ])
    data_to_latents(vae, data_folder, latent_folder, transform=transform, batch_size=5)