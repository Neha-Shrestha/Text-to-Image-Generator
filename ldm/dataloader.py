from dataset import NumpyDS
from torch.utils.data import DataLoader

def latent_dataloader(latent_folder, batch_size, shuffle):
    data_set = NumpyDS(root_dir=latent_folder)
    return DataLoader(dataset=data_set, batch_size=batch_size, shuffle=shuffle)