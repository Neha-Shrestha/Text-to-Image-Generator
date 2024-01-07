import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.file_list = []

        for cls in self.classes:
            class_path = os.path.join(self.root_dir, cls)
            files = os.listdir(class_path)
            self.file_list.extend([(cls, os.path.join(class_path, file)) for file in files])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        class_name, img_path = self.file_list[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image) / 255.0

        label = self.class_to_idx[class_name]

        if self.transform:
            image = self.transform(image)

        return image, label

class NumpyDS(CustomDataset):
    def __getitem__(self, idx):
        class_name, img_path = self.file_list[idx]
        
        image = np.load(img_path)
        label = self.class_to_idx[class_name]

        if self.transform:
            image = self.transform(image)
        
        return image, label
