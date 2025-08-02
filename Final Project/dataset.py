import os
from PIL import Image
from torch.utils.data import Dataset

class ImagetoImageDataset(Dataset):
    """
    Dataset for loading paired images from two domains (unpaired mapping).
    Each item is a tuple (imageA, imageB) transformed with the same transform.
    """
    def __init__(self, domainA_dir, domainB_dir, transform=None):
        self.domainA_paths = sorted([
            os.path.join(domainA_dir, f) for f in os.listdir(domainA_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.domainB_paths = sorted([
            os.path.join(domainB_dir, f) for f in os.listdir(domainB_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        # Ensure both domains have same length (CycleGAN assumption)
        self.length = min(len(self.domainA_paths), len(self.domainB_paths))
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Get image paths
        A_path = self.domainA_paths[index % len(self.domainA_paths)]
        B_path = self.domainB_paths[index % len(self.domainB_paths)]
        # Load images
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # Apply transforms if any
        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)
        return A_img, B_img
