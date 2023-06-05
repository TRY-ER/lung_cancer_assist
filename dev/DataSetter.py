import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
  

class LungImageData(Dataset):
    def __init__(self,train_split,transform=None):
        self.train_split = train_split
        self.transform = transform

    def __len__(self):
        return len(self.train_split)

    def __getitem__(self, idx):
        img_name = "../dataset"+self.train_split[idx]["image"][1:]
        lbl_name = "../dataset"+self.train_split[idx]["label"][1:]
        img = nib.load(img_name)
        img = img.get_fdata()
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        lbl = nib.load(lbl_name)
        lbl = lbl.get_fdata()
        lbl = lbl.astype(np.float32)
        lbl = torch.from_numpy(lbl)

        # color channel specific treatment
        img = np.clip(img, -1024, 1024)
        img = (img + 1024) / 2048

        # finding mean and std
        if self.transform:
            img = self.transform(img)
            lbl = self.transform(lbl)

        return img, lbl