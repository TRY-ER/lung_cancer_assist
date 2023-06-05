from DataSetter import LungImageData
from torch.utils.data import DataLoader 
from torchvision.transforms import transforms
import json
import random

random.seed(69)

MEAN = 0.2125 
STD = 0.2347


with open("../dataset/dataset.json", "r") as file:
    data = json.load(file)

val_idx = random.choice(range(len(data["training"])))

val_split = [data["training"][val_idx]]
data["training"].pop(val_idx)
train_split = data["training"]

transform = transforms.Compose([
    transforms.Normalize(MEAN, STD)
]) 

train_dataset = LungImageData(train_split,transform=transform)
val_dataset = LungImageData(val_split,transform=transform)
batch_size = 1

train_data_loader = DataLoader(train_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        )


val_data_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        )


