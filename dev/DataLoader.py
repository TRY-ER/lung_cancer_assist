from DataSetter import LungImageData
from torch.utils.data import DataLoader 
import json

with open("../dataset/dataset.json", "r") as file:
    data = json.load(file)


train_split = data["training"]
test_split = data["test"] 

dataset = LungImageData(train_split,transform=None)
batch_size = 1

dataLoader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        )

