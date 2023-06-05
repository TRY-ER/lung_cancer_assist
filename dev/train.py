import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.unet_main import CustUnet
from DataLoader import train_data_loader,val_data_loader
from tqdm import tqdm





# Hyperparameters 
LR = 3e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device",DEVICE)
BATCH_SIZE = 1
NUM_EPOCHS = 1 
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = True



def train_fn(train_data_loader,model,optimizer,loss_fn,scaler):
    train_data_loader = tqdm(train_data_loader,desc='Training')
    for i,(t_image,l_image) in enumerate(train_data_loader):
        t_image = t_image.to(DEVICE)
        l_image = l_image.to(DEVICE)
        # training setup
        for i in range(t_image.shape[-1]):
            train_img = t_image.unsqueeze(1)

            # converting 512 x 512 to 1,1,512,512
            img = t_image[:,:,:,i].view([1,1,512,512])
            lbl = l_image[:,:,:,i].view([1,1,512,512])

            # forward propagation
            with torch.cuda.amp.autocast(): 
                pred = model(img)
                # converting lbl image to shape of the unet-output
                re_lbl = F.interpolate(lbl,pred.shape[2:],mode='bilinear',align_corners=False)
                loss = loss_fn(pred,re_lbl)

            # back propagation
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # refresh tqdm
        train_data_loader.set_postfix(loss=loss.item()) 


def main():
    # model 
    model = CustUnet(1,1).to(DEVICE)
    # loss function
    loss_fn = nn.BCEWithLogitsLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)
    # scaler
    scaler = torch.cuda.amp.GradScaler()
    # train loop
    for epoch in range(NUM_EPOCHS):
        train_fn(train_data_loader,model,optimizer,loss_fn,scaler)
        # # validation loop
        # val_fn(val_data_loader,model,loss_fn)
        # break
    # save model
    # torch.save(model.state_dict(),'model.pth')
    # break 


if __name__ == '__main__':
    main()  
