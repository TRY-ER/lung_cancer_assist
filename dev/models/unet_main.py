import torch
import torch.nn as nn
import numpy as np

def double_conv(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv
   

def crop_img(x,y):
    tensor_size = x.size()[2]
    target_size = y.size()[2]
    delta_pre = tensor_size - target_size
    if(delta_pre % 2):
        delta = delta_pre // 2
        return x[:,:,delta:tensor_size-delta-1,delta:tensor_size-delta-1]
    else:
        delta = delta_pre // 2
        return x[:,:,delta:tensor_size-delta,delta:tensor_size-delta]


class CustUnet(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(CustUnet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # downsampling // encoding
        self.double_down_conv_1 = double_conv(n_channels, 64)
        self.double_down_conv_2 = double_conv(64, 128)
        self.double_down_conv_3 = double_conv(128, 256)
        self.double_down_conv_4 = double_conv(256, 512)
        self.double_down_conv_5 = double_conv(512, 1024)

        # upsampling // decoding 
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.double_up_conv_1 = double_conv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.double_up_conv_2 = double_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.double_up_conv_3 = double_conv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.double_up_conv_4 = double_conv(128, 64)

        # output layer
        self.reout = nn.Conv2d(64, n_classes, kernel_size=1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        
        # encoding forward
        
        x1 = self.double_down_conv_1(x)
        x2 = self.max_pool_2x2(x1)

        x3 = self.double_down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)

        x5 = self.double_down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)

        x7 = self.double_down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)

        x9 = self.double_down_conv_5(x8)

        # decoding forward
        x10 = self.up1(x9) 
        x11 = crop_img(x7, x10)
        x12 = torch.cat([x11, x10], dim=1) 
        x13 = self.double_up_conv_1(x12)

        x14 = self.up2(x13)
        x15 = crop_img(x5, x14)
        x16 = torch.cat([x15, x14], dim=1)
        x17 = self.double_up_conv_2(x16)

        x18 = self.up3(x17)
        x19 = crop_img(x3, x18)
        x20 = torch.cat([x19, x18], dim=1)
        x21 = self.double_up_conv_3(x20)

        x22 = self.up4(x21)
        x23 = crop_img(x1, x22)
        x24 = torch.cat([x23, x22], dim=1)
        x25 = self.double_up_conv_4(x24)

        # output layer
        xreout = self.reout(x25)
        xout = self.out(xreout)  ## using xout for binary classification mask

        return xout



# # testing code 
# if __name__ == "__main__":
#     model = CustUnet(1, 1)
#     print(model)
#     x = torch.randn(1, 1, 512, 512)
#     y = model(x)
#     print(y.shape)        