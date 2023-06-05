from DataLoader import train_data_loader
import matplotlib.pyplot as plt

count = 1 

for batch in train_data_loader:
    img, lbl = batch
    print(f"loading for count : {count}")
    count += 1
    # print(img.shape)
    # print(lbl.shape)
    # testing for memory overflow 
    # no memory overflow detected
    for i in range(img.shape[-1]):
        train_img = img.unsqueeze(1)
        print("train",train_img.shape)
        img = img[:,:,:,i].view([512,512])
        lbl = lbl[:,:,:,i].view([512,512])
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        plt.show()
        plt.figure(figsize=(10,10))
        plt.imshow(lbl) 
        plt.show()
        break
    break


