import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import ARGAN
import data_loader
import AGAN
from torchsummary import summary

# BKD
img_path = 'D:/BKD/U/4_Project/ISTD_Dataset/train/'
test_path = 'D:/BKD/U/4_Project/ISTD_Dataset/test/'
# BKL

# display image
def imshow(image):
#  image = image/2 + 0.5
  # numpy
  npimage = image.detach().numpy()
  plt.imshow(np.transpose(npimage, (1,2,0)))
  plt.show()


# Load images
batch_num = 4
dprow = 2

train_img = data_loader.ARGAN_Dataset(img_path, src_trans=data_loader.img2tensor,
                               matt_trans=data_loader.matt2tensor, is_test=False)
trainloader = torch.utils.data.DataLoader(train_img, batch_size=batch_num, shuffle=True)

test_img = data_loader.ARGAN_Dataset(test_path, src_trans=data_loader.img2tensor,
                               matt_trans=data_loader.matt2tensor, is_test=True)
testloader = torch.utils.data.DataLoader(test_img, batch_size=batch_num, shuffle=True)


############# DATA Display ############
print(train_img)

#for i, (src,matt) in enumerate(trainloader):
dataiter = iter(trainloader)
print(type(dataiter))
images, mattes, frees = dataiter.next()

print(images.shape)
print(mattes.shape)
print(frees.shape)

#imshow(torchvision.utils.make_grid(images, nrow=dprow))
#imshow(torchvision.utils.make_grid(mattes, nrow=dprow))
#imshow(torchvision.utils.make_grid(frees, nrow=dprow))


###################################
gen_net = AGAN.Gen(batch_size=batch_num, step_num=3)

gen_net.train()

num_params = sum(p.numel() for p in gen_net.parameters() if p.requires_grad)
print('Number of parameters : %d' %(num_params) )

#summary(gen_net, (3,256,256), batch_size=4)



for epoch in range(1):
    running_loss = 0.0

    for i, datas in enumerate(trainloader, 0):
        images, mattes, frees = datas
        matt, free = gen_net(images)
        print(images.shape)
        print(matt.shape)
        print(free.shape)
        imshow(torchvision.utils.make_grid(images, nrow=dprow))
        imshow(torchvision.utils.make_grid(matt[0], nrow=dprow))
        imshow(torchvision.utils.make_grid(free[0], nrow=dprow))
        
