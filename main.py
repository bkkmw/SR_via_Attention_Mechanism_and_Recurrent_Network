import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import ARGAN

img_path = 'c:/users/BKL/Desktop/KU/4/ISTD_Dataset/train'
test_path = 'c:/users/BKL/Desktop/KU/4/ISTD_Dataset/test'


# display image
def imshow(image):
#  image = image/2 + 0.5
  # numpy
  npimage = image.numpy()
  plt.imshow(np.transpose(npimage, (1,2,0)))
  plt.show()


# Load images
batch_num = 4
dprow = 2

train_img = ARGAN.ARGAN_Dataset(img_path, src_trans=ARGAN.img2tensor,
                               matt_trans=ARGAN.matt2tensor, is_test=False)
trainloader = torch.utils.data.DataLoader(train_img, batch_size=batch_num, shuffle=True)

test_img = ARGAN.ARGAN_Dataset(test_path, src_trans=ARGAN.img2tensor,
                               matt_trans=ARGAN.matt2tensor, is_test=True)
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
gen_net = ARGAN.Gen(batch_size=batch_num)

gen_net.train()




for epoch in range(2):
    running_loss = 0.0

    for i, datas in enumerate(trainloader, 0):
        images, mattes, frees = datas

        matt, free, h = gen_net(images, gen_net.hidden)
        imshow(images)
        imshow(matt)
        imshow(free)
