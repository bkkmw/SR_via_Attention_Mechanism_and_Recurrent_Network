import torch
import torchvision
import data_loader
import ARGAN
from torchsummary import summary
import def_func as ff
import os
import numpy as np
from PIL import Image
import vgg_perceptual_loss as VGGLoss


# Directories
# BKD
img_path = 'D:/BKD/U/4_Project/ISTD_Dataset/train/'
test_path = 'D:/BKD/U/4_Project/ISTD_Dataset/test/'
# BKL
img_path = 'c:/users/BKL/Desktop/KU/4/ISTD_Dataset/train'
test_path = 'c:/users/BKL/Desktop/KU/4/ISTD_Dataset/test'

# Load images
batch_num = 4
dprow = 2

trainloader = data_loader.get_data('train', img_path, batch_num)
dataiter = iter(trainloader)
images, mattes, frees = dataiter.next()
# visualize
ff.imshow(torchvision.utils.make_grid(images, nrow=dprow))
ff.imshow(torchvision.utils.make_grid(mattes, nrow=dprow))
ff.imshow(torchvision.utils.make_grid(frees, nrow=dprow))

# Learning Parameters
steps = 3   # Number of progressive step
beta = 0.7  # weight for MSE loss of each step
lamb = 0.7  # weight for Semi-Superviesd learning
l_rate_d = 0.0002   # learning rate : Discriminator
l_rate_g = 0.0005   # learning rate : Generator
num_epoch = 5

gen_net = ARGAN.Gen(batch_size=batch_num, step_num=steps)
dis_net = ARGAN.Disc(batch_size=batch_num)

# Model Summary
#summary(gen_net, (3,128,128))
#summary(dis_net, (3,128,128))


# Loss function
MSE = torch.nn.MSELoss()
VGG = VGGLoss.VGGPerceptualLoss()
ADV = torch.nn.BCELoss()
# Optimizer
gen_optim = torch.optim.SGD(gen_net.parameters(), lr=l_rate, momentum=0.9)
dis_optim = torch.optim.Adam(dis_net.parameters(), lr=l_rate)

trained = 0
"""
# Load model parameters
if os.path.isfile('gen_net.pth'):
    gen_net.load_state_dict(torch.load('gen_net.pth'))
if os.path.isfile('dis_net.pth'):
    dis_net.load_state_dict(torch.load('dis_net.pth'))
    trained = 5
"""
gen_net.train()
dis_net.train()

for epoch in range(num_epoch):
    print('=================%d epoch running==============' %(epoch+1))

    for i, datas in enumerate(trainloader):
        total_loss = 0.0
        det_loss = 0.0
        det_err = 0.0
        rem_loss = 0.0
        rem_err = 0.0
        adv_loss = 0.0
        # load data
        image, matte, free = datas
        # Estimated Results
        mattes, frees = gen_net(image)

        # train Discriminator
        dis_optim.zero_grad()
        # real data
        real_err = ADV(dis_net(free), torch.ones(free.shape[0], 1))
        real_err.backward()
        # fake data
        fake_err = ADV(dis_net(frees[steps-1]),
                               torch.zeros(frees[steps-1].shape[0], 1))
        fake_err.backward()

        dis_err = real_err + fake_err
        dis_optim.step()

        # train Generator
        gen_optim.zero_grad()
        # for N steps
        for n in range(steps):
            # detector loss : MSE
            det_loss += pow(beta, steps-n) * MSE(matte, mattes[n])
            # removal loss : accuracy loss
            rem_loss += pow(beta, steps-n) * MSE(free, frees[n])
            # removal loss : perceptual loss
            rem_loss += MSE(VGG(frees), VGG(free[n]))

        # Adversarial loss
        adv_loss = ADV(dis_net(frees[steps-1]),
                       torch.ones(frees[steps-1].shape[0],1))

        total_loss = det_loss + rem_loss + adv_loss
        total_loss.backward()
        gen_optim.step()


    # 1 epoch Finished
    torch.save(gen_net.state_dict(), 'gen_net.pth')
    torch.save(dis_net.state_dict(), 'dis_net.pth')
    print("%d epoch params saved" %(epoch+1))

    SAVE_PATH = 'C:/Users/BKL/Desktop/KU/4/Out/'

    img_fname = SAVE_PATH + str(epoch+trained) + "_img.jpg"
    mat_fname = SAVE_PATH + str(epoch+trained) + "_matt.jpg"
    fre_fname = SAVE_PATH + str(epoch+trained) + "_free.jpg"

    ff.save_batch(images, dprow, img_fname)
    ff.save_batch(matt[steps-1], dprow, mat_fname)
    ff.save_batch(free[steps-1], dprow, fre_fname)