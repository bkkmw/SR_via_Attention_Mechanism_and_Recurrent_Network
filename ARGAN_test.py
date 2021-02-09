import torch
import torchvision
import data_loader
import ARGAN
from torchsummary import summary
import def_func as ff
import os
import numpy as np
from PIL import Image


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

ttt = torchvision.utils.make_grid(images, nrow=dprow)
print(ttt.shape)
ff.imshow(torchvision.utils.make_grid(images, nrow=dprow))
ff.imshow(torchvision.utils.make_grid(mattes, nrow=dprow))
ff.imshow(torchvision.utils.make_grid(frees, nrow=dprow))

# Learning Parameters
steps = 3   # Number of progressive step
beta = 0.7  # weight for MSE loss of each step
lamb = 0.7  # weight for Semi-Superviesd learning
l_rate = 0.0002  # learning rate
dis_steps = 0 # training Discriminator
num_epoch = 5

gen_net = ARGAN.Gen(batch_size=batch_num, step_num=steps)
dis_net = ARGAN.Disc(batch_size=batch_num)
# Model Summary
#summary(gen_net, (3,128,128))
summary(dis_net, (3,128,128))

# Loss function
MSE = torch.nn.MSELoss()
#VGG = VGGPerceptualLoss()
ADV = torch.nn.BCELoss()
# Optimizer
gen_optim = torch.optim.SGD(gen_net.parameters(), lr=l_rate, momentum=0.5)
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
    print('===========%d epoch running==========' %(epoch+1))
    batch_err_a = 0.0
    batch_err_d = 0.0
    batch_err_r = 0.0

    for i, datas in enumerate(trainloader):
        total_loss = 0.0
        det_loss = 0.0
        rem_loss = 0.0
        adv_loss = 0.0

        images, mattes, frees = datas

        matt, free = gen_net(images)

        if i<dis_steps:
            dis_optim.zero_grad()
            real_est = dis_net(frees)
            real_label = torch.ones(frees.shape[0],1)
            real_err = ADV(real_est, real_label)
            real_err.backward()

            fake_est = dis_net(free[steps-1])
            fake_label = torch.zeros(free[steps-1].shape[0],1)
            fake_err = ADV(fake_est, fake_label)
            fake_err.backward()

            dis_err = real_err + fake_err
            dis_optim.step()

            batch_err_a += dis_err

            if i % 10 ==9:
                print("[%d batch]\ttotal loss : %f"
                      %(i+1, batch_err_a))
                batch_err = 0.0

        else:
            gen_optim.zero_grad()
            for n in range(steps):
                det_loss += pow(beta, steps-n) * MSE(mattes, matt[n])
                rem_loss += pow(beta, steps-n) * MSE(frees, free[n])
                #rem_loss += MSE(VGG(frees), VGG(free[n]))

            rem_loss.backward(retain_graph=True)
            print("rem_loss backward()")

            det_loss.backward(retain_graph=True)
            print("det_loss backward")

            adv_loss = ADV(dis_net(free[steps-1]),
                           torch.ones(free[steps-1].shape[0],1))
            adv_loss.backward()
            print("adv_loss backward")

            total_loss = det_loss + rem_loss + adv_loss
            gen_optim.step()
            print("params updated")

            batch_err_a += adv_loss
            batch_err_d += det_loss
            batch_err_r += rem_loss

            if i % 10 == 9:
                print("[%d batch]\tdet : %f, rem : %f, adv : %f"
                      %(i+1, batch_err_d, batch_err_r, batch_err_a))

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