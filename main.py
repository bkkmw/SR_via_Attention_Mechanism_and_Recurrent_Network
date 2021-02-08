import torch
import torchvision
import data_loader
import AGAN
from torchsummary import summary
import def_func as ff
import os

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


ff.imshow(torchvision.utils.make_grid(images, nrow=dprow))
ff.imshow(torchvision.utils.make_grid(mattes, nrow=dprow))
ff.imshow(torchvision.utils.make_grid(frees, nrow=dprow))

# Learning Parameters
steps = 3   # Number of progressive step
beta = 0.7  # weight for MSE loss of each step
lamb = 0.7  # weight for Semi-Superviesd learning
l_rate = 0.002  # learning rate
dis_steps = 120 # training Discriminator
num_epoch = 5

gen_net = AGAN.Gen(batch_size=batch_num, step_num=steps)
dis_net = AGAN.Disc(batch_size=batch_num)
# Model Summary
summary(gen_net, (3,128,128))
summary(dis_net, (3,128,128))

# Loss function
MSE = torch.nn.MSELoss()
#VGG = VGGPerceptualLoss()
ADV = torch.nn.BCELoss()
# Optimizer
gen_optim = torch.optim.SGD(gen_net.parameters(), lr=l_rate, momentum=0.5)
dis_optim = torch.optim.Adam(dis_net.parameters(), lr=l_rate)

# Load model parameters
if os.path.isfile('gen_net.pth'):
    gen_net.load_state_dict(torch.load('gen_net.pth'))
if os.path.isfile('dis_net.pth'):
    dis_net.load_state_dict(torch.load('dis_net.pth'))

gen_net.train()
dis_net.train()

for epoch in range(num_epoch):
    print('===========%d epoch running==========' %(epoch))
    batch_err = 0.0

    for i, datas in enumerate(trainloader):
        total_loss = 0.0
        det_loss = 0.0
        rem_loss = 0.0
        adv_loss = 0.0

        images, mattes, frees = datas

        matt, free = gen_net(images)

        if i<dis_steps:
            real_err = ADV(dis_net(frees), torch.ones(frees.shape[0],1))
            real_err.backward()
            fake_err = ADV(dis_net(free[steps-1]),
                                   torch.zeros(free[steps-1].shape[0],1))
            fake_err.backward()

            dis_err = real_err + fake_err
            dis_optim.step()
            dis_optim.zero_grad()

            batch_err += dis_err

        else:
            for n in range(steps):
                det_loss += pow(beta, steps-n) * MSE(mattes, matt[n])
                rem_loss += pow(beta, steps-n) * MSE(frees, free[n])
                #rem_loss += MSE(VGG(frees), VGG(free[n]))
            adv_loss = ADV(dis_net(free[steps-1]),
                           torch.ones(free[steps-1].shape[0],1))

            total_loss = det_loss + rem_loss + adv_loss
            total_loss.backward()
            gen_optim.step()
            gen_optim.zero_grad()

            batch_err += total_loss

        torch.save(gen_net.state_dict(), 'gen_net.pth')
        torch.save(dis_net.state_dict(), 'dis_net.pth')