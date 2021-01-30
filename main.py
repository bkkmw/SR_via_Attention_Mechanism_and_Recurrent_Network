import torch
import torchvision
import ARGAN
import data_loader
import AGAN
from torchsummary import summary
import def_func as ff


# BKD
img_path = 'D:/BKD/U/4_Project/ISTD_Dataset/train/'
test_path = 'D:/BKD/U/4_Project/ISTD_Dataset/test/'
# BKL
img_path = 'c:/users/BKL/Desktop/KU/4/ISTD_Dataset/train'
test_path = 'c:/users/BKL/Desktop/KU/4/ISTD_Dataset/test'


# Load images
batch_num = 4
dprow = 2
"""
train_img = data_loader.ARGAN_Dataset(img_path, src_trans=data_loader.img2tensor,
                               matt_trans=data_loader.matt2tensor, is_test=False)
trainloader = torch.utils.data.DataLoader(train_img, batch_size=batch_num, shuffle=True)

test_img = data_loader.ARGAN_Dataset(test_path, src_trans=data_loader.img2tensor,
                               matt_trans=data_loader.matt2tensor, is_test=True)
testloader = torch.utils.data.DataLoader(test_img, batch_size=batch_num, shuffle=True)
"""

trainloader = data_loader.get_data('train', img_path, batch_num)

############# DATA Display ############
#print(train_img)

#for i, (src,matt) in enumerate(trainloader):
dataiter = iter(trainloader)
print(type(dataiter))
images, mattes, frees = dataiter.next()

print(images.shape)
print(mattes.shape)
print(frees.shape)

"""
ff.imshow(torchvision.utils.make_grid(images, nrow=dprow))
ff.imshow(torchvision.utils.make_grid(mattes, nrow=dprow))
ff.imshow(torchvision.utils.make_grid(frees, nrow=dprow))
"""

###################################
steps = 3

gen_net = AGAN.Gen(batch_size=batch_num, step_num=steps)
dis_net = AGAN.Disc()

MSE = torch.nn.MSELoss()
vgg = torchvision.models.vgg16(pretrained=True, progress=False)

gen_optim = torch.optim.SGD(gen_net.parameters(), lr=0.002, momentum=0.5)

gen_net.train()

num_params = sum(p.numel() for p in gen_net.parameters() if p.requires_grad)
print('Number of parameters : %d' %(num_params) )

#summary(gen_net, (3,256,256), batch_size=4)



for epoch in range(1):
    total_loss = 0.0
    det_loss = 0.0
    rem_loss = 0.0
    adv_loss = 0.0

    for i, datas in enumerate(trainloader, 0):
        images, mattes, frees = datas
        matt, free = gen_net(images)
        """
        print(images.shape)
        print(matt.shape)
        print(free.shape)
        imshow(torchvision.utils.make_grid(images, nrow=dprow))
        imshow(torchvision.utils.make_grid(matt[0], nrow=dprow))
        imshow(torchvision.utils.make_grid(free[0], nrow=dprow))
        """

        for n in range(steps):
            det_loss += MSE(mattes, matt[n])
            rem_loss += MSE(frees, free[n])
            # + pretrained VGG
            rem_loss += MSE(vgg(frees), vgg(free[n]))

        total_loss = det_loss + rem_loss
        print('%dth batch loss : %f' % (i, total_loss))
        with torch.autograd.set_detect_anomaly(True):

            total_loss.backward()
            gen_optim.step()
            print('Gradient updated')