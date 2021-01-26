import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os



# My dataset loading function
def make_dataset(root, test) -> list:
    dataset = []
    # sub folder names of data set
    if test is True:
        src_dir = 'test_A'
        matt_dir = 'test_B'
        free_dir = 'test_C'
    else:
        src_dir = 'train_A'
        matt_dir = 'train_B'
        free_dir = 'train_C'

    # file names of dataset
    src_fnames = sorted(os.listdir(os.path.join(root, src_dir)))
    matt_fnames = sorted(os.listdir(os.path.join(root, matt_dir)))
    free_fnames = sorted(os.listdir(os.path.join(root, free_dir)))

    # matching datasets by name
    # same fname for triplets
    for src_fname in src_fnames:
        # source image (image with shadow)
        src_path = os.path.join(root, src_dir, src_fname)
        if src_fname in matt_fnames:
            # shadow matte image
            matt_path = os.path.join(root, matt_dir, src_fname)
            if src_fname in free_fnames:
                # shadow free image
                free_path = os.path.join(root, free_dir, src_fname)
                # if triplets exists append to dataset
                temp = (src_path, matt_path, free_path)
                dataset.append(temp)
            # if one of triplets missing do NOT append to dataset
            else:
                print(free_fname, 'Shadow free file missing')
                continue
        else:
            print(matt_fname, 'Shadow matte file missing')
            continue

    return dataset


class ARGAN_Dataset(torchvision.datasets.vision.VisionDataset):
    # ARGAN dataset class composed of 3 func
    def __init__(self, root, loader=torchvision.datasets.folder.default_loader,
                 is_test=False, src_trans=None, matt_trans=None):
        super().__init__(root, transform=src_trans, target_transform=matt_trans)

        # Custom dataset loader for Training
        samples = make_dataset(self.root, test=is_test)
        self.loader = loader
        self.samples = samples
        # train data list

    #    self.src_samples = [s[0] for s in samples]
    #    self.matt_samples = [s[1] for s in samples]
    #    self.free_samples = [s[2] for s in samples]

    # Get single data
    def __getitem__(self, index):
        # load training data
        src_path, matt_path, free_path = self.samples[index]
        src_sample = self.loader(src_path)
        matt_sample = self.loader(matt_path)
        free_sample = self.loader(free_path)

        # transform data if required
        if self.transform is not None:
            # transform for RGB image : Shadow image and Shadow free image
            src_sample = self.transform(src_sample)
            free_sample = self.transform(free_sample)
        if self.target_transform is not None:
            # transform for Binary image : Shaode Matte
            matt_sample = self.target_transform(matt_sample)

        return src_sample, matt_sample, free_sample

    # Get dataset length
    def __len__(self):
        return len(self.samples)


# image Transforms
# image size 256x256 used for training _ from paper
img2tensor = transforms.Compose([
                                 transforms.Resize(size=(256,256)),
                                 transforms.ToTensor()
                                 # additional tasks
])
matt2tensor = transforms.Compose([
                                  transforms.Resize(size=(256,256)),
                                  transforms.Grayscale(1),
                                  transforms.ToTensor()
                                  # additional tasks

])


#NETWORK_ GEN
## Generator Net class

class Gen(nn.Module):
    def __init__(self, batch_size=None):
        super(Gen, self).__init__()
        self.batch_size = batch_size
        # shadow attention detector
        self.det0 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(64), nn.LeakyReLU())
        self.det1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(64), nn.LeakyReLU())
        self.det2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(64), nn.LeakyReLU())
        self.det3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(64), nn.LeakyReLU())
        self.det4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(64), nn.LeakyReLU())
        self.det5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(64), nn.LeakyReLU())
        self.det6 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(64), nn.LeakyReLU())
        self.det7 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(64), nn.LeakyReLU())
        self.det8 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(64), nn.LeakyReLU())
        self.det9 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(64), nn.LeakyReLU())
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=3)
        #self.lstm_c = nn.LSTMCell(input_size=64, hidden_size=64, )
        # hidden layers for LSTM
        self.hidden = self.init_h(batch_size)

        self.att_map = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
                                     nn.Sigmoid())

        #shadow removal encoder
        # CONV LAYERS : extract feature
        self.conv0 = nn.Sequential(nn.Conv2d(3,64, kernel_size=3, stride=2),
                                   nn.BatchNorm2d(64), nn.LeakyReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(64,128, kernel_size=3, stride=2),
                                   nn.BatchNorm2d(128), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(128,256, kernel_size=3, stride=2),
                                   nn.BatchNorm2d(256), nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(256,512, kernel_size=3, stride=2),
                                   nn.BatchNorm2d(512), nn.LeakyReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(512,512, kernel_size=3, stride=2),
                                   nn.BatchNorm2d(512), nn.LeakyReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(512,512, kernel_size=3, stride=2),
                                   nn.BatchNorm2d(512), nn.LeakyReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(512,512, kernel_size=3, stride=2),
                                   nn.BatchNorm2d(512), nn.LeakyReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(512,512, kernel_size=3, stride=2),
                                   nn.BatchNorm2d(512), nn.LeakyReLU())

        # DECONV LAYERS : generate image with feature data
        self.dconv0 = nn.Sequential(nn.ConvTranspose2d(512,512, kernel_size=4, stride=2),
                                    nn.BatchNorm2d(512), nn.LeakyReLU())
        self.dconv1 = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2),
                                    nn.BatchNorm2d(512), nn.LeakyReLU())
        self.dconv2 = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2),
                                    nn.BatchNorm2d(512), nn.LeakyReLU())
        self.dconv3 = nn.Sequential(nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2),
                                    nn.BatchNorm2d(512), nn.LeakyReLU())
        self.dconv4 = nn.Sequential(nn.ConvTranspose2d(512,256, kernel_size=4, stride=2),
                                    nn.BatchNorm2d(256), nn.LeakyReLU())
        self.dconv5 = nn.Sequential(nn.ConvTranspose2d(256,128, kernel_size=4, stride=2),
                                    nn.BatchNorm2d(128), nn.LeakyReLU())
        self.dconv6 = nn.Sequential(nn.ConvTranspose2d(128,64, kernel_size=4, stride=2),
                                    nn.BatchNorm2d(64), nn.LeakyReLU())
        self.dconv7 = nn.Sequential(nn.ConvTranspose2d(64,3, kernel_size=4, stride=2),
                                    nn.BatchNorm2d(3), nn.LeakyReLU())

        # Convert to Neg residual
        self.rem0 = nn.Sequential(nn.Conv2d(3,3, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(3), nn.LeakyReLU())
        self.rem1 = nn.Sequential(nn.Conv2d(3,3, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(3), nn.LeakyReLU())
        self.rem2 = nn.Sequential(nn.Conv2d(3,1, kernel_size=3, stride=1, padding=1),
                                  nn.Sigmoid())


    # init hidden layers of LSTM : all zeros
    def init_h(self, batch_size):
        return (torch.zeros(batch_size, 64, 256, 256))


    def forward(self, inp, h):
        # Attention detector
        x = self.det0(inp)
        x = self.det1(x)
        x = self.det2(x)
        x = self.det3(x)
        x = self.det4(x)
        x = self.det5(x)
        x = self.det6(x)
        x = self.det7(x)
        x = self.det8(x)
        x = self.det9(x)
        # LSTM

        print(h.shape)
        print('feature size : ', x.shape)
        #for i in range(self.batch_size):
        #    x[i], self.hidden[i] = self.lstm(x[i], self.hidden[i])
        #x, self.hidden = self.lstm(x, h)

        # Attention map
        matt = self.att_map(x)

        # Removal Encoder
        x0 = self.conv0(x)
        x1 = self.conv1(xx)
        x2 = self.conv2(xx)
        x3 = self.conv3(xx)
        x4 = self.conv4(xx)
        x5 = self.conv5(xx)
        x6 = self.conv6(xx)
        x7 = self.conv7(xx)

        xx = self.dconv0(x7)
        xx = self.dconv1(xx + x6)
        xx = self.dconv2(xx + x5)
        xx = self.dconv3(xx + x4)
        xx = self.dconv4(xx + x3)
        xx = self.dconv5(xx + x2)
        xx = self.dconv6(xx + x1)
        xx = self.dconv7(xx + x0)

        xx = self.rem0(xx)
        xx = self.rem1(xx)
        xx = self.rem2(xx)

        out = xx*matt + inp

        return matt, out, h
