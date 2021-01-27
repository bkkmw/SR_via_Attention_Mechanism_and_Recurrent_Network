import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from Conv_LSTM import ConvLSTM as CLSTM


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
        #self.lstm = nn.LSTM(input_size=256*256, hidden_size=256)
        self.lstm = CLSTM(input_dim=64, hidden_dim=[64,64,128], kernel_size=(3,3),
                          num_layers=3, batch_first=True, bias=True, return_all_layers=False)
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
        return (torch.zeros(batch_size, 1, 64, 256),
                torch.zeros(batch_size, 1, 64, 256))

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

        #print(h.shape)
        #print('feature size : ', x.view([self.batch_size,64,-1]).shape)
#        x, self.hidden = self.lstm(x.view([self.batch_size,64,-1]), h)
        x, self.hidden = self.lstm(x)
        # Attention map
        matt = self.att_map(x)

        # Removal Encoder
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)

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
