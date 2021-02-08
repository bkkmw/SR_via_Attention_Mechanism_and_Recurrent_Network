"""
Class for AGAN : LSTM layer is removed from ARGAN
using ISTD dataset images(1330 triplets with 640 * 480)
image size 128*128 used
"""

import torch
import torch.nn as nn
import torchvision

# Generator Net class
class ConvL(nn.Module):
    def __init__(self, inp_ch, out_ch, k=None, s=None, p=None):
        super(ConvL, self).__init__()
        # (3, 1, 1) keep spatial resolution
        if k is None:
            k = 3
            s = 1
            p = 1
        # default params
        elif s is None:
            s = 1
        elif p is None:
            p = 0
        self.conv = nn.Sequential(nn.Conv2d(inp_ch, out_ch,
                                            kernel_size=k, stride=s, padding=p),
                                  nn.BatchNorm2d(out_ch),
                                  nn.LeakyReLU())

    def forward(self, x):
        return self.conv(x)


class DConvL(nn.Module):
    def __init__(self, inp_ch, out_ch, k=None, s=None, p=None):
        super(DConvL, self).__init__()
        # (3, 1, 1) keep spatial resolution
        if k is None:
            k = 3
            s = 1
            p = 1
        # default params
        elif s is None:
            s = 1
        elif p is None:
            p = 0
        self.dconv = nn.Sequential(nn.ConvTranspose2d(inp_ch, out_ch,
                                                      kernel_size=k, stride=s, padding=p),
                                   nn.BatchNorm2d(out_ch),
                                   nn.LeakyReLU())

    def forward(self, x):
        return self.dconv(x)


class AttDet(nn.Module):
    def __init__(self):
        super(AttDet, self).__init__()
        self.block = nn.Sequential(
            ConvL(3, 8), ConvL(8, 8), ConvL(8, 16), ConvL(16, 16),
            ConvL(16, 16),ConvL(16, 32), ConvL(32,32),
            ConvL(32, 64), ConvL(64, 64), ConvL(64, 64)
        )

    def forward(self, inp):
        return self.block(inp)


class REncoder(nn.Module):
    def __init__(self):
        super(REncoder, self).__init__()
        # CONV LAYERS : extract feature
        self.conv0 = ConvL(3, 64, 3, 2, 3)
        self.conv1 = ConvL(64, 128, 3, 2, 2)
        self.conv2 = ConvL(128, 256, 3, 2, 2)
        self.conv3 = ConvL(256, 512, 3, 2, 2)
        self.conv4 = ConvL(512, 512, 3, 2, 2)
        self.conv5 = ConvL(512, 512, 3, 2, 2)
        self.conv6 = ConvL(512, 512, 3, 2, 2)
        self.conv7 = ConvL(512, 512, 3, 2, 2)

        # DECONV LAYERS : generate image with feature data
        self.dconv0 = DConvL(512, 512, 3, 1, 1)
        self.dconv1 = DConvL(512, 512, 4, 2, 2)
        self.dconv2 = DConvL(512, 512, 4, 2, 2)
        self.dconv3 = DConvL(512, 512, 4, 2, 2)
        self.dconv4 = DConvL(512, 256, 4, 2, 2)
        self.dconv5 = DConvL(256, 128, 4, 2, 2)
        self.dconv6 = DConvL(128, 64, 4, 2, 2)
        self.dconv7 = DConvL(64, 3, 4, 2, 3)

        # Convert to Neg residual
        self.rem0 = ConvL(3, 3)
        self.rem1 = ConvL(3, 3)
        self.rem2 = nn.Sequential(nn.Conv2d(3,1, kernel_size=3,
                                            stride=1, padding=1),
                                  nn.Sigmoid())

    def forward(self, x, att_map):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)

        xx = self.dconv0(x7)
        xx += x6
        xx = self.dconv1(xx)
        xx += x5
        xx = self.dconv2(xx)
        xx += x4
        xx = self.dconv3(xx)
        xx += x3
        xx = self.dconv4(xx)
        xx += x2
        xx = self.dconv5(xx)
        xx += x1
        xx = self.dconv6(xx)
        xx += x0
        xx = self.dconv7(xx)

        xx = self.rem0(xx)
        xx = self.rem1(xx)
        xx = self.rem2(xx)

        res = torch.matmul(xx, att_map)
        out = res + x
        return out


# Generative Network
class Gen(nn.Module):
    def __init__(self, batch_size=None, step_num=None):
        super(Gen, self).__init__()
        self.batch_size = batch_size
        self.step = step_num
        # Attention Detector
        self.attL = AttDet()
        # Convolutional LSTM cell

        # Attention Map
        self.attM = nn.Sequential(
                  nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
                  nn.Sigmoid()
        )
        # Removal Encoder
        self.remE = REncoder()

    def forward(self, x):
      in_batch = x.shape[0]
      if in_batch != self.batch_size:
        self.batch_size = in_batch

      with torch.autograd.set_detect_anomaly(True):
        # attention map & output tensor
        att_map = torch.empty(self.step, self.batch_size, 1, 128, 128)
        out = torch.empty(self.step , self.batch_size, 3, 128, 128)

        # for N progressive steps
        for i in range(self.step):
            # attention detector
            lstm_in = self.attL(x)
            # LSTM Layer

            # Generate attention map
            temp = self.attM(lstm_in)
            # removal encoder
            res = self.remE(x, temp)
            # append to output
            att_map[i] = temp
            out[i] = res
            x = res

        # output to tensor
        att_map = torch.FloatTensor(att_map)
        out = torch.FloatTensor(out)

        return att_map, out


# Discriminator
class Disc(nn.Module):
    def __init__(self, batch_size=None):
        super(Disc, self).__init__()
        self.batch_size = batch_size
        self.conv0 = ConvL(3, 64, 4, 2, 1)
        self.conv1 = ConvL(64, 128, 4, 2, 1)
        self.conv2 = ConvL(128, 256, 4, 2, 1)
        self.conv3 = ConvL(256, 512, 4, 2, 1)
        self.conv4 = ConvL(512, 256, 4, 2, 1)
        self.fc = nn.Sequential(nn.Linear(256 * 16, 512),
                                nn.Linear(512, 1),
                                nn.Sigmoid())

    def forward(self, inp):
        with torch.autograd.set_detect_anomaly(True):
            self.batch_size = inp.shape[0]
            x = self.conv0(inp)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = torch.flatten(x, 1)
            out = self.fc(x)

            return out