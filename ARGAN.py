"""
Class for ARGAN : with LSTM layer
using ISTD dataset images(1330 triplets with 640 * 480)
image size 128*128 used
"""

import torch
import torch.nn as nn
import torchvision

# Generator Network


# Conv + BN + LReLU
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


# ConvT + BN + LReLU
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


# Attention Detector : 10 (Conv + BN + LReLU) layers
# Extract Feature
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


# Removal Encoder : 8 Conv + 8 DConv + 3Conv layers
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
        # Conv
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        # DConv with Skip Connection
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
        # Conv
        xx = self.rem0(xx)
        xx = self.rem1(xx)
        xx = self.rem2(xx)
        # Residual : product with attention map
        res = torch.matmul(xx, att_map)
        out = res + x
        return out


# LSTM layer
class ConvLSTM(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim=None):
        super(ConvLSTM, self).__init__()
        self.in_dim = inp_dim + hid_dim
        self.hidden_dim = hid_dim
        # Same output Channel as input Channel
        if out_dim is None:
            self.out_dim = inp_dim
        else:
            self.out_dim = out_dim
        self.conv_i = nn.Conv2d(self.in_dim, self.out_dim, 3, 1, 1)
        self.conv_f = nn.Conv2d(self.in_dim, self.out_dim, 3, 1, 1)
        self.conv_c = nn.Conv2d(self.in_dim, self.out_dim, 3, 1, 1)
        self.conv_o = nn.Conv2d(self.in_dim, self.out_dim, 3, 1, 1)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, c_prev, h_prev):
        # input X and hidden state H_prev
        xh = torch.cat((x, h_prev), 1)
        i = self.sig(self.conv_i(xh))
        f = self.sig(self.conv_f(xh))
        c = (f * c_prev) + (i * self.tanh(self.conv_c(xh)))
        o = self.sig(self.conv_o(xh))
        h = o * self.tanh(c)
        # C_next : cell output, H_next : hidden state
        return c, h

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return torch.zeros(batch_size, self.hidden_dim, height, width)


# Generative Network
class Gen(nn.Module):
    def __init__(self, batch_size=None, step_num=None):
        super(Gen, self).__init__()
        self.batch_size = batch_size
        self.step = step_num
        # Attention Detector
        self.attL = AttDet()
        # Convolutional LSTM cell
        self.lstm = ConvLSTM(inp_dim=64, hid_dim=64)
        # init hidden state
        self.hidden = self.lstm.init_hidden(self.batch_size, (128, 128))
        # Attention Map
        self.attM = nn.Sequential(
                  nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
                  nn.Sigmoid()
        )
        # Removal Encoder
        self.remE = REncoder()

    def init_h(self):
        self.hidden = self.lstm.init_hidden(self.batch_size, (128,128))

    def forward(self, x):
        in_batch = x.shape[0]
        if in_batch != self.batch_size:
            self.batch_size = in_batch
        # init hidden state for each forward
        self.init_h();
        self.hidden = self.lstm.init_hidden(self.batch_size, (128,128))


        # attention map & output tensor
        att_map = torch.empty(self.step, self.batch_size, 1, 128, 128)
        out = torch.empty(self.step, self.batch_size, 3, 128, 128)
        lstm_out = torch.zeros(self.batch_size, 64, 128, 128)
        # for N progressive steps
        for i in range(self.step):
            # attention detector
            lstm_in = self.attL(x)
            # LSTM Layer
            lstm_out, self.hidden = self.lstm(lstm_in, lstm_out, self.hidden)
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
        self.fc = nn.Sequential(nn.Linear(256 * 16, 1),
                                nn.Sigmoid())

    def forward(self, inp):

        self.batch_size = inp.shape[0]
        x = self.conv0(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)

        return out

