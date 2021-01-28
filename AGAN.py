import torch
import torch.nn as nn
import torchvision
from Conv_LSTM import ConvLSTM as CLSTM


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
        self.conv = nn.Sequential(nn.Conv2d(inp_ch, out_ch, kernel_size=k, stride=s, padding=p),
                                  nn.BatchNorm2d(out_ch), nn.LeakyReLU())

    def forward(self, inp):
        x = self.conv(inp)
        return x


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
        self.dconv = nn.Sequential(nn.ConvTranspose2d(inp_ch, out_ch, kernel_size=k, stride=s, padding=p),
                                   nn.BatchNorm2d(out_ch), nn.LeakyReLU())

    def forward(self, inp):
        return self.dconv(inp)


class AttDet(nn.Module):
    def __init__(self):
        super(AttDet, self).__init__()
        self.block = nn.Sequential(
            ConvL(3, 3), ConvL(3, 4), ConvL(4, 8), ConvL(8, 16), ConvL(16, 16),
            ConvL(16, 16), ConvL(16,32), ConvL(32, 32), ConvL(32, 64), ConvL(64, 64)
        )

    def forward(self, inp):
        x = self.block(inp)
        return x


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
        self.dconv0 = DConvL(512, 512, 4, 2, 2)
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
        self.rem2 = nn.Sequential(nn.Conv2d(3,1, kernel_size=3, stride=1, padding=1),
                                  nn.Sigmoid())

    def forward(self, inp, att_map):
        x0 = self.conv0(inp)
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

        out = (xx * att_map) + inp

        return out


class Gen(nn.Module):
    def __init__(self, batch_size=None, step_num=None):
        super(Gen, self).__init__()
        self.batch_size = batch_size
        self.step = step_num
        self.attL = []
        self.attM = []
        self.remE = []
        # shadow attention detector
        for i in range(self.step):
            self.attL.append(AttDet())
            self.attM.append(nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
                                     nn.Sigmoid()))
        # Convolutional LSTM layer
        self.lstm = CLSTM(input_dim=64, hidden_dim=64, kernel_size=(3, 3),
                          num_layers=self.step, batch_first=True,
                          bias=True, return_all_layers=False)
        # shadow removal encoder
        for i in range(self.step):
            self.remE.append(REncoder())

    def forward(self, inp):
        print('input data size : ', inp.shape)

        att_map = torch.empty(self.step, self.batch_size, 1, 256, 256)
        out = torch.empty(self.step, self.batch_size, 3, 256, 256)

        for i in range(self.step):
            # attention detector
            if i == 0:
                x = inp
            else:
                x = out[i-1]
            print("%d th step input image : " %(i+1), x.shape)
            lstm_in = self.attL[i](x)
            print("extracted feature : ", lstm_in.shape)
            '''
            lstm_out, h = self.lstm(lstm_in)
            print("output of LSTM layer : ", lstm_out.shape)
            temp = self.attM[i](lstm_out)
            '''
            temp = self.attM[i](lstm_in)
            print("temp attemtion map : ", temp.shape)
            # removal encoder
            res = self.remE[i](x, temp)
            print("output image : ", res.shape)

            att_map[i] = temp
            out[i] = res
            """
            if i == 0:
                att_map = temp
                out = res
            else:
                att_map = torch.stack([att_map, temp])
                out = torch.stack([out, res])
            """

        att_map = torch.FloatTensor(att_map)
        out = torch.FloatTensor(out)
        print('Final output : ', att_map.shape, out.shape)

        return att_map, out


# Discriminator
class Disc(nn.Module):
    def __init__(self):
        super(Disc, self).__init__()

    def forward(self):
        None