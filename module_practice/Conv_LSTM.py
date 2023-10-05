
import torch.nn as nn
import torch

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