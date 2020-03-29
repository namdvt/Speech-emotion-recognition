import torch
import torch.nn as nn
import librosa
import librosa.display
import numpy as np
import torch.nn.functional as F
import scipy
import matplotlib.pyplot as plt


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, isrelu=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)
        #         nn.init.kaiming_normal_(self.conv.weight, mode='fan_in')
        self.bn = nn.BatchNorm2d(out_channels)
        self.isrelu = isrelu
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.isrelu:
            x = self.relu(x)
        return self.dropout(x)


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, isrelu=True, dilation=1):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias,
                              dilation=dilation)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in')
        self.bn = nn.BatchNorm1d(out_channels)
        self.isrelu = isrelu
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.isrelu:
            x = self.relu(x)
        return self.dropout(x)


class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        # nn.init.kaiming_normal_(self.linear.weight, mode='fan_in')
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.linear(x)
        if x.shape[0] != 1:
            x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Unsqueeze(nn.Module):
    def forward(self, x):
        return x.unsqueeze(1)


def get_filters():
    params = nn.Parameter(torch.rand(1025))
    f = torch.zeros(128, 1025)
    for i in range(128):
        f[i][i*8:(i+1)*8] = params[i:i+8]
    return f


class Model_CNN(nn.Module):
    def __init__(self):
        super(Model_CNN, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(1, 16, kernel_size=(3, 3)),
            nn.MaxPool2d(2),
            Conv2d(16, 32, kernel_size=(3, 3)),
            nn.MaxPool2d(2),
            Conv2d(32, 64, kernel_size=(3, 3)),
            nn.MaxPool2d(2),
        )
        self.rnn = nn.GRU(input_size=24, hidden_size=64, num_layers=1, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(64, 8),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        _, hidden = self.rnn(x.squeeze(), torch.zeros(1, x.shape[0], 64))
        x = self.fc(hidden.squeeze())
        return x


def dct(x):
    mfcc = []
    batch_size = x.shape[0]
    for i in range(batch_size):
        dct = scipy.fftpack.dct(x[i][0].detach().numpy(), axis=0, type=2, norm='ortho')[:40]
        mfcc.append(dct)
    mfcc = np.concatenate(mfcc).reshape(batch_size, 1, 40, 32)
    return torch.Tensor(mfcc)


class Model_CNN_1D(nn.Module):
    def __init__(self):
        super(Model_CNN_1D, self).__init__()
        self.conv = nn.Sequential(
            Conv1d(1, 16, kernel_size=5),
            Conv1d(16, 32, kernel_size=5),
            nn.MaxPool1d(2),
            Conv1d(32, 64, kernel_size=5),
            Conv1d(64, 128, kernel_size=5),
            nn.MaxPool1d(2),
            Conv1d(128, 256, kernel_size=5),
            Conv1d(256, 512, kernel_size=5),
            nn.AdaptiveAvgPool1d(1),
            Flatten(),
            nn.Linear(512, 8),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
