import torch
from torch import nn
import torch.nn.functional as F


class TDConvModule(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, step=(1, 1)):
        super(TDConvModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=step, padding=(0, kernel[1] // 2)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_chan)
        )

    def forward(self, x):
        return self.conv(x)


class TemporalGateConv(nn.Module):
    def __init__(self):
        super(TemporalGateConv, self).__init__()
        self.conv_1 = nn.Conv2d(16, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7))
        self.conv_2 = nn.Conv2d(16, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7))
        self.conv_3 = nn.Conv2d(16, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7))
        self.avgpool = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)),
            nn.Dropout(p=0.25)
        )

    def forward(self, x):
        P = self.conv_1(x)
        Q = torch.sigmoid(self.conv_2(x))
        x = F.relu(P * Q + self.conv_3(x))
        # x = self.conv_1(x)
        x = self.avgpool(x)
        return x


class CLNet(nn.Module):
    def __init__(self, args, kernel_sizes=((1, 16), (1, 32), (1, 64), (1, 128))):
        super(CLNet, self).__init__()
        self.TDConvModules = nn.ModuleList(
            [TDConvModule(1, 4, kernel=kernel_size, step=(1, 1)) for kernel_size in kernel_sizes]
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )
        self.SpatialDepthwiseConv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(args.eeg_channel, 1), stride=(1, 1), groups=16),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=0.25)
        )
        self.TemporalGateConv = TemporalGateConv()
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(352, args.num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        outputs = [tdconv_block(x) for tdconv_block in self.TDConvModules]
        x = torch.cat(outputs, dim=1)
        x = x * self.se(x)
        x = self.SpatialDepthwiseConv(x)
        x = self.TemporalGateConv(x)
        x = self.classify(x)
        return x


class CLNet8(CLNet):
    def __init__(self, args):
        kernel_sizes = ((1, 8), (1, 16), (1, 32), (1, 64))
        super(CLNet8, self).__init__(args, kernel_sizes)


class CLNet16(CLNet):
    def __init__(self, args):
        kernel_sizes = ((1, 16), (1, 32), (1, 64), (1, 128))
        super(CLNet16, self).__init__(args, kernel_sizes)


class CLNet32(CLNet):
    def __init__(self, args):
        kernel_sizes = ((1, 32), (1, 64), (1, 128), (1, 256))
        super(CLNet32, self).__init__(args, kernel_sizes)


class CLNet64(CLNet):
    def __init__(self, args):
        kernel_sizes = ((1, 64), (1, 128), (1, 256), (1, 512))
        super(CLNet64, self).__init__(args, kernel_sizes)


class CLNet16_noSe(nn.Module):
    def __init__(self, args, kernel_sizes=((1, 16), (1, 32), (1, 64), (1, 128))):
        super(CLNet16_noSe, self).__init__()
        self.TDConvModules = nn.ModuleList(
            [TDConvModule(1, 4, kernel=kernel_size, step=(1, 1)) for kernel_size in kernel_sizes]
        )
        self.SpatialDepthwiseConv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(args.eeg_channel, 1), stride=(1, 1), groups=16),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=0.25)
        )
        self.TemporalGateConv = TemporalGateConv()
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(480, args.num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        outputs = [tdconv_block(x) for tdconv_block in self.TDConvModules]
        x = torch.cat(outputs, dim=1)
        x = self.SpatialDepthwiseConv(x)
        x = self.TemporalGateConv(x)
        x = self.classify(x)
        return x


class CLNet16_noTGC(nn.Module):
    def __init__(self, args, kernel_sizes=((1, 16), (1, 32), (1, 64), (1, 128))):
        super(CLNet16_noTGC, self).__init__()
        self.TDConvModules = nn.ModuleList(
            [TDConvModule(1, 4, kernel=kernel_size, step=(1, 1)) for kernel_size in kernel_sizes]
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )
        self.SpatialDepthwiseConv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(args.eeg_channel, 1), stride=(1, 1), groups=16),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=0.25)
        )
        self.TemporalGateConv = TemporalGateConv()
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(480, args.num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        outputs = [tdconv_block(x) for tdconv_block in self.TDConvModules]
        x = torch.cat(outputs, dim=1)
        x = x * self.se(x)
        x = self.SpatialDepthwiseConv(x)
        x = self.TemporalGateConv(x)
        x = self.classify(x)
        return x


class CLNet16_noSe_noTGC(nn.Module):
    def __init__(self, args, kernel_sizes=((1, 16), (1, 32), (1, 64), (1, 128))):
        super(CLNet16_noSe_noTGC, self).__init__()
        self.TDConvModules = nn.ModuleList(
            [TDConvModule(1, 4, kernel=kernel_size, step=(1, 1)) for kernel_size in kernel_sizes]
        )
        self.SpatialDepthwiseConv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(args.eeg_channel, 1), stride=(1, 1), groups=16),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=0.25)
        )
        self.TemporalGateConv = TemporalGateConv()
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(480, args.num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        outputs = [tdconv_block(x) for tdconv_block in self.TDConvModules]
        x = torch.cat(outputs, dim=1)
        x = self.SpatialDepthwiseConv(x)
        x = self.TemporalGateConv(x)
        x = self.classify(x)
        return x
