import torch
import torch.nn as nn
from approx_conv_module import ApproximateConv2d

class BottleneckApprox(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BottleneckApprox, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = ApproximateConv2d(in_channels, 4 * growth_rate, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = ApproximateConv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = torch.relu(self.bn1(x))
        out = self.conv1(out)
        out = torch.relu(self.bn2(out))
        out = self.conv2(out)
        out = torch.cat([x, out], 1)
        return out

class TransitionApprox(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionApprox, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = ApproximateConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = torch.relu(self.bn(x))
        out = self.conv(out)
        out = self.pool(out)
        return out

class DenseNetApprox(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNetApprox, self).__init__()
        num_channels = 2 * growth_rate
        self.conv1 = ApproximateConv2d(3, num_channels, kernel_size=3, stride=1, padding=1)

        self.dense1 = self._make_dense_layers(block, num_channels, nblocks[0], growth_rate)
        num_channels += nblocks[0] * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans1 = TransitionApprox(num_channels, out_channels)
        num_channels = out_channels

        self.dense2 = self._make_dense_layers(block, num_channels, nblocks[1], growth_rate)
        num_channels += nblocks[1] * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans2 = TransitionApprox(num_channels, out_channels)
        num_channels = out_channels

        self.dense3 = self._make_dense_layers(block, num_channels, nblocks[2], growth_rate)
        num_channels += nblocks[2] * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans3 = TransitionApprox(num_channels, out_channels)
        num_channels = out_channels

        self.dense4 = self._make_dense_layers(block, num_channels, nblocks[3], growth_rate)
        num_channels += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_channels)
        self.linear = nn.Linear(num_channels, num_classes)

    def _make_dense_layers(self, block, in_channels, nblocks, growth_rate):
        layers = []
        for _ in range(nblocks):
            layers.append(block(in_channels, growth_rate))
            in_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        out = self.trans3(out)
        out = self.dense4(out)
        out = torch.relu(self.bn(out))
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

# Ax-DenseNet-121
def DenseNet121Approx():
    return DenseNetApprox(BottleneckApprox, [6, 12, 24, 16], growth_rate=32)

# Ax-DenseNet-169
def DenseNet169Approx():
    return DenseNetApprox(BottleneckApprox, [6, 12, 32, 32], growth_rate=32)
