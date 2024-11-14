import torch
import torch.nn as nn
from approx_conv_module import ApproximateConv2d

# EfficientNet Approx
class MBConvApprox(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(MBConvApprox, self).__init__()
        mid_channels = in_channels * expansion_factor
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        self.expand_conv = nn.Sequential(
            ApproximateConv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        ) if expansion_factor != 1 else nn.Identity()

        self.depthwise_conv = nn.Sequential(
            ApproximateConv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.pointwise_conv = nn.Sequential(
            ApproximateConv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.expand_conv(x)
        out = self.depthwise_conv(out)
        out = self.pointwise_conv(out)
        if self.use_residual:
            out += x
        return out

class EfficientNetApprox(nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientNetApprox, self).__init__()
        self.stem_conv = nn.Sequential(
            ApproximateConv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.Sequential(
            MBConvApprox(32, 16, expansion_factor=1, stride=1),
            MBConvApprox(16, 24, expansion_factor=6, stride=2),
            MBConvApprox(24, 24, expansion_factor=6, stride=1),
            MBConvApprox(24, 40, expansion_factor=6, stride=2),
            MBConvApprox(40, 40, expansion_factor=6, stride=1),
            MBConvApprox(40, 80, expansion_factor=6, stride=2),
            MBConvApprox(80, 80, expansion_factor=6, stride=1),
            MBConvApprox(80, 80, expansion_factor=6, stride=1),
            MBConvApprox(80, 112, expansion_factor=6, stride=1),
            MBConvApprox(112, 112, expansion_factor=6, stride=1),
            MBConvApprox(112, 192, expansion_factor=6, stride=2),
            MBConvApprox(192, 192, expansion_factor=6, stride=1),
            MBConvApprox(192, 192, expansion_factor=6, stride=1),
            MBConvApprox(192, 320, expansion_factor=6, stride=1)
        )

        self.head_conv = nn.Sequential(
            ApproximateConv2d(320, 1280, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def EfficientNetB0Approx():
    return EfficientNetApprox(num_classes=10)