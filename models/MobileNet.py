import torch
import torch.nn as nn
from approx_conv_module import ApproximateConv2d

# MobileNet Approx
class MobileNetApprox(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetApprox, self).__init__()
        def conv_bn(in_channels, out_channels, stride):
            return nn.Sequential(
                ApproximateConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def conv_dw(in_channels, out_channels, stride):
            return nn.Sequential(
                ApproximateConv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),

                ApproximateConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            *[conv_dw(512, 512, 1) for _ in range(5)],
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

# Model functions
def MobileNetV1Approx():
    return MobileNetApprox(num_classes=10)

def EfficientNetB0Approx():
    return EfficientNetApprox(num_classes=10)