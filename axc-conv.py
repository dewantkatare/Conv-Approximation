import torch
import torch.nn as nn
import torch.nn.functional as F

# Approximate Convolution Module
class ApproximateConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ApproximateConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        # mean of absolute values
        mu_w = torch.mean(torch.abs(self.weights))
        
        x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        w_unf = self.weights.view(self.weights.size(0), -1)
        
        min_val, _ = torch.min(x_unf.unsqueeze(1), w_unf.unsqueeze(2))
        min_val = min_val.view(x.size(0), self.out_channels, -1)
        
        # Approximate convolution output
        z_approx = mu_w * torch.sum(min_val, dim=2)
        z_approx = z_approx.view(x.size(0), self.out_channels, int((x.size(2) + 2 * self.padding - self.kernel_size) / self.stride + 1), int((x.size(3) + 2 * self.padding - self.kernel_size) / self.stride + 1))
        
        return z_approx
