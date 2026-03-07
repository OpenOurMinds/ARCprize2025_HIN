import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetLayer(nn.Module):
    """
    A single residual block for the VAE's encoder and decoder.
    This helps prevent the vanishing gradient problem and allows for deeper networks.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to handle changes in dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add the residual
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class AttentionLayer(nn.Module):
    """
    A simplified attention mechanism to help the model focus on important features
    in the input grid.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv_q = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv_k = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Reshape to create queries, keys, and values
        q = self.conv_q(x).view(batch_size, -1, H * W).permute(0, 2, 1) # (N, H*W, C/8)
        k = self.conv_k(x).view(batch_size, -1, H * W)                  # (N, C/8, H*W)
        v = self.conv_v(x).view(batch_size, -1, H * W)                  # (N, C, H*W)
        
        # Calculate attention map
        attention_map = torch.bmm(q, k)  # (N, H*W, H*W)
        attention_map = F.softmax(attention_map, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(v, attention_map.permute(0, 2, 1)) # (N, C, H*W)
        out = out.view(batch_size, C, H, W)
        
        # Add the attention output to the original input (residual connection)
        out = self.gamma * out + x
        return out
