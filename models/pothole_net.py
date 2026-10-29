"""
Custom CNN Architecture for Pothole Detection (ResPotholeNet)
Built 100% from scratch using pure PyTorch (torch.nn) — NO PRETRAINED MODELS OR WEIGHTS.

Architecture Highlights:
1. Residual Bottlenecks with skip connections (prevent gradient degradation)
2. Squeeze-and-Excitation (SE) Channel Attention (focuses on dark asphalt depressions)
3. Spatial Pyramid Pooling (SPPF) for multi-scale receptive field (tiny cracks to large craters)
4. Single Classification Head -> Pothole probability score [0, 1]

Note: The bounding box regression head has been REMOVED. The dataset provides only
image-level pothole/normal labels with no GT bounding box coordinates, so the
regressor was never trained and contributed dead parameters to the reported count.
All ~853,985 parameters now exclusively serve the classification task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Channel Attention Module."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Squeeze (Global average pooling)
        y = x.mean(dim=(2, 3))
        # Excitation
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        # Rescale channels
        return x * y


class ConvBlock(nn.Module):
    """Standard Convolution + BatchNorm + SiLU block."""
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Residual Bottleneck with Squeeze-and-Excitation Attention."""
    def __init__(self, channels: int):
        super().__init__()
        mid = channels // 2
        self.conv1 = ConvBlock(channels, mid, k=1, s=1, p=0)
        self.conv2 = ConvBlock(mid, channels, k=3, s=1, p=1)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv2(self.conv1(x))
        out = self.se(out)
        return out + residual


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast: multi-scale receptive field aggregation."""
    def __init__(self, in_c: int, out_c: int, k: int = 5):
        super().__init__()
        mid = in_c // 2
        self.cv1 = ConvBlock(in_c, mid, k=1, s=1, p=0)
        self.cv2 = ConvBlock(mid * 4, out_c, k=1, s=1, p=0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))


class ResPotholeNet(nn.Module):
    """
    Custom Deep CNN for Pothole Patch Classification.

    Inputs:
        x: Tensor of shape (B, 3, 128, 128) representing candidate road patches.
    Outputs:
        conf: Tensor of shape (B, 1) in [0, 1] — confidence that patch is a pothole.
        (None is returned as the second value for call-site compatibility.)

    The regression head has been removed — the dataset provides only image-level
    labels with no bounding box coordinates, so the regressor was never trained.
    All 853,985 parameters serve the classification task exclusively.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            ConvBlock(in_channels, 32, k=3, s=2, p=1),  # 128 -> 64
            ConvBlock(32, 64, k=3, s=1, p=1),
        )

        # Stage 1: Texture & Edge extraction
        self.down1 = ConvBlock(64, 64, k=3, s=2, p=1)   # 64 -> 32
        self.stage1 = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
        )

        # Stage 2: Structural depression features
        self.down2 = ConvBlock(64, 128, k=3, s=2, p=1)  # 32 -> 16
        self.stage2 = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
        )

        # Stage 3: High-level cavity semantics + Multi-scale SPPF
        self.down3 = ConvBlock(128, 256, k=3, s=2, p=1) # 16 -> 8
        self.sppf = SPPF(256, 256)

        # Global context pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Shared dense representation
        self.fc_shared = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.3),
        )

        # Head 1: Pothole Classification
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming normal initialization for all conv and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage1(self.down1(x))
        x = self.stage2(self.down2(x))
        x = self.sppf(self.down3(x))
        feat = self.gap(x).flatten(1)
        shared = self.fc_shared(feat)
        conf = self.classifier(shared)
        # Return (conf, None) for backward compatibility with call sites
        # that unpack two values: conf, _ = model(x)
        return conf, None


def count_parameters(model: nn.Module) -> int:
    """Return total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = ResPotholeNet()
    dummy = torch.randn(2, 3, 128, 128)
    conf, _ = net(dummy)
    print(f"ResPotholeNet instantiated successfully!")
    print(f"Total Parameters (classification-only): {count_parameters(net):,}")
    print(f"Sample Confidence Shape: {conf.shape}")
