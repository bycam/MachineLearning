"""
1D residual CNN (a small ResNet for ECG beats).

Input : (batch, 1, 216)  -> one ECG lead, 0.6 s at 360 Hz
Output: (batch,)         -> one logit; sigmoid(logit) = P(abnormal)

Usage (from workshop_2/):  python src/model.py
"""
import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """
    conv -> BN -> ReLU -> dropout -> conv -> BN, plus a skip connection.

    The skip connection (x + F(x)) lets gradients flow straight back to
    the first layers, so the network can be deeper without the
    vanishing-gradient problem. When the number of channels or the length
    changes, a 1x1 convolution adapts the shortcut so the shapes match.
    """

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=5, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                               padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ECGResNet1D(nn.Module):
    """
    stem (wide kernel) -> 4 residual stages -> global average pool -> linear.

    Length 216 -> 216 -> 108 -> 54 -> 27 while channels grow 32 -> 128:
    early layers see small details (onset of the QRS), later layers
    combine them into the whole beat morphology.
    """

    def __init__(self, in_channels=1, base_channels=32, dropout=0.3):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
        )
        self.stage1 = ResidualBlock1D(c, c, stride=1)
        self.stage2 = ResidualBlock1D(c, 2 * c, stride=2)
        self.stage3 = ResidualBlock1D(2 * c, 4 * c, stride=2)
        self.stage4 = ResidualBlock1D(4 * c, 4 * c, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(4 * c, 1),
        )

    def forward(self, x):
        for layer in [self.stem, self.stage1, self.stage2,
                      self.stage3, self.stage4, self.pool, self.head]:
            x = layer(x)
        return x.squeeze(1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = ECGResNet1D()
    model.eval()

    x = torch.randn(8, 1, 216)
    print(f"{'Layer':<8}{'Output shape':>18}{'Params':>10}")
    print(f"{'input':<8}{str(tuple(x.shape)):>18}")
    with torch.no_grad():
        for name in ["stem", "stage1", "stage2", "stage3",
                     "stage4", "pool", "head"]:
            layer = getattr(model, name)
            x = layer(x)
            print(f"{name:<8}{str(tuple(x.shape)):>18}"
                  f"{count_parameters(layer):>10,}")

    out = model(torch.randn(8, 1, 216))
    assert out.shape == (8,), f"Unexpected output shape {out.shape}"
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
    print(f"Forward pass OK: output shape {tuple(out.shape)}")
    print(f"P(abnormal) for the 8 random beats: {torch.sigmoid(out).detach().numpy().round(2)}")
