"""
GAN models for ECG beats (DCGAN-style, 1D).

Data: one beat = 216 samples (0.6 s at 360 Hz), min-max scaled to [-1, 1] per beat.

Generator G   : z (64) -> Linear -> (128, 27) -> 3 x ConvTranspose1d (x2 each)
                -> (16, 216) -> Conv1d -> Tanh -> (1, 216)
Discriminator : Conv1d with stride 2 (216 -> 108 -> 54 -> 27) -> Linear -> Sigmoid
                (probability that the beat is real)
Critic        : same convolutional body, NO BatchNorm (LayerNorm instead: the
                gradient penalty is computed per sample, and BatchNorm mixes samples)
                and NO Sigmoid (unbounded real-valued score), for WGAN-GP.

Optional conditioning (n_classes > 0): G receives a learned label embedding
concatenated to z; D/Critic receive the label as an extra input channel.

Usage (from workshop_5/):  python src/gan.py     # shape and parameter check
"""
import torch
import torch.nn as nn

BEAT_LEN = 216
Z_DIM = 64


def init_weights(m):
    """DCGAN initialization: N(0, 0.02) for conv/linear, N(1, 0.02) for norm scales."""
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)) and m.weight is not None:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, n_classes=0, emb_dim=16):
        super().__init__()
        self.z_dim, self.n_classes = z_dim, n_classes
        in_dim = z_dim + (emb_dim if n_classes else 0)
        self.embed = nn.Embedding(n_classes, emb_dim) if n_classes else None
        self.fc = nn.Sequential(nn.Linear(in_dim, 128 * 27), nn.BatchNorm1d(128 * 27), nn.ReLU(True))

        def up(c_in, c_out):        # ConvTranspose1d k4 s2 p1 doubles the length exactly
            return [nn.ConvTranspose1d(c_in, c_out, 4, 2, 1, bias=False),
                    nn.BatchNorm1d(c_out), nn.ReLU(True)]

        self.net = nn.Sequential(*up(128, 64), *up(64, 32), *up(32, 16),   # 27 -> 54 -> 108 -> 216
                                 nn.Conv1d(16, 1, 7, padding=3), nn.Tanh())
        self.apply(init_weights)

    def forward(self, z, labels=None):
        if self.n_classes:
            z = torch.cat([z, self.embed(labels)], dim=1)
        return self.net(self.fc(z).view(-1, 128, 27))

    def sample(self, n, labels=None):
        z = torch.randn(n, self.z_dim)
        return self.forward(z, labels)


class Discriminator(nn.Module):
    """
    critic=False: vanilla GAN discriminator (BatchNorm, Sigmoid output).
    critic=True : WGAN-GP critic (LayerNorm, linear output).
    """

    def __init__(self, critic=False, n_classes=0):
        super().__init__()
        self.critic, self.n_classes = critic, n_classes
        in_ch = 1 + (1 if n_classes else 0)
        self.embed = nn.Embedding(n_classes, BEAT_LEN) if n_classes else None

        def norm(c, length):
            return nn.LayerNorm([c, length]) if critic else nn.BatchNorm1d(c)

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, 2, 3), nn.LeakyReLU(0.2, True),                 # (32, 108)
            nn.Conv1d(32, 64, 5, 2, 2), norm(64, 54), nn.LeakyReLU(0.2, True),      # (64, 54)
            nn.Conv1d(64, 128, 5, 2, 2), norm(128, 27), nn.LeakyReLU(0.2, True),    # (128, 27)
            nn.Flatten(),
            nn.Linear(128 * 27, 1),
        )
        self.apply(init_weights)

    def forward(self, x, labels=None):
        if self.n_classes:
            x = torch.cat([x, self.embed(labels).unsqueeze(1)], dim=1)
        out = self.net(x).squeeze(1)
        return out if self.critic else torch.sigmoid(out)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    for n_classes in [0, 3]:
        labels = torch.randint(0, 3, (8,)) if n_classes else None
        G = Generator(n_classes=n_classes).eval()
        D = Discriminator(critic=False, n_classes=n_classes).eval()
        C = Discriminator(critic=True, n_classes=n_classes).eval()
        with torch.no_grad():
            fake = G.sample(8, labels)
            d, c = D(fake, labels), C(fake, labels)
        tag = "conditional (3 classes)" if n_classes else "unconditional"
        print(f"\n{tag}")
        print(f"  Generator     {count_parameters(G):>9,} params  z {tuple((8, Z_DIM))} -> {tuple(fake.shape)}"
              f"  range [{fake.min():.2f}, {fake.max():.2f}]")
        print(f"  Discriminator {count_parameters(D):>9,} params  -> {tuple(d.shape)} in [0, 1]: "
              f"{bool(((d >= 0) & (d <= 1)).all())}")
        print(f"  Critic        {count_parameters(C):>9,} params  -> {tuple(c.shape)} (real-valued)")
        assert fake.shape == (8, 1, BEAT_LEN)
    print("\nForward pass OK")
