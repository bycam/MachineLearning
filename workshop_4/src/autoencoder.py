"""
Convolutional Autoencoder (AE) and Variational Autoencoder (VAE) for ECG beats.

Input : (batch, 1, 216)  one ECG beat (0.6 s at 360 Hz), z-scored per beat
Latent: z_dim values
Output: (batch, 1, 216)  reconstruction (linear output: z-scored beats are unbounded)

Encoder  : 3 strided Conv1d blocks (216 -> 108 -> 54 -> 27) + Linear -> bottleneck
Decoder  : mirror of the encoder with ConvTranspose1d (27 -> 54 -> 108 -> 216)
AE       : bottleneck = one vector z
VAE      : bottleneck = mean mu and log-variance log(sigma^2);
           z = mu + sigma * eps, eps ~ N(0, I)   (reparameterization trick)

Usage (from workshop_4/):  python src/autoencoder.py   # shape and parameter check
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

BEAT_LEN = 216
FEAT_CH, FEAT_LEN = 64, 27            # encoder output before the bottleneck


def conv_block(c_in, c_out, k):
    """Conv1d with stride 2 halves the length; BN + LeakyReLU."""
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, k, stride=2, padding=k // 2),
        nn.BatchNorm1d(c_out),
        nn.LeakyReLU(0.1),
    )


def deconv_block(c_in, c_out):
    """ConvTranspose1d (k=4, stride 2, padding 1) doubles the length exactly."""
    return nn.Sequential(
        nn.ConvTranspose1d(c_in, c_out, 4, stride=2, padding=1),
        nn.BatchNorm1d(c_out),
        nn.LeakyReLU(0.1),
    )


class Encoder(nn.Module):
    """(B, 1, 216) -> (B, 64 * 27) features."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(1, 16, 7),      # (16, 108)
            conv_block(16, 32, 5),     # (32, 54)
            conv_block(32, 64, 5),     # (64, 27)
            nn.Flatten(),              # 1728
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """z (B, z_dim) -> (B, 1, 216)."""

    def __init__(self, z_dim):
        super().__init__()
        self.fc = nn.Linear(z_dim, FEAT_CH * FEAT_LEN)
        self.net = nn.Sequential(
            deconv_block(64, 32),      # (32, 54)
            deconv_block(32, 16),      # (16, 108)
            deconv_block(16, 16),      # (16, 216)
            nn.Conv1d(16, 1, 7, padding=3),   # (1, 216), linear output
        )

    def forward(self, z):
        h = F.leaky_relu(self.fc(z), 0.1).view(-1, FEAT_CH, FEAT_LEN)
        return self.net(h)


class ConvAE(nn.Module):
    kind = "ae"

    def __init__(self, z_dim=32):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder()
        self.to_z = nn.Linear(FEAT_CH * FEAT_LEN, z_dim)
        self.decoder = Decoder(z_dim)

    def encode(self, x):
        return self.to_z(self.encoder(x))

    def forward(self, x):
        z = self.encode(x)
        return {"recon": self.decoder(z), "z": z}


class ConvVAE(nn.Module):
    kind = "vae"

    def __init__(self, z_dim=32):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder()
        self.to_mu = nn.Linear(FEAT_CH * FEAT_LEN, z_dim)
        self.to_logvar = nn.Linear(FEAT_CH * FEAT_LEN, z_dim)
        self.decoder = Decoder(z_dim)

    def encode(self, x):
        """Returns mu and log-variance of q(z|x)."""
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h).clamp(-10, 10)

    @staticmethod
    def reparameterize(mu, logvar):
        """z = mu + sigma * eps: the randomness is in eps, so gradients flow through mu and sigma."""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x, sample=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        return {"recon": self.decoder(z), "z": z, "mu": mu, "logvar": logvar}


def kl_divergence(mu, logvar):
    """KL(q(z|x) || N(0, I)) per example: -1/2 * sum(1 + log s^2 - mu^2 - s^2)."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def recon_error(recon, x):
    """Mean squared error per example (average over the 216 samples)."""
    return ((recon - x) ** 2).mean(dim=(1, 2))


def loss_fn(model, out, x, beta=1.0):
    """
    AE : mean MSE.
    VAE: ELBO = sum-of-squares reconstruction + beta * KL, averaged over the batch.
         The reconstruction is summed over the 216 samples (Gaussian log-likelihood
         up to a constant) so that it is on the same scale as the KL summed over z.
    Returns the loss to optimize and logging terms (per-element MSE, KL per example).
    """
    mse = recon_error(out["recon"], x)
    if model.kind == "ae":
        return mse.mean(), {"mse": mse.mean().item(), "kl": 0.0}
    kl = kl_divergence(out["mu"], out["logvar"])
    loss = (mse * BEAT_LEN + beta * kl).mean()
    return loss, {"mse": mse.mean().item(), "kl": kl.mean().item()}


def build_model(kind, z_dim):
    return {"ae": ConvAE, "vae": ConvVAE}[kind](z_dim)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(8, 1, BEAT_LEN)
    for kind in ["ae", "vae"]:
        model = build_model(kind, z_dim=32).eval()
        print(f"\n{kind.upper()}  ({count_parameters(model):,} parameters)")
        h = x
        for i, block in enumerate(model.encoder.net):
            h = block(h)
            print(f"  encoder {i}: {tuple(h.shape)}")
        with torch.no_grad():
            out = model(x)
            loss, logs = loss_fn(model, out, x, beta=1.0)
        for k, v in out.items():
            print(f"  {k:<7}: {tuple(v.shape)}")
        assert out["recon"].shape == x.shape, "reconstruction must have the input shape"
        print(f"  loss {loss.item():.3f} | per-element MSE {logs['mse']:.3f} | KL {logs['kl']:.3f}")
    print("\nForward pass OK for AE and VAE")
