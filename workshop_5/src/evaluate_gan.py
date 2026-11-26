"""
Generation quality of the vanilla GAN and the WGAN-GP (run generate_samples.py first).

Metrics (all with real-vs-real reference values, so they can be interpreted):
  - Frechet distance in the Workshop 2 ResNet1D feature space, against the
    training abnormal beats (what the GAN saw) AND against the abnormal beats of
    the VALIDATION patients (never seen): measures generalization, not copying.
  - Plausibility: share classified as abnormal by the Workshop 2 ResNet1D.
  - Diversity: mean distance between random pairs (mode collapse check).
  - Memorization: distance to the nearest training beat; reference = the same
    distance for real validation beats. Share of "near copies" (closer than the
    1st percentile of the reference).
  - Class coverage (domain-specific): AAMI class (S/V/F) of the nearest real
    training beat; a GAN that ignores a class (partial mode collapse) shows it here.
  - R-peak alignment (domain-specific): share of beats whose largest deflection
    lies within +/-15 ms of the annotated R-peak position (sample 72).
  - Spectral distance: L1 distance between the average power spectra.
Also draws the training-curve comparison of both models.

Usage (from workshop_5/):  python src/evaluate_gan.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_gan import CLASSES, W5_DIR, abnormal_train_beats, load_splits, load_w2_classifier, scale_beats
from gan_metrics import diversity, frechet_distance, nearest_neighbor_distance, w2_features_and_probs

DATA = W5_DIR / "data"
REPORTS = W5_DIR / "reports"
RUNS = W5_DIR / "runs" / "workshop5"
R_PEAK, TOL = 72, int(round(0.015 * 360))


def r_peak_aligned(X):
    return float(np.mean(np.abs(np.argmax(np.abs(X), axis=1) - R_PEAK) <= TOL))


def mean_spectrum(X):
    s = np.abs(np.fft.rfft(X, axis=1)) ** 2
    return s.mean(0) / s.mean(0).sum()


def main():
    REPORTS.mkdir(exist_ok=True)
    train_X, train_lab = abnormal_train_beats()
    X_val, y_val, meta_val = load_splits()["val"]
    val_X = scale_beats(X_val[y_val == 1])
    w2, thr = load_w2_classifier()

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(train_X))
    half_a, half_b = perm[: len(perm) // 2], perm[len(perm) // 2:]

    tr_f, tr_p = w2_features_and_probs(w2, train_X)
    va_f, va_p = w2_features_and_probs(w2, val_X)
    ref_nn = nearest_neighbor_distance(val_X, train_X)          # real unseen -> train
    near_copy = np.percentile(ref_nn, 1)
    real_spec = mean_spectrum(train_X)
    real_cov = np.bincount(train_lab, minlength=len(CLASSES)) / len(train_lab)

    sources = {"Real train (half A vs half B)": train_X[half_a],
               "Real validation patients": val_X}
    for key, name in [("vanilla", "Vanilla GAN"), ("wgan", "WGAN-GP")]:
        sources[name] = np.load(DATA / f"synthetic_{key}.npy")

    rows, coverage = [], {"Real train": real_cov}
    for name, X in sources.items():
        f, p = w2_features_and_probs(w2, X)
        ref_train = tr_f[half_b] if name.startswith("Real train") else tr_f
        nn_d = nearest_neighbor_distance(X, train_X if not name.startswith("Real train")
                                         else train_X[half_b])
        idx_nn = None
        if name in ("Vanilla GAN", "WGAN-GP"):
            # class of the nearest real training beat
            import torch
            ref = torch.from_numpy(train_X)
            idx_nn = np.concatenate([torch.cdist(torch.from_numpy(X[i:i + 512]), ref).argmin(1).numpy()
                                     for i in range(0, len(X), 512)])
            coverage[name] = np.bincount(train_lab[idx_nn], minlength=len(CLASSES)) / len(X)
        rows.append({
            "source": name, "n": len(X),
            "FD vs train abnormal": frechet_distance(ref_train, f),
            "FD vs validation abnormal": (frechet_distance(va_f, f)
                                          if name != "Real validation patients" else np.nan),
            "W2 abnormal (%)": 100 * float((p >= thr).mean()),
            "diversity": diversity(X),
            "NN distance to train (median)": float(np.median(nn_d)),
            "near copies (%)": 100 * float((nn_d < near_copy).mean()),
            "R-peak aligned (%)": 100 * r_peak_aligned(X),
            "spectral L1 vs train": float(np.abs(mean_spectrum(X) - real_spec).sum()),
        })
    table = pd.DataFrame(rows).set_index("source")
    print("Generation quality (abnormal beats, [-1, 1] scale):")
    print(table.round(3).to_string())
    table.to_csv(REPORTS / "generation_quality.csv")

    cov = pd.DataFrame(coverage, index=CLASSES).T * 100
    print("\nClass coverage: AAMI class of the nearest real training beat (%):")
    print(cov.round(1).to_string())
    cov.to_csv(REPORTS / "class_coverage.csv")

    # Diversity over training (mode collapse monitoring) and training curves
    hv = pd.read_csv(RUNS / "gan_vanilla" / "history.csv")
    hw = pd.read_csv(RUNS / "gan_wgan" / "history.csv")
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    a = axes[0, 0]
    a.plot(hv.epoch, hv.d_loss, label="D loss")
    a.plot(hv.epoch, hv.g_loss, label="G loss")
    a.set_title("Vanilla GAN: BCE losses")
    a = axes[0, 1]
    a.plot(hv.epoch, hv.acc_real, label="D accuracy on real")
    a.plot(hv.epoch, hv.acc_fake, label="D accuracy on fake")
    a.set_ylim(0, 1.02)
    a.set_title("Vanilla GAN: discriminator accuracy")
    a = axes[1, 0]
    a.plot(hw.epoch, hw.w_dist, label="Wasserstein estimate", color="C3")
    a.plot(hw.epoch, hw.d_loss, label="critic loss", color="C1", alpha=0.7)
    a.set_title("WGAN-GP: Wasserstein estimate and critic loss")
    a = axes[1, 1]
    a.plot(hw.epoch, hw.gp, color="C4", label="gradient penalty")
    a.plot(hw.epoch, hw.g_loss / 100, color="C2", alpha=0.7, label="G loss / 100")
    a.set_title("WGAN-GP: gradient penalty")
    for h, name, c in [(hv, "Vanilla GAN", "C0"), (hw, "WGAN-GP", "C3")]:
        e = h.dropna(subset=["fd"])
        axes[0, 2].plot(e.epoch, e.fd, "o-", ms=3, color=c, label=name)
        axes[1, 2].plot(e.epoch, e.diversity, "o-", ms=3, color=c, label=name)
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("Frechet distance vs train (log)")
    axes[1, 2].axhline(table.loc["Real train (half A vs half B)", "diversity"], color="black",
                       ls="--", label="real beats")
    axes[1, 2].set_title("Diversity of generated beats")
    for a in axes.flat:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "training_comparison.png", dpi=150)
    plt.close(fig)

    # Stability summary
    ev_v, ev_w = hv.dropna(subset=["fd"]), hw.dropna(subset=["fd"])
    late_v, late_w = ev_v[ev_v.epoch > ev_v.epoch.max() / 2], ev_w[ev_w.epoch > ev_w.epoch.max() / 2]
    print("\nTraining stability (second half of training):")
    print(f"  Vanilla GAN: FD mean {late_v.fd.mean():.2f}, std {late_v.fd.std():.2f}; "
          f"epochs with D accuracy on fake > 95%: {(hv.acc_fake > 0.95).sum()}")
    print(f"  WGAN-GP    : FD mean {late_w.fd.mean():.2f}, std {late_w.fd.std():.2f}; "
          f"Wasserstein estimate {hw.w_dist.iloc[0]:.2f} -> {hw.w_dist.iloc[-1]:.2f} "
          f"(epochs where it increased: {(hw.w_dist.diff() > 0).sum()} of {len(hw) - 1})")
    print(f"\nSaved tables and figures to {REPORTS}")


if __name__ == "__main__":
    main()
