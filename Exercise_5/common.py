"""Κοινή υποδομή για την Άσκηση 5 (Bonus) — DS-CDMA/Rake και OFDM.

Περιέχει:
  * Q-function και θεωρητικές καμπύλες BER για MRC / selection combining
  * εκτίμηση τάξης διαφοροποίησης από την κλίση της BER καμπύλης
  * βοηθητικά για τα σχήματα (reference slopes f_i = SNR^-i, αποθήκευση)

Αντιστοιχία με τη θεωρία:
  * mrc_ber()            -> γενίκευση των εξ. (3.11)-(3.25), 3_Chapter_Diversity.pdf.
                            Είναι η θεωρητική καμπύλη τόσο για την έξοδο του Rake
                            (εξ. 4.16, Chapter_Frequency_Diversity_1.pdf) όσο και για
                            την επανάληψη στις συχνότητες (εδάφιο 4.2, Chapter_OFDM.pdf)
                            — και στις δύο περιπτώσεις ο δέκτης είναι MRC πάνω σε
                            i.i.d. Rayleigh κλάδους.
  * mrc_ber_asymptotic() -> εξ. (3.25): P_e ~ C(2L-1,L)/(4*gbar)^L, από όπου προκύπτει
                            ότι η κλίση της log-log καμπύλης είναι -L.
  * sc_ber()             -> selection combining: F_max(γ) = (1 - e^{-γ/gbar})^L.

ΠΡΟΣΟΧΗ στο `gbar` (μέσο SNR ανά κλάδο), που διαφέρει ανά μέρος:
  * Μέρος Α: gbar = SNR/L,  γιατί h_l ~ CN(0, 1/L)  -> E|h_l|^2 = 1/L.
  * Μέρος Β: gbar = SNR/2,  γιατί SNR = 2/σ_n^2 ενώ E|H_n|^2 = 1 (εξ. 4.15).
"""

import os
import sys

import numpy as np
from scipy.special import comb
from scipy.stats import norm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Το console των Windows είναι cp1252 -> τα ελληνικά prints θα έσκαγαν.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "figure.figsize": (7.5, 5.2),
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
        "legend.fontsize": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    }
)


def Q(x):
    """Q(x) = P(Z > x), Z ~ N(0,1)."""
    return norm.sf(x)


def db2lin(snr_db):
    return 10.0 ** (np.asarray(snr_db, dtype=float) / 10.0)


# --------------------------------------------------------------------------
# Θεωρητικές καμπύλες (BPSK / ανά διάσταση 4-QAM, i.i.d. Rayleigh κλάδοι)
# --------------------------------------------------------------------------
def mrc_ber(gbar, L):
    """Ακριβές BER MRC με L i.i.d. Rayleigh κλάδους, μέσο SNR/κλάδο `gbar`.

    P(e|γ) = Q(sqrt(2γ)) με γ = Σ_l γ_l  (γεν. της εξ. 3.11 -> 3.25).
    """
    gbar = np.asarray(gbar, dtype=float)
    mu = np.sqrt(gbar / (1.0 + gbar))
    acc = np.zeros_like(mu)
    for k in range(L):
        acc += comb(L - 1 + k, k) * ((1.0 + mu) / 2.0) ** k
    return ((1.0 - mu) / 2.0) ** L * acc


def mrc_ber_asymptotic(gbar, L):
    """Ασυμπτωτική μορφή: P(e) ≈ C(2L-1, L) / (4*gbar)^L  (εξ. 3.25)."""
    gbar = np.asarray(gbar, dtype=float)
    return comb(2 * L - 1, L) / (4.0 * gbar) ** L


def sc_ber(gbar, L):
    """Ακριβές BER selection combining (1 finger, ισχυρότερος κλάδος).

    F_max(γ) = (1 - e^{-γ/gbar})^L  ->  ανάπτυγμα σε άθροισμα εκθετικών.
    """
    gbar = np.asarray(gbar, dtype=float)
    acc = np.zeros_like(gbar)
    for k in range(L):
        gk = gbar / (k + 1.0)
        mu = np.sqrt(gk / (1.0 + gk))
        acc += comb(L - 1, k) * (-1.0) ** k * (L / (k + 1.0)) * 0.5 * (1.0 - mu)
    return acc


# --------------------------------------------------------------------------
# Τάξη διαφοροποίησης
# --------------------------------------------------------------------------
def estimate_diversity_order(snr_db, ber, n_last=3, counts=None, min_errors=50):
    """Κλίση της log10(BER) ως προς log10(SNR_lin) στα τελευταία `n_last` σημεία.

    Επιστρέφει -slope, δηλαδή την (εμπειρική) τάξη διαφοροποίησης.

    Αγνοούνται τα σημεία με BER = 0, καθώς και — αν δοθεί το `counts` (πλήθος
    σφαλμάτων ανά σημείο) — όσα μετρήθηκαν με λιγότερα από `min_errors`
    σφάλματα: η σχετική τους αβεβαιότητα (~1/sqrt(errors)) είναι τόσο μεγάλη
    που θα αλλοίωνε την εκτίμηση της κλίσης.
    """
    snr_db = np.asarray(snr_db, dtype=float)
    ber = np.asarray(ber, dtype=float)
    ok = np.isfinite(ber) & (ber > 0)
    if counts is not None:
        ok &= np.asarray(counts) >= min_errors
    if ok.sum() < 2:
        return np.nan
    x = np.log10(db2lin(snr_db[ok]))[-n_last:]
    y = np.log10(ber[ok])[-n_last:]
    if x.size < 2:
        return np.nan
    slope = np.polyfit(x, y, 1)[0]
    return -slope


def plot_reference_slopes(ax, snr_db, orders=(1, 2, 3), colors=("0.25", "0.45", "0.65")):
    """Σχεδιάζει τις f_i = 1/SNR^i, i ∈ `orders` (ζητούμενα Α.1, Α.2)."""
    snr_lin = db2lin(snr_db)
    for i, c in zip(orders, colors):
        ax.semilogy(
            snr_db,
            snr_lin ** (-float(i)),
            "--",
            lw=1.1,
            color=c,
            label=rf"$f_{{{i}}} = \mathrm{{SNR}}^{{-{i}}}$",
        )


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    [figure] figures/{name}")
    return path
