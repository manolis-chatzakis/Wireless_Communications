"""Άσκηση 5 (Bonus) — Μέρος Β: διαμόρφωση OFDM και διαφοροποίηση συχνότητας.

Μοντέλο (σύμφωνα με την εκφώνηση):
  * κανάλι h_l, l = 0..L-1, με h_l i.i.d. ~ CN(0, 1/L)  (ενδεικτικά L = 4)
  * είσοδος 4-QAM d[k] ∈ {±1 ± j}, ισοπίθανη, k = 1..N  (ενδεικτικά N = 64, 128)
  * OFDM με κυκλικό πρόθεμα μήκους L-1
  * SNR_dB = 10 log10(2 / σ_n^2)   (βλ. απόδειξη στη snr_definition_proof())

Ζητούμενα που καλύπτονται:
  B1-B4  κατασκευή καναλιού/συμβόλων, μετάδοση OFDM, επαλήθευση N παράλληλων
         flat καναλιών σε αθόρυβο περιβάλλον
  B5     προσθήκη θορύβου (SNR = 15 dB) και συσχέτιση των θέσεων σφάλματος με
         τα subcarriers όπου το |H[k]| είναι μικρό
  B6-B8  διαφοροποίηση με «επανάληψη στις συχνότητες» τάξης 1, 2, 4:
         BER για SNR_dB = 0:4:20 και επιβεβαίωση της κλίσης σε log-log άξονες

Αντιστοιχία με τη θεωρία (Chapter_OFDM.pdf):
  εξ. (4.1)       y[m] = Σ_l h_l x[m-l] + w[m]
  εξ. (4.3)-(4.4) d = IDFT(d~), κυκλικό πρόθεμα μήκους L-1
  εξ. (4.6)       y' = τα Nc δείγματα y[L..Nc+L-1]  (αγνοούμε L-1 στην αρχή/τέλος)
  εξ. (4.7)       y' = d (*)_Nc h + w   (κυκλική συνέλιξη)
  εξ. (4.11)      DFT με συντελεστή 1/sqrt(Nc) ΚΑΙ στους δύο μετασχηματισμούς
  εξ. (4.14)      y~'_n = h~_n d~_n + w~_n
  εξ. (4.15)      h~_n = Σ_l h_l e^{-j2πln/Nc}   (ΜΗ κανονικοποιημένος DFT του h)
  εξ. (4.21)      E[H_i H_k^*] = 0 όταν i-k = c·Nc/L  -> ανεξάρτητοι φορείς
  εδάφιο 4.2      επανάληψη σε M subcarriers με βήμα Nc/M -> διαφοροποίηση M

Η υποσημείωση 1 της εξ. (4.11) τονίζει ότι η MATLAB/numpy δεν χρησιμοποιεί τον
συντελεστή 1/sqrt(Nc)· γι' αυτό καλούμε παντού `norm="ortho"`. Με αυτή τη
σύμβαση ο λευκός θόρυβος του πεδίου χρόνου παραμένει λευκός με την ίδια
διασπορά στο πεδίο συχνότητας και ο ορισμός SNR = 2/σ_n^2 ισχύει ακριβώς.

Εκτέλεση:  python ofdm_sim.py
"""

import time

import numpy as np

from common import (
    FIG_DIR,
    db2lin,
    estimate_diversity_order,
    mrc_ber,
    plt,
    save,
)

SEED = 2021030061  # ΑΜ, για αναπαραγωγιμότητα


# ==========================================================================
# B1 - B2: κανάλι και είσοδος 4-QAM
# ==========================================================================
def generate_channel(L, rng, size=()):
    """h_l ~ CN(0, 1/L), i.i.d.  ->  E[|H(f)|^2] = Σ_l E|h_l|^2 = 1."""
    std = np.sqrt(1.0 / (2.0 * L))
    shape = tuple(size) + (L,)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * std


def qam4(shape, rng):
    """Ισοπίθανα σύμβολα 4-QAM ±1 ± j.  E[|d|^2] = 2."""
    re = rng.choice([-1.0, 1.0], size=shape)
    im = rng.choice([-1.0, 1.0], size=shape)
    return re + 1j * im


def qam4_decide(r):
    """Απόφαση στο πλησιέστερο σημείο του αστερισμού {±1 ± j}."""
    return np.where(r.real >= 0, 1.0, -1.0) + 1j * np.where(r.imag >= 0, 1.0, -1.0)


def bit_errors(d_hat, d):
    """Gray mapping: το πραγματικό και το φανταστικό μέρος είναι ένα bit το καθένα."""
    return np.count_nonzero(d_hat.real != d.real) + np.count_nonzero(
        d_hat.imag != d.imag
    )


# ==========================================================================
# B3: διαμόρφωση / αποδιαμόρφωση OFDM
# ==========================================================================
def ofdm_modulate(d, cp_len):
    """d = IDFT(d~) (εξ. 4.3) και κυκλικό πρόθεμα L-1 στοιχείων (εξ. 4.4)."""
    x = np.fft.ifft(d, norm="ortho")
    return np.concatenate([x[-cp_len:], x]) if cp_len > 0 else x


def ofdm_channel(x_cp, h, sigma2, rng):
    """Γραμμική συνέλιξη με το κανάλι + λευκός κυκλικός Gaussian θόρυβος (εξ. 4.5)."""
    y = np.convolve(x_cp, h)
    if sigma2 > 0:
        n = (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape)) * np.sqrt(
            sigma2 / 2.0
        )
        y = y + n
    return y


def ofdm_demodulate(y, N, cp_len):
    """Αφαίρεση κυκλικού προθέματος (εξ. 4.6) και DFT (εξ. 4.11) -> y~'_n."""
    return np.fft.fft(y[cp_len : cp_len + N], norm="ortho")


def channel_frequency_response(h, N):
    """h~_n = Σ_l h_l e^{-j2πln/Nc}, εξ. (4.15) — ΜΗ κανονικοποιημένος DFT."""
    return np.fft.fft(h, N)


# ==========================================================================
# Απόδειξη του ορισμού του SNR (υποσημείωση 2 της εκφώνησης)
# ==========================================================================
def snr_definition_proof():
    print(
        """
    Απόδειξη ότι SNR_dB = 10 log10(2 / σ_n^2):

      Μετά την αφαίρεση του CP και το FFT, κάθε subcarrier δίνει
          Y[k] = H[k] d[k] + W[k],   W[k] ~ CN(0, σ_n^2)  (λευκός, ορθοκανονικός DFT)

      (α) Ισχύς συμβόλου:  d = ±1 ± j  =>  E[|d|^2] = 1 + 1 = 2
      (β) Ισχύς καναλιού:  H[k] = Σ_l h_l e^{-j2πkl/N}, h_l ανεξάρτητα, μηδενικής μέσης
                           E[|H[k]|^2] = Σ_l E[|h_l|^2] = L · (1/L) = 1
      (γ) Ισχύς θορύβου:   E[|W[k]|^2] = σ_n^2

      Άρα το μέσο λαμβανόμενο SNR ανά subcarrier είναι
          SNR = E[|H[k]|^2] E[|d[k]|^2] / E[|W[k]|^2] = 2 / σ_n^2
      και σε dB:  SNR_dB = 10 log10(2 / σ_n^2).                          ∎
    """
    )


def sigma2_from_snr_db(snr_db):
    """Αντιστροφή του παραπάνω: σ_n^2 = 2 · 10^(-SNR_dB/10)."""
    return 2.0 * 10.0 ** (-np.asarray(snr_db, dtype=float) / 10.0)


# ==========================================================================
# B4: επαλήθευση των N παράλληλων flat καναλιών (αθόρυβο)
# ==========================================================================
def experiment_flat_subchannels(N_values=(64, 128), L=4):
    print("\n[B1-B4] Αθόρυβη επαλήθευση: frequency selective -> N παράλληλα flat κανάλια")
    rng = np.random.default_rng(SEED)
    cp_len = L - 1
    for N in N_values:
        h = generate_channel(L, rng)
        d = qam4(N, rng)
        H = channel_frequency_response(h, N)

        y = ofdm_channel(ofdm_modulate(d, cp_len), h, 0.0, rng)
        Y = ofdm_demodulate(y, N, cp_len)
        err_cp = np.max(np.abs(Y - H * d))

        # ίδια μετάδοση χωρίς CP -> η γραμμική συνέλιξη δεν γίνεται κυκλική
        y0 = ofdm_channel(ofdm_modulate(d, 0), h, 0.0, rng)
        Y0 = ofdm_demodulate(y0, N, 0)
        err_nocp = np.max(np.abs(Y0 - H * d))

        print(f"    N = {N:3d}:  max|Y[k] - H[k]d[k]|  με CP = {err_cp:.3e}")
        print(f"               max|Y[k] - H[k]d[k]| χωρίς CP = {err_nocp:.3e}")
        print(f"               max|Y[k]/H[k] - d[k]|  με CP = {np.max(np.abs(Y/H - d)):.3e}")

        if N == N_values[0]:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            axes[0].plot(np.abs(H), "o-", ms=3, lw=1)
            axes[0].set_xlabel("subcarrier k")
            axes[0].set_ylabel(r"$|H[k]|$")
            axes[0].set_title(f"Channel frequency response (L={L}, N={N})")
            axes[1].plot(
                (Y / H).real, (Y / H).imag, "o", ms=5, label="με CP (ακριβές)"
            )
            axes[1].plot(
                (Y0 / H).real, (Y0 / H).imag, "x", ms=5, color="C3", label="χωρίς CP"
            )
            axes[1].set_xlabel("Re")
            axes[1].set_ylabel("Im")
            axes[1].set_title(r"$Y[k]/H[k]$ χωρίς θόρυβο")
            axes[1].axis("equal")
            axes[1].legend()
            save(fig, f"B_flat_subchannels_N{N}.png")


# ==========================================================================
# B5: θόρυβος και συσχέτιση σφαλμάτων με το |H[k]|
# ==========================================================================
def experiment_errors_vs_gain(N=128, L=4, snr_db=15.0, n_packets=4000):
    print(f"\n[B5] SNR = {snr_db} dB — πού εμφανίζονται τα σφάλματα; (N={N}, L={L})")
    rng = np.random.default_rng(SEED + 1)
    cp_len = L - 1
    sigma2 = float(sigma2_from_snr_db(snr_db))

    mags, errs = [], []
    demo = None
    for p in range(n_packets):
        h = generate_channel(L, rng)
        d = qam4(N, rng)
        H = channel_frequency_response(h, N)
        Y = ofdm_demodulate(
            ofdm_channel(ofdm_modulate(d, cp_len), h, sigma2, rng), N, cp_len
        )
        d_hat = qam4_decide(Y / H)  # one-tap ZF ανά subcarrier
        e = (d_hat.real != d.real) | (d_hat.imag != d.imag)  # σφάλμα σε αυτό το subcarrier
        mags.append(np.abs(H))
        errs.append(e)
        if demo is None and e.sum() >= 4:
            demo = (np.abs(H), e)

    mags = np.concatenate(mags)
    errs = np.concatenate(errs)

    ser = errs.mean()
    mean_ok, mean_bad = mags[~errs].mean(), mags[errs].mean()
    corr = np.corrcoef(mags, errs.astype(float))[0, 1]
    print(f"    Symbol error rate ανά subcarrier : {ser:.4f}")
    print(f"    Μέσο |H[k]| σε ΣΩΣΤΑ subcarriers : {mean_ok:.4f}")
    print(f"    Μέσο |H[k]| σε ΛΑΘΟΣ subcarriers : {mean_bad:.4f}")
    print(f"    Συντελεστής συσχέτισης |H[k]| <-> σφάλμα : {corr:+.4f}")

    # ποσοστό σφαλμάτων ανά δεκατημόριο του |H[k]|
    edges = np.quantile(mags, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(mags, edges[1:-1]), 0, 9)
    per_bin = np.array([errs[idx == b].mean() for b in range(10)])
    centers = np.array([mags[idx == b].mean() for b in range(10)])
    print("    Ποσοστό σφαλμάτων ανά δεκατημόριο του |H[k]| (από το ασθενέστερο):")
    print("      " + "  ".join(f"{v:.3f}" for v in per_bin))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    Hd, ed = demo
    axes[0].plot(Hd, "-", lw=1, color="C0", label=r"$|H[k]|$")
    axes[0].plot(np.where(ed)[0], Hd[ed], "x", ms=9, color="C3", label="σφάλμα απόφασης")
    axes[0].axhline(Hd.mean(), ls=":", color="0.4", label="μέσο $|H|$")
    axes[0].set_xlabel("subcarrier k")
    axes[0].set_ylabel(r"$|H[k]|$")
    axes[0].set_title("Ένα πακέτο: τα σφάλματα πέφτουν στα deep fades")
    axes[0].legend(fontsize=8)

    axes[1].hist(mags[~errs], bins=60, density=True, alpha=0.6, label="σωστά")
    axes[1].hist(mags[errs], bins=60, density=True, alpha=0.6, color="C3", label="λάθος")
    axes[1].set_xlabel(r"$|H[k]|$")
    axes[1].set_ylabel("πυκνότητα")
    axes[1].set_title(f"Κατανομή του $|H[k]|$ (συσχ. = {corr:+.3f})")
    axes[1].legend()

    # τα δεκατημόρια χωρίς κανένα σφάλμα δεν σχεδιάζονται (δεν είναι «floor»)
    axes[2].semilogy(centers, np.where(per_bin > 0, per_bin, np.nan), "o-")
    axes[2].set_xlabel(r"μέσο $|H[k]|$ του δεκατημορίου")
    axes[2].set_ylabel("ποσοστό σφαλμάτων")
    axes[2].set_title("Σφάλματα ανά δεκατημόριο του channel gain\n(κενά = κανένα σφάλμα)")

    fig.suptitle(f"Θέσεις σφαλμάτων vs $|H[k]|$ — SNR = {snr_db:.0f} dB, N={N}, L={L}")
    save(fig, f"B_errors_vs_gain_SNR{int(snr_db)}.png")
    return corr, mean_ok, mean_bad


# ==========================================================================
# B6-B8: διαφοροποίηση με επανάληψη στις συχνότητες
# ==========================================================================
def repetition_ber(
    order,
    snr_db_list,
    rng,
    normalize_power=False,
    target_errors=500,
    max_bits=None,
    batch=200_000,
):
    """BER 4-QAM με επανάληψη του ίδιου συμβόλου σε `order` subcarriers.

    Τα subcarriers έχουν ανεξάρτητες υλοποιήσεις καναλιού H_m ~ CN(0,1) και
    συνδυάζονται με MRC:  r = h^H y / ||h||  (γενίκευση της εξ. 3.11).

    normalize_power=False : κάθε αντίγραφο στέλνεται με πλήρη ισχύ (SNR = μέσο
                            λαμβανόμενο SNR ανά κλάδο) — όπως στην εκφώνηση.
    normalize_power=True  : η συνολική ισχύς μοιράζεται στα `order` αντίγραφα,
                            οπότε απομονώνεται το κέρδος διαφοροποίησης.
    """
    # Όσο μεγαλύτερη η τάξη, τόσο χαμηλότερο το BER στα υψηλά SNR -> χρειάζονται
    # περισσότερα δείγματα για να παραμείνει μετρήσιμο.
    if max_bits is None:
        max_bits = 20_000_000 * order**2

    ber = np.empty(len(snr_db_list))
    counts = np.zeros(len(snr_db_list), dtype=int)
    for i, snr_db in enumerate(snr_db_list):
        sigma2 = float(sigma2_from_snr_db(snr_db))
        scale = 1.0 / np.sqrt(order) if normalize_power else 1.0
        errors = 0
        total = 0
        t0 = time.time()
        while total < max_bits and errors < target_errors:
            B = batch
            h = (
                rng.standard_normal((B, order)) + 1j * rng.standard_normal((B, order))
            ) / np.sqrt(2.0)  # CN(0,1) ανά κλάδο
            d = qam4(B, rng)
            w = (
                rng.standard_normal((B, order)) + 1j * rng.standard_normal((B, order))
            ) * np.sqrt(sigma2 / 2.0)
            y = h * (scale * d)[:, None] + w
            r = np.sum(np.conj(h) * y, axis=1) / np.linalg.norm(h, axis=1)
            d_hat = qam4_decide(r)
            errors += bit_errors(d_hat, d)
            total += 2 * B  # 2 bits ανά σύμβολο 4-QAM
        ber[i] = errors / total
        counts[i] = errors
        flag = "  (<- κάτω από το όριο μέτρησης)" if errors < 50 else ""
        print(
            f"      SNR = {snr_db:5.1f} dB   BER = {ber[i]:.3e}"
            f"   ({errors} σφάλματα / {total} bits, {time.time()-t0:.1f}s){flag}"
        )
    return ber, counts


SNR_B = np.arange(0, 21, 4)  # ζητούμενο B7: SNR_dB = 0:4:20


# ==========================================================================
# §4.2.1 — ανεξαρτησία φορέων:  E[H_i H_k^*] = σ_h^2 Σ_l e^{-j2πl(i-k)/N}
# ==========================================================================
def experiment_subcarrier_correlation(N=128, L=4, n_real=200_000):
    """Αριθμητική επαλήθευση των εξ. (4.20)-(4.21) του Κεφαλαίου OFDM.

    Η θεωρία δίνει
        E[H_i H_k^*] = σ_h^2 Σ_{l=0}^{L-1} e^{-j2πl(i-k)/N},   σ_h^2 = 1/L,
    και μηδενίζεται όταν i - k = c·N/L, c = 1,...,L-1. Δηλαδή subcarriers που
    απέχουν N/L είναι ασυσχέτιστοι — και, επειδή τα H_i είναι από κοινού
    κυκλικά Gaussian, ασυσχέτιστοι σημαίνει ανεξάρτητοι. Αυτό ακριβώς είναι που
    δικαιολογεί την «επανάληψη στις συχνότητες» με βήμα N/M του εδαφίου 4.2.
    """
    print(f"\n[§4.2.1] Ανεξαρτησία φορέων — E[H_i H_k*] (N={N}, L={L})")
    rng = np.random.default_rng(SEED + 3)
    lags = [1, 2, N // (2 * L), N // L, 2 * N // L, 3 * N // L]

    # Συσσώρευση σε παρτίδες: το (n_real x N) ταυτόχρονα δεν χωράει στη μνήμη.
    acc = {d: 0.0 + 0.0j for d in lags}
    cnt = {d: 0 for d in lags}
    batch = 20_000
    done = 0
    while done < n_real:
        b = min(batch, n_real - done)
        h = generate_channel(L, rng, size=(b,))
        H = np.fft.fft(h, N, axis=1)  # (b, N) — μη κανονικοποιημένος DFT, εξ. (4.15)
        for d in lags:
            acc[d] += np.sum(H[:, d:] * np.conj(H[:, :-d]))
            cnt[d] += b * (N - d)
        done += b

    print("    Δ = i-k |  θεωρία (4.20)  |  προσομοίωση  |  ασυσχέτιστοι;")
    ok_all = True
    for d in lags:
        theory = (1.0 / L) * np.sum(np.exp(-2j * np.pi * np.arange(L) * d / N))
        # i - k = +d  ->  E[H_{k+d} H_k^*]
        emp = acc[d] / cnt[d]
        zero_theory = abs(theory) < 1e-12
        flag = "ΝΑΙ (εξ. 4.21)" if zero_theory else "όχι"
        print(
            f"    {d:7d} |  {theory.real:+.4f}{theory.imag:+.4f}j  |"
            f"  {emp.real:+.4f}{emp.imag:+.4f}j  |  {flag}"
        )
        if abs(emp - theory) > 0.02:
            ok_all = False
    print(
        f"    Μέγιστη απόκλιση προσομοίωσης-θεωρίας εντός στατιστικού σφάλματος: "
        f"{'ΝΑΙ' if ok_all else 'ΟΧΙ'}"
    )
    print(
        f"    -> Οι φορείς με απόσταση N/L = {N // L} είναι ανεξάρτητοι, άρα η"
        f" επανάληψη σε M ≤ L = {L}"
    )
    print(f"       subcarriers με βήμα N/M δίνει διαφοροποίηση τάξης M (εδάφιο 4.2).")
    return ok_all


# ==========================================================================
# §4.2 — επανάληψη στις συχνότητες μέσα σε ΠΡΑΓΜΑΤΙΚΟ σύμβολο OFDM
# ==========================================================================
def ofdm_repetition_ber(
    order,
    snr_db_list,
    rng,
    Nc=128,
    L=4,
    target_errors=500,
    max_bits=None,
    packets_per_batch=2000,
):
    """BER με επανάληψη κάθε συμβόλου σε `order` subcarriers που απέχουν Nc/order.

    Σε αντίθεση με την `repetition_ber`, εδώ ΔΕΝ υποθέτουμε ανεξάρτητους κλάδους:
    στήνουμε ολόκληρη την αλυσίδα OFDM (IDFT -> CP -> γραμμική συνέλιξη με το
    κανάλι -> θόρυβος -> αφαίρεση CP -> DFT) και η ανεξαρτησία των αντιγράφων
    προκύπτει *από το ίδιο το κανάλι*, μέσω της εξ. (4.21).

    Ο συνδυασμός των αντιγράφων γίνεται με MRC πάνω στα Y[k] = H[k] d[k] + W[k]
    της εξ. (4.14).
    """
    if max_bits is None:
        max_bits = 20_000_000 * order**2
    spacing = Nc // order
    cp_len = L - 1
    # idx[g] = οι `order` φορείς πάνω στους οποίους κάθεται το σύμβολο g
    idx = np.arange(spacing)[:, None] + spacing * np.arange(order)[None, :]
    G = spacing  # σύμβολα πληροφορίας ανά σύμβολο OFDM

    ber = np.empty(len(snr_db_list))
    counts = np.zeros(len(snr_db_list), dtype=int)
    for i, snr_db in enumerate(snr_db_list):
        sigma2 = float(sigma2_from_snr_db(snr_db))
        errors = 0
        total = 0
        t0 = time.time()
        while total < max_bits and errors < target_errors:
            P = packets_per_batch
            h = generate_channel(L, rng, size=(P,))  # (P, L)
            H = np.fft.fft(h, Nc, axis=1)  # (P, Nc), εξ. (4.15)

            d = qam4((P, G), rng)
            D = np.zeros((P, Nc), dtype=complex)
            D[:, idx] = d[:, :, None]  # επανάληψη με βήμα Nc/order

            x = np.fft.ifft(D, axis=1, norm="ortho")  # εξ. (4.3)
            x_cp = np.concatenate([x[:, Nc - cp_len :], x], axis=1)  # εξ. (4.4)

            T = Nc + cp_len + L - 1
            y = np.zeros((P, T), dtype=complex)
            for l in range(L):  # εξ. (4.5): γραμμική συνέλιξη
                y[:, l : l + Nc + cp_len] += h[:, l : l + 1] * x_cp
            y += (
                rng.standard_normal((P, T)) + 1j * rng.standard_normal((P, T))
            ) * np.sqrt(sigma2 / 2.0)

            yp = y[:, cp_len : cp_len + Nc]  # εξ. (4.6)
            Y = np.fft.fft(yp, axis=1, norm="ortho")  # εξ. (4.14)

            Yg, Hg = Y[:, idx], H[:, idx]  # (P, G, order)
            r = np.sum(np.conj(Hg) * Yg, axis=2) / np.linalg.norm(Hg, axis=2)
            errors += bit_errors(qam4_decide(r), d)
            total += 2 * P * G
        ber[i] = errors / total
        counts[i] = errors
        flag = "  (<- κάτω από το όριο μέτρησης)" if errors < 50 else ""
        print(
            f"      SNR = {snr_db:5.1f} dB   BER = {ber[i]:.3e}"
            f"   ({errors} σφάλματα / {total} bits, {time.time()-t0:.1f}s){flag}"
        )
    return ber, counts


def experiment_ofdm_repetition(orders=(1, 2, 4), Nc=128, L=4):
    """Σύγκριση πραγματικού OFDM (βήμα Nc/M) με το εξιδανικευμένο μοντέλο."""
    print(
        f"\n[§4.2] Επανάληψη σε ΠΡΑΓΜΑΤΙΚΟ OFDM, βήμα Nc/M (Nc={Nc}, L={L}, CP={L-1})"
    )
    rng = np.random.default_rng(SEED + 4)
    snr_lin = db2lin(SNR_B)
    res, nerr, est, est_th = {}, {}, {}, {}
    for order in orders:
        print(f"    M = {order} αντίγραφα, βήμα Nc/M = {Nc // order} φορείς:")
        res[order], nerr[order] = ofdm_repetition_ber(order, SNR_B, rng, Nc=Nc, L=L)
        est[order] = estimate_diversity_order(SNR_B, res[order], counts=nerr[order])
        th = mrc_ber(snr_lin / 2.0, order)
        window = np.where((res[order] > 0) & (nerr[order] >= 50))[0][-3:]
        est_th[order] = estimate_diversity_order(SNR_B[window], th[window])
        print(
            f"      -> κλίση προσομοίωσης = {est[order]:.2f}  |  "
            f"κλίση θεωρίας στο ίδιο παράθυρο "
            f"({SNR_B[window[0]]:.0f}-{SNR_B[window[-1]]:.0f} dB)"
            f" = {est_th[order]:.2f}  |  ασυμπτωτική τιμή = {order}"
        )

    fig, ax = plt.subplots()
    for j, order in enumerate(orders):
        ber = np.where(res[order] > 0, res[order], np.nan)
        ax.semilogy(
            SNR_B, ber, "o-", color=f"C{j}", lw=1.8, label=f"OFDM Nc/M, M={order}"
        )
        ax.semilogy(
            SNR_B,
            mrc_ber(snr_lin / 2.0, order),
            "--",
            color=f"C{j}",
            lw=1.0,
            label=f"θεωρία MRC, M={order}",
        )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title(
        f"Επανάληψη σε πραγματικό OFDM με βήμα $N_c/M$ (Nc={Nc}, L={L})\n"
        "η ανεξαρτησία των αντιγράφων προκύπτει από την εξ. (4.21)"
    )
    ax.legend(ncol=2, loc="lower left")
    save(fig, f"B_ofdm_repetition_Nc{Nc}.png")
    return est


def experiment_repetition_diversity(orders=(1, 2, 4), normalize_power=False):
    tag = "norm" if normalize_power else "raw"
    label = (
        "με κανονικοποίηση συνολικής ισχύος"
        if normalize_power
        else "χωρίς κανονικοποίηση ισχύος"
    )
    print(f"\n[B6-B8] Επανάληψη στις συχνότητες, τάξεις {orders} ({label})")
    rng = np.random.default_rng(SEED + 2)
    snr_lin = db2lin(SNR_B)
    results, nerrs, est, est_th = {}, {}, {}, {}
    for order in orders:
        print(f"    τάξη διαφοροποίησης {order}:")
        results[order], nerrs[order] = repetition_ber(
            order, SNR_B, rng, normalize_power=normalize_power
        )
        est[order] = estimate_diversity_order(
            SNR_B, results[order], counts=nerrs[order]
        )

        # Η θεωρητική καμπύλη, με κλίση μετρημένη στο ΙΔΙΟ παράθυρο SNR: η
        # σύγκριση αυτή είναι η ουσιαστική επαλήθευση, γιατί η τάξη
        # διαφοροποίησης είναι ασυμπτωτική έννοια και η κλίση την προσεγγίζει
        # μόνο σε αρκετά υψηλό SNR.
        gbar = snr_lin / 2.0 / (order if normalize_power else 1.0)
        th = mrc_ber(gbar, order)
        window = np.where((results[order] > 0) & (nerrs[order] >= 50))[0][-3:]
        est_th[order] = estimate_diversity_order(SNR_B[window], th[window])
        rng_lo, rng_hi = SNR_B[window[0]], SNR_B[window[-1]]
        print(
            f"      -> κλίση προσομοίωσης = {est[order]:.2f}  |  "
            f"κλίση θεωρίας στο ίδιο παράθυρο ({rng_lo:.0f}-{rng_hi:.0f} dB)"
            f" = {est_th[order]:.2f}  |  ασυμπτωτική τιμή = {order}"
        )

    # --- BER vs SNR (ζητούμενο B7) ---
    fig, ax = plt.subplots()
    for j, order in enumerate(orders):
        ber = np.where(results[order] > 0, results[order], np.nan)
        ax.semilogy(SNR_B, ber, "o-", color=f"C{j}", lw=1.8, label=f"προσομοίωση, M={order}")
        # θεωρία: μέσο SNR/κλάδο ανά bit = (SNR_lin/2) · scale^2
        gbar = snr_lin / 2.0 / (order if normalize_power else 1.0)
        ax.semilogy(SNR_B, mrc_ber(gbar, order), "--", color=f"C{j}", lw=1.0,
                    label=f"θεωρία MRC, M={order}")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title(f"Επανάληψη στις συχνότητες με MRC ({label})")
    ax.legend(ncol=2, loc="lower left")
    save(fig, f"B_repetition_ber_{tag}.png")

    # --- log10(SNR) vs log10(BER) (ζητούμενο B8) ---
    fig, ax = plt.subplots()
    x = np.log10(snr_lin)
    for j, order in enumerate(orders):
        ber = results[order]
        ok = ber > 0
        ax.plot(x[ok], np.log10(ber[ok]), "o-", color=f"C{j}", lw=1.8,
                label=f"M={order} (κλίση προσομ. ≈ {est[order]:.2f}, θεωρίας {est_th[order]:.2f})")
        # Ευθεία αναφοράς κλίσης -order, αγκυρωμένη στο τελευταίο ΑΞΙΟΠΙΣΤΟ σημείο:
        # εκεί ισχύει η ασυμπτωτική συμπεριφορά, άρα εκεί έχει νόημα η σύγκριση.
        a = np.where(ok & (nerrs[order] >= 50))[0][-1]
        ax.plot(x, np.log10(ber[a]) - order * (x - x[a]), "--", lw=1.1,
                color="0.4" if j == 0 else f"C{j}", alpha=0.8,
                label=rf"ευθεία κλίσης $-{order}$")
    ax.set_xlabel(r"$\log_{10}(\mathrm{SNR})$")
    ax.set_ylabel(r"$\log_{10}(\mathrm{BER})$")
    ax.set_title(f"Επιβεβαίωση τάξης διαφοροποίησης από την κλίση ({label})")
    ax.legend(ncol=2, fontsize=8)
    save(fig, f"B_repetition_loglog_{tag}.png")
    return est


def main():
    print("=" * 72)
    print("Άσκηση 5 (Bonus) — Μέρος Β: OFDM & διαφοροποίηση συχνότητας")
    print("=" * 72)
    snr_definition_proof()
    experiment_flat_subchannels(N_values=(64, 128), L=4)
    experiment_errors_vs_gain(N=128, L=4, snr_db=15.0)
    experiment_subcarrier_correlation(N=128, L=4)
    experiment_repetition_diversity(orders=(1, 2, 4), normalize_power=False)
    experiment_repetition_diversity(orders=(1, 2, 4), normalize_power=True)
    experiment_ofdm_repetition(orders=(1, 2, 4), Nc=128, L=4)
    print(f"\nΤα σχήματα αποθηκεύτηκαν στο {FIG_DIR}")


if __name__ == "__main__":
    main()
