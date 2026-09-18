"""Άσκηση 5 (Bonus) — Μέρος Α: uplink DS-CDMA με δέκτη Rake.

Μοντέλο (σύμφωνα με την εκφώνηση):
  * K σύγχρονοι χρήστες, πακέτα M δυαδικών ισοπίθανων συμβόλων από {+1,-1}
  * κώδικας εξάπλωσης c_k = sign(randn(N,1))/sqrt(N), σταθερός σε όλο το πείραμα
  * κανάλι h_k ~ CN(0, (1/L) I_L), σταθερό ανά πακέτο, ανεξάρτητο από πακέτο σε πακέτο
  * λευκός κυκλικός Gaussian θόρυβος διασποράς σ_N^2 στην έξοδο του καναλιού
  * SNR_dB = 10 log10(1/σ_N^2)

Αντιστοιχία με τη θεωρία (Chapter_Frequency_Diversity_1.pdf, εδάφιο 4.2):
  εξ. (4.9)       r_k = c_k^T y = ||c_k||^2 s_k + w'_k   (AWGN, ορθογώνιοι κώδικες)
  εξ. (4.10)      y_m = Σ_l h_l x_{m-l} + w_m,  x = kron(s, c)
  εξ. (4.11)      προσεγγιστικό μοντέλο που αγνοεί την ISI των γειτονικών συμβόλων
  εξ. (4.12)      c^(l) := [0_l ; c ; 0_{L-1-l}]  (κώδικας ολισθημένος κατά l)
  εξ. (4.13)      y = Σ_l h_l s c^(l) + w
  εξ. (4.14)      ||c|| = 1 και c^(l)T c^(m) ≈ 0 για l ≠ m
  εξ. (4.15)      έξοδος finger l:  (|h_l|^2/||h||) s + w'_l
  εξ. (4.16)      έξοδος Rake:      r = ||h|| s + w'',  w'' ~ CN(0, N_0)

Δύο σημεία όπου η προσομοίωση είναι ΑΚΡΙΒΕΣΤΕΡΗ από το μοντέλο του κεφαλαίου:
η μετάδοση γίνεται με πλήρη γραμμική συνέλιξη, άρα ΔΕΝ γίνεται η προσέγγιση
(4.11) «ISI = 0», και η (4.14) δεν επιβάλλεται αλλά ελέγχεται αριθμητικά στη
experiment_code_assumptions(). Το ότι η καμπύλη BER πέφτει παρ' όλα αυτά πάνω
στη θεωρητική MRC δείχνει ότι οι προσεγγίσεις του κεφαλαίου είναι θεμιτές.

Ζητούμενα που καλύπτονται:
  A1  Rake με L fingers (MRC), K=1, BER για SNR_dB = 0:2:20 + f_i = SNR^-i
  A2  Rake με 1 finger (selection combining), K=1, BER + f_i = SNR^-i
  A3  K=2, Rake με L fingers, BER για SNR_dB = 0:2:16, σύγκριση με K=1 και ως προς N
  A4  K = 1..5 χρήστες

Εκτέλεση:  python cdma_sim.py
"""

import time

import numpy as np

from common import (
    FIG_DIR,
    db2lin,
    estimate_diversity_order,
    mrc_ber,
    mrc_ber_asymptotic,
    plot_reference_slopes,
    plt,
    save,
    sc_ber,
)

SEED = 2021030061  # ΑΜ, για αναπαραγωγιμότητα


# ==========================================================================
# 1.1  Κώδικες εξάπλωσης
# ==========================================================================
def generate_codes(K, N, rng):
    """c_k = sign(randn(N,1))/sqrt(N).  Επιστρέφει πίνακα (K, N), ||c_k|| = 1."""
    return np.sign(rng.standard_normal((K, N))) / np.sqrt(N)


# ==========================================================================
# 1.2  Κανάλι
# ==========================================================================
def generate_channels(P, K, L, rng):
    """h_k ~ CN(0, (1/L) I_L) — ανεξάρτητο για καθένα από τα P πακέτα.

    Επιστρέφει (P, K, L) με E[||h_k||^2] = 1.
    """
    std = np.sqrt(1.0 / (2.0 * L))
    return (rng.standard_normal((P, K, L)) + 1j * rng.standard_normal((P, K, L))) * std


# ==========================================================================
# 1.3  Μετάδοση σε επίπεδο chip + κανάλι (εξ. 4.10)
# ==========================================================================
def channel_output(S, C, H, sigma2, rng):
    """y_m = Σ_k Σ_l h_{k,l} x_{k,m-l} + w_m, για P πακέτα ταυτόχρονα.

    S : (P, K, M) σύμβολα ±1
    C : (K, N) κώδικες
    H : (P, K, L) κρουστικές αποκρίσεις
    -> y : (P, N*M + L - 1)

    Η ροή chip κάθε χρήστη είναι x_k = kron(s_k, c_k) και περνάει από γραμμική
    συνέλιξη με το h_k· έτσι η ISI μεταξύ διαδοχικών συμβόλων μπαίνει αυτόματα
    (δεν γίνεται η προσέγγιση «ISI = 0»).
    """
    P, K, M = S.shape
    N = C.shape[1]
    L = H.shape[2]

    # kron(s_k, c_k) για κάθε πακέτο/χρήστη -> (P, K, M*N)
    X = (S[:, :, :, None] * C[None, :, None, :]).reshape(P, K, M * N)

    T = M * N + L - 1
    y = np.zeros((P, T), dtype=complex)
    for l in range(L):  # συνέλιξη: κάθε tap είναι μια ολίσθηση κατά l chips
        y[:, l : l + M * N] += np.einsum("pk,pkt->pt", H[:, :, l], X)

    w = (rng.standard_normal((P, T)) + 1j * rng.standard_normal((P, T))) * np.sqrt(
        sigma2 / 2.0
    )
    return y + w


# ==========================================================================
# 1.4  Μπροστινό τμήμα του Rake: οι L συσχετιστές c^(l)T y  (εξ. 4.12-4.13)
# ==========================================================================
def rake_correlate(y, c1, M, L):
    """z_{j,l} = c_1^T y[jN+l : jN+l+N] — το finger l του συμβόλου j.

    Επιστρέφει (P, M, L).  Το παράθυρο του συμβόλου j ξεκινά στο chip jN και
    το finger l το «κοιτάζει» καθυστερημένο κατά l.
    """
    P = y.shape[0]
    N = c1.size
    Z = np.empty((P, M, L), dtype=complex)
    for l in range(L):
        # (P, M*N) -> (P, M, N): strided view, χωρίς αντιγραφή
        Z[:, :, l] = y[:, l : l + M * N].reshape(P, M, N) @ c1
    return Z


# ==========================================================================
# 1.4/1.5  Συνδυαστές
# ==========================================================================
def combine_mrc(Z, h1):
    """Πλήρης Rake με L fingers (Σχήμα 4.3 του Κεφαλαίου).

    Κάθε finger l δίνει (εξ. 4.15)   (|h_l|^2/||h||) s + w'_l,
    και το άθροισμά τους (εξ. 4.16)  r = ||h|| s + w'',  w'' ~ CN(0, N_0),
    δηλαδή ακριβώς r = h_1^H z / ||h_1||. Αν τα h_l είναι i.i.d., ο δέκτης
    επιτυγχάνει διαφοροποίηση τάξης L.
    """
    r = np.einsum("pjl,pl->pj", Z, np.conj(h1))
    r /= np.linalg.norm(h1, axis=1, keepdims=True)
    return np.where(r.real >= 0.0, 1.0, -1.0)


def combine_selection(Z, h1):
    """Rake με 1 finger (selection combining): μόνο ο ισχυρότερος συντελεστής."""
    p = np.arange(h1.shape[0])
    lstar = np.argmax(np.abs(h1), axis=1)
    z = Z[p, :, lstar]  # (P, M)
    g = h1[p, lstar]
    r = np.conj(g)[:, None] * z / np.abs(g)[:, None]
    return np.where(r.real >= 0.0, 1.0, -1.0)


# ==========================================================================
# 1.6  Monte Carlo βρόχος BER
# ==========================================================================
def simulate_ber(
    snr_db_list,
    C,
    M,
    L,
    combiner,
    rng,
    target_errors=300,
    max_symbols=3_000_000,
    packets_per_batch=250,
):
    """BER του χρήστη 1 για κάθε SNR.

    Adaptive stopping: σταματάμε όταν μαζευτούν `target_errors` σφάλματα ή
    ξεπεραστούν τα `max_symbols` σύμβολα — αλλιώς τα υψηλά SNR θα ήταν απαγορευτικά.
    Σημεία όπου δεν καταγράφηκε κανένα σφάλμα επιστρέφονται ως BER = 0 και
    αγνοούνται στη συνέχεια (δεν είναι μετρήσιμα με αυτόν τον αριθμό δειγμάτων).
    """
    K = C.shape[0]
    ber = np.empty(len(snr_db_list))
    counts = np.zeros(len(snr_db_list), dtype=int)
    for i, snr_db in enumerate(snr_db_list):
        sigma2 = 10.0 ** (-snr_db / 10.0)
        errors = 0
        total = 0
        t0 = time.time()
        while total < max_symbols and errors < target_errors:
            P = packets_per_batch
            S = rng.choice([-1.0, 1.0], size=(P, K, M))
            H = generate_channels(P, K, L, rng)
            y = channel_output(S, C, H, sigma2, rng)
            Z = rake_correlate(y, C[0], M, L)
            s_hat = combiner(Z, H[:, 0, :])
            errors += int(np.count_nonzero(s_hat != S[:, 0, :]))
            total += P * M
        ber[i] = errors / total
        counts[i] = errors
        flag = "  (<- κάτω από το όριο μέτρησης)" if errors < 50 else ""
        print(
            f"      SNR = {snr_db:5.1f} dB   BER = {ber[i]:.3e}"
            f"   ({errors} σφάλματα / {total} σύμβολα, {time.time()-t0:.1f}s){flag}"
        )
    return ber, counts


def _m(ber):
    """Μάσκα των μη μετρήσιμων σημείων (BER = 0) ώστε να μη σχεδιάζονται."""
    ber = np.asarray(ber, dtype=float).copy()
    ber[ber <= 0] = np.nan
    return ber


# ==========================================================================
# Πειράματα
# ==========================================================================
M_SYMBOLS = 100
SNR_A = np.arange(0, 21, 2)  # ζητούμενα 1 και 2 (K = 1)
SNR_MU = np.arange(0, 17, 2)  # πολυχρηστικά ζητούμενα


def experiment_code_assumptions(N_values=(32, 64, 128), L=5):
    """Επαλήθευση των υποθέσεων (4.14) του Κεφαλαίου «Διαφοροποίηση στις συχνότητες».

    Η θεωρία του Rake (εξ. 4.13-4.16) στηρίζεται σε δύο παραδοχές:
      (i)  ||c|| = 1                     -> ισχύει εξ ορισμού του c = sign(randn)/sqrt(N)
      (ii) c^(l)T c^(m) ≈ 0 για l ≠ m    -> εξ. (4.14), ΜΟΝΟ προσεγγιστικά

    όπου c^(l) := [0_l ; c ; 0_{L-1-l}] είναι ο κώδικας ολισθημένος κατά l chips
    (εξ. 4.12). Εδώ μετράμε πόσο καλά ισχύει η (ii) και δείχνουμε ότι η απόκλιση
    είναι τάξης 1/sqrt(N) — δηλαδή το processing gain N είναι αυτό που κάνει την
    προσέγγιση δουλευτική. Με το ίδιο σκεπτικό, η διασυσχέτιση c_1^T c_2 μεταξύ
    δύο χρηστών είναι κι αυτή ~1/sqrt(N) και όχι μηδέν, γεγονός που εξηγεί την
    multi-user interference του ζητουμένου A3.
    """
    print(f"\n[Υποθέσεις 4.14] Ορθογωνιότητα ολισθημένων κωδίκων (L = {L})")
    print("      N   | rms |c^(l)T c^(m)|, l≠m | rms |c_1^T c_2| |  1/sqrt(N)")
    rng = np.random.default_rng(SEED + 11)
    n_codes = 4000
    for N in N_values:
        # (i) αυτοσυσχέτιση του ίδιου κώδικα σε ολισθήσεις l ≠ m — εξ. (4.12), (4.14)
        codes = generate_codes(n_codes, N, rng)
        off_sq = [
            (codes[:, lag:] * codes[:, : N - lag]).sum(axis=1) ** 2
            for lag in range(1, L)
        ]
        off = np.sqrt(np.mean(np.concatenate(off_sq)))

        # (ii) διασυσχέτιση κωδίκων δύο διαφορετικών χρηστών — η πηγή της MUI
        pairs = generate_codes(n_codes, N, rng)
        cross = np.sqrt(np.mean((pairs[::2] * pairs[1::2]).sum(axis=1) ** 2))

        print(
            f"    {N:4d}   |         {off:.4f}          |     {cross:.4f}     |"
            f"   {1/np.sqrt(N):.4f}"
        )
    print("    -> και οι δύο ποσότητες κλιμακώνονται ως 1/sqrt(N): η (4.14) είναι")
    print("       τόσο καλύτερη όσο μεγαλώνει το processing gain N.")


def experiment_single_user(N=64, L_values=(3, 5)):
    """A1 + A2: K = 1, πλήρης Rake και Rake 1 finger."""
    print("\n[A1/A2] K = 1 — πλήρης Rake (L fingers) vs Rake 1 finger, N =", N)
    results = {}
    for L in L_values:
        rng = np.random.default_rng(SEED + L)
        C = generate_codes(1, N, rng)  # ο κώδικας μένει σταθερός σε όλο το πείραμα
        budget = dict(target_errors=300, max_symbols=10_000_000)
        print(f"    L = {L}, πλήρης Rake (MRC):")
        ber_mrc, n_mrc = simulate_ber(SNR_A, C, M_SYMBOLS, L, combine_mrc, rng, **budget)
        print(f"    L = {L}, Rake 1 finger (selection combining):")
        ber_sc, n_sc = simulate_ber(
            SNR_A, C, M_SYMBOLS, L, combine_selection, rng, **budget
        )
        results[L] = (ber_mrc, ber_sc)

        # μέσο SNR ανά κλάδο: γ_l = |h_l|^2 / σ_N^2, E|h_l|^2 = 1/L
        gbar = db2lin(SNR_A) / L

        for tag, ber, nerr, theory, title in [
            (
                "mrc",
                ber_mrc,
                n_mrc,
                mrc_ber(gbar, L),
                f"Rake με L={L} fingers (MRC), K=1, N={N}",
            ),
            (
                "sc",
                ber_sc,
                n_sc,
                sc_ber(gbar, L),
                f"Rake με 1 finger (selection combining), L={L}, K=1, N={N}",
            ),
        ]:
            fig, ax = plt.subplots()
            ax.semilogy(SNR_A, _m(ber), "o-", lw=1.8, color="C0", label="Προσομοίωση")
            ax.semilogy(SNR_A, theory, "-", lw=1.2, color="C3", label="Θεωρία (Rayleigh)")
            if tag == "mrc":
                ax.semilogy(
                    SNR_A,
                    mrc_ber_asymptotic(gbar, L),
                    ":",
                    lw=1.2,
                    color="C1",
                    label=r"Ασυμπτωτικό $\binom{2L-1}{L}/(4\bar\gamma)^L$",
                )
            plot_reference_slopes(ax, SNR_A)
            d = estimate_diversity_order(SNR_A, ber, counts=nerr)
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel("BER")
            ax.set_title(f"{title}\nεκτιμώμενη τάξη diversity ≈ {d:.2f}")
            ax.set_ylim(1e-7, 1.0)
            ax.legend(loc="lower left")
            save(fig, f"A_{tag}_L{L}_N{N}.png")
            print(f"      -> εκτιμώμενη τάξη diversity ({tag.upper()}) = {d:.2f}")

    # Συγκριτικό σχήμα MRC vs SC
    fig, ax = plt.subplots()
    for j, L in enumerate(L_values):
        ber_mrc, ber_sc = results[L]
        ax.semilogy(SNR_A, _m(ber_mrc), "o-", color=f"C{j}", label=f"MRC, L={L} fingers")
        ax.semilogy(SNR_A, _m(ber_sc), "s--", color=f"C{j}", label=f"SC, 1 finger (L={L})")
    plot_reference_slopes(ax, SNR_A)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title(f"Πλήρης Rake vs selection combining, K=1, N={N}")
    ax.set_ylim(1e-7, 1.0)
    ax.legend(loc="lower left")
    save(fig, f"A_mrc_vs_sc_N{N}.png")
    return results


def experiment_two_users(N_values=(32, 64, 128), L=3):
    """A3: εισαγωγή δεύτερου χρήστη — επίδραση του N."""
    print(f"\n[A3] K = 2 vs K = 1 — πλήρης Rake, L = {L}")
    fig, ax = plt.subplots()
    orders = {}
    for j, N in enumerate(N_values):
        rng = np.random.default_rng(SEED + 100 * N)
        C1 = generate_codes(1, N, rng)
        C2 = generate_codes(2, N, rng)
        print(f"    N = {N}, K = 1:")
        ber1, n1 = simulate_ber(SNR_MU, C1, M_SYMBOLS, L, combine_mrc, rng)
        print(f"    N = {N}, K = 2:")
        ber2, n2 = simulate_ber(SNR_MU, C2, M_SYMBOLS, L, combine_mrc, rng)
        orders[N] = (
            estimate_diversity_order(SNR_MU, ber1, counts=n1),
            estimate_diversity_order(SNR_MU, ber2, counts=n2),
        )
        ax.semilogy(SNR_MU, _m(ber1), "o--", color=f"C{j}", alpha=0.55, label=f"K=1, N={N}")
        ax.semilogy(SNR_MU, _m(ber2), "s-", color=f"C{j}", lw=1.8, label=f"K=2, N={N}")
        print(f"      -> τάξη διαφ.: K=1 {orders[N][0]:.2f} | K=2 {orders[N][1]:.2f}")

    plot_reference_slopes(ax, SNR_MU)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER (χρήστης 1)")
    ax.set_title(f"Επίδραση δεύτερου χρήστη για διάφορα processing gains N (Rake L={L} fingers)")
    ax.set_ylim(1e-6, 1.0)
    ax.legend(loc="lower left", ncol=2)
    save(fig, f"A_K2_vs_K1_L{L}.png")
    return orders


def experiment_many_users(K_values=(1, 2, 3, 4, 5), N=64, L=3):
    """A4: περισσότεροι χρήστες — αύξηση της multi-user παρεμβολής."""
    print(f"\n[A4] K = {K_values} — πλήρης Rake, N = {N}, L = {L}")
    rng = np.random.default_rng(SEED + 7)
    C_all = generate_codes(max(K_values), N, rng)  # ίδιοι κώδικες, μεταβλητό πλήθος
    fig, ax = plt.subplots()
    orders = {}
    for j, K in enumerate(K_values):
        print(f"    K = {K}:")
        ber, nerr = simulate_ber(SNR_MU, C_all[:K], M_SYMBOLS, L, combine_mrc, rng)
        orders[K] = estimate_diversity_order(SNR_MU, ber, counts=nerr)
        ax.semilogy(SNR_MU, _m(ber), "o-", color=f"C{j}", label=f"K = {K}")
        print(f"      -> τάξη διαφοροποίησης = {orders[K]:.2f}")
    plot_reference_slopes(ax, SNR_MU)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER (χρήστης 1)")
    ax.set_title(f"BER χρήστη 1 vs πλήθος ενεργών χρηστών K (N={N}, L={L}, Rake L fingers)")
    ax.set_ylim(1e-6, 1.0)
    ax.legend(loc="lower left", ncol=2)
    save(fig, f"A_manyusers_N{N}_L{L}.png")
    return orders


def main():
    print("=" * 72)
    print("Άσκηση 5 (Bonus) — Μέρος Α: DS-CDMA & δέκτης Rake")
    print("=" * 72)
    experiment_code_assumptions(N_values=(32, 64, 128), L=5)
    experiment_single_user(N=64, L_values=(3, 5))
    experiment_two_users(N_values=(32, 64, 128), L=3)
    experiment_many_users(K_values=(1, 2, 3, 4, 5), N=64, L=3)
    print(f"\nΤα σχήματα αποθηκεύτηκαν στο {FIG_DIR}")


if __name__ == "__main__":
    main()
