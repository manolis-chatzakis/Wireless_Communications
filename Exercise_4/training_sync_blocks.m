function [corr_Y, d_opt_corr] = training_sync_blocks(Z_r, training_s, over, Ntr, d_total, Lb)
    % TRAINING_SYNC_BLOCKS: συγχρονισμός με σύμβολα εκπαίδευσης, ανθεκτικός σε CFO
    % Αθροίζει τα μέτρα των μερικών συσχετίσεων σε μπλοκ Lb συμβόλων, ώστε η
    % ολίσθηση φάσης λόγω CFO να μη χαλάει το άθροισμα.
    % Inputs:
    %   Z_r: matched filter output
    %   training_s: τα Ntr σύμβολα εκπαίδευσης
    %   over: oversampling factor (T/Ts)
    %   Ntr: πλήθος συμβόλων εκπαίδευσης
    %   d_total: πλήθος υποψήφιων d (αναζήτηση σε d = 1:d_total, 1-based)
    %   Lb: μήκος μπλοκ σε σύμβολα
    % Outputs:
    %   corr_Y: sum_b |sum_{k in block b} conj(A_k)*Z_r(d + k*over)|
    %   d_opt_corr: argmax corr_Y (1-based index)

    need_len = d_total + (Ntr-1)*over;
    if length(Z_r) < need_len
        Z_r = [Z_r; complex(zeros(need_len - length(Z_r), 1))];
    end

    nblk = floor(Ntr/Lb);
    corr_Y = zeros(d_total, 1);
    for dd = 1:d_total
        for b = 0:nblk-1
            k = (b*Lb : b*Lb + Lb - 1).';   % 0-based symbol indices
            corr_Y(dd) = corr_Y(dd) + abs(sum(conj(training_s(k+1)) .* Z_r(dd + k*over)));
        end
    end
    [~, d_opt_corr] = max(corr_Y);
end
