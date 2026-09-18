function plot_constellation(x, title_str)
% Scatter plot of a complex sequence together with the 4-QAM points

figure;
scatter(real(x), imag(x), 20, 'b', 'filled'); hold on;
scatter([1 1 -1 -1], [1 -1 1 -1], 60, 'r', 'filled');
grid on; axis equal;
xlabel('Re'); ylabel('Im');
title(title_str);
legend('output samples', '4-QAM points', 'Location', 'bestoutside');
hold off;
end
