function save_figure(name)
% Saves the current figure as png inside the folder figures/ (used in the report)

if ~exist('figures', 'dir')
    mkdir('figures');
end
% hide the axes toolbar so that it does not appear inside the saved image
ax = findall(gcf, 'Type', 'axes');
for i = 1:length(ax)
    ax(i).Toolbar.Visible = 'off';
end
drawnow;
exportgraphics(gcf, fullfile('figures', [name '.png']), 'Resolution', 150);
end
