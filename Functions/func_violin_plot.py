## ----------------- FUNCTIONS TO CREATE AND STYLE VIOLIN PLOTS -------------------------------------------------------------------------------------

# Replaces func_al_goodplot with seaborn, keeping the same elements:
#       - violin = distribution of the data
#       - individual data points
#       - boxplots indicate the interquartile range
#       - notches indicate the median
#       - asterisks indicate the mean (drawn on top of the data points)

from itertools import cycle

import seaborn as sns


def func_violin_plot(data, y, order, colors, ax, point_alpha=0.3, bandwidth=1):
    #   Input parameters:
    #       - data = table with a 'Condition' column and a column of values to plot.
    #       - y = name of the column of values to plot.
    #       - order = conditions in the order they will be plotted.
    #       - colors = list of colours, one per condition. If there are more conditions than colours, extra colours are added automatically.
    #       - ax = axes to plot onto.
    #       - point_alpha = transparency of the individual data points (0 = invisible, 1 = solid).
    #       - bandwidth = smoothness of the violin (higher = smoother).
    extra_colors = cycle(sns.color_palette('tab10')[4:])  # purple, brown, pink, grey, olive, cyan (then repeats)
    colors = list(colors) + [next(extra_colors) for _ in range(len(order) - len(colors))]
    palette = dict(zip(order, colors))
    common = dict(data=data, x='Condition', y=y, order=order, hue='Condition', hue_order=order, palette=palette, dodge=False, legend=False, ax=ax)

    sns.violinplot(**common, inner=None, cut=0, bw_adjust=bandwidth, density_norm='width', width=0.8, linewidth=0, alpha=0.2)
    if point_alpha > 0:
        sns.stripplot(**common, size=3, jitter=0, alpha=point_alpha)
    sns.boxplot(**common, width=0.3, notch=True, showfliers=False, showcaps=False, whis=0, linewidth=1, boxprops=dict(alpha=0.6))

    # mean of each condition as a black asterisk, on top of everything else
    means = data.groupby('Condition', observed=False)[y].mean().reindex(order)
    ax.scatter(range(len(order)), means, marker=(6, 2, 0), s=40, color='k', linewidths=1.5, zorder=10)
    ax.set_xlabel('')
    ax.grid(True, alpha=0.3)


def func_report_off_axis(data, y, yMin, yMax):
    # Prints how many data points are outside the y axis limits (None = automatic limit, so nothing is outside).
    # These points are hidden from view, but are still included in the violin, box & mean (the same data as the tables).
    outside = data[((data[y] > yMax) if yMax is not None else False) | ((data[y] < yMin) if yMin is not None else False)]
    if outside.empty:
        print('All data points are within the y axis limits.')
    else:
        print('%d data points are outside the y axis limits (%s to %s) and not shown, but are still included in the violin, box & mean:'
              % (len(outside), yMin, yMax))
        print(outside.groupby(['Condition', 'Repeat'], observed=True).size().to_string())


def func_style_axes(ax):
    # Same axis styling as the original MATLAB plots
    ax.tick_params(labelsize=18, width=2)
    ax.yaxis.label.set_size(18 * 1.1)
    ax.xaxis.label.set_size(18 * 1.1)
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment('right')
        label.set_rotation_mode('anchor')
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.spines[['top', 'right']].set_visible(False)  # box off
