#####################################################################################################
#   My own wrapper around Seaborn and Matplotlib for making plots that I use frequently
#####################################################################################################

import seaborn as sns
import matplotlib.pyplot as plt
import pandas
import numpy as np

def black_white_barplot(
    data_frame, 
    xcol, 
    ycol, 
    decimal=False, 
    ax=None
    ):

    clrs = ['#E5E7E9' for x in range(data_frame.shape[0])]

    sns.set_context(rc = {'patch.linewidth': 1.0})
    sns.set_style("white", {'axes.grid' : False})
    if ax:
        ax = sns.barplot(
            x=xcol, 
            y=ycol, 
            palette=clrs, 
            data=data_frame, 
            ci=None,
            ax=ax
        )
    else:
        ax = sns.barplot(
            x=xcol,
            y=ycol,
            palette=clrs,
            data=data_frame,
            ci=None
        )
    sns.despine()

    for p in ax.patches:
        height = p.get_height()
        y_lim = ax.get_ylim()[1]
        if decimal:
            ax.text(
                p.get_x() + 0.25 * p.get_width(), 
                height + 0.025 * y_lim, 
                str(height), 
                fontsize=13
            )
        else:
             ax.text(
                p.get_x() + 0.25 * p.get_width(),
                height + 0.025 * y_lim,
                '%d' % height,
                fontsize=13
            )

    sns.plt.ylabel(ycol, fontsize=15)
    sns.plt.xlabel(xcol, fontsize=15)
    sns.plt.tick_params(labelsize=13);

    return ax


def histogram(
    data_points,
    x_label,
    y_label="Counts",
    ax=None):

    sns.distplot(data_points, hist=True, rug=False, kde=False, ax=ax)
    for p in ax.patches:
        height = p.get_height()
        y_lim = ax.get_ylim()[1]
        if height > 0:
            ax.text(
                p.get_x() + 0.25 * p.get_width(),
                height + 0.025 * y_lim,
                '%d' % height,
                fontsize=13
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    

def single_var_strip_plot(
    data_frame, 
    xcol, 
    ycol, 
    x_lims=None, 
    x_ticks=None,
    label_y_axis=True,
    label_x_axis=True,
    y_tick_font_size=None,
    x_grid=True,
    dot_size=10,
    ax=None, 
    show=True,
    x_label_font_size=20,
    y_label_font_size=15,
    x_tick_rotation=None
    ):

    yticks = np.arange(0, data_frame.shape[0], 1)
    ax.set_yticks(yticks)

    if x_lims:
        ax.set_xlim(x_lims)

    if x_ticks:
        ax.set_xticks(x_ticks)
    elif x_lims:
        ax.set_xticks(x_lims)

    for xmaj in ax.yaxis.get_majorticklocs():
        ax.axhline(
            y=xmaj,
            ls='-', 
            color='grey', 
            alpha=0.5, 
            lw=0.002
        )

    #ax.xaxis.grid(x_grid)
    ax.xaxis.grid(linewidth=0.002)
    sns.stripplot(
        data=data_frame,
        x=xcol,
        y=ycol,
        palette="husl",
        size=dot_size,
        ax=ax
    )

    if y_tick_font_size:
        ax.tick_params(axis='y', labelsize=y_tick_font_size)

    if x_tick_rotation:
        for tick in ax.get_xticklabels():
            tick.set_rotation(x_tick_rotation)

    if label_x_axis:
        ax.set_xlabel(xcol, size=x_label_font_size)
    else:
        ax.set_xlabel("")

    if label_y_axis:
        ax.set_ylabel(ycol, size=y_label_font_size)
    else:
        ax.set_ylabel("")

    return ax

def mult_var_strip_plot(
    data_frame,
    xcol,
    ycol,
    hue_col,
    x_lims=None,
    x_ticks=None,
    label_y_axis=True,
    label_x_axis=True,
    y_tick_font_size=None,
    x_grid=True,
    dot_size=10,
    ax=None,
    show=True,
    x_label_font_size=20,
    y_label_font_size=15,
    x_tick_rotation=None
    ):

    yticks = np.arange(0, data_frame.shape[0], 1)
    ax.set_yticks(yticks)

    if x_lims:
        ax.set_xlim(x_lims)

    if x_ticks:
        ax.set_xticks(x_ticks)
    elif x_lims:
        ax.set_xticks(x_lims)

    for xmaj in ax.yaxis.get_majorticklocs():
        ax.axhline(
            y=xmaj,
            ls='-',
            color='grey',
            alpha=0.5,
            lw=0.002
        )

    #ax.xaxis.grid(x_grid)
    ax.xaxis.grid(linewidth=0.002)
    sns.stripplot(
        data=data_frame,
        x=xcol,
        y=ycol,
        hue=hue_col,
        palette="husl",
        size=dot_size,
        ax=ax
    )

    ax.legend(bbox_to_anchor=(1.1, 1.05))

    if y_tick_font_size:
        ax.tick_params(axis='y', labelsize=y_tick_font_size)

    if x_tick_rotation:
        for tick in ax.get_xticklabels():
            tick.set_rotation(x_tick_rotation)

    if label_x_axis:
        ax.set_xlabel(xcol, size=x_label_font_size)
    else:
        ax.set_xlabel("")

    if label_y_axis:
        ax.set_ylabel(ycol, size=y_label_font_size)
    else:
        ax.set_ylabel("")

    return ax


def horizontal_bar_graph_clusters(
    label_to_df, 
    xcol, 
    ycol, 
    hue=None, 
    label_order=None
    ):

    sns.set_context(rc = {'patch.linewidth': 0.0})
    sns.set_style('white')

    if not label_order:
        label_order = list(label_to_df.keys())
    num_labels = len(label_order)   

    fig, axarr = plt.subplots(
        num_labels,
        1,
        sharex='col',
        figsize=(10, 1*num_labels),
        gridspec_kw = {
            'height_ratios':[
                label_to_df[x].shape[0]
                for x in label_order    
            ]
        }
    )

    for label_i, label in enumerate(label_order):
        df = label_to_df[label]
        ax = axarr[label_i]
        ax.set_title(label)
        ax = sns.barplot(
            data=df,
            x=xcol, 
            y=ycol, 
            hue=hue, 
            ax=ax, 
            ci=None
        )

        ax.set_ylabel("")
        ax.set_yticks([])
        if label_i != len(label_order)-1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel(xcol)

        ax.legend_.remove()

    sns.plt.tight_layout() 
    return fig, axarr

def performance_comparison_scatterplot(data_frame, xcol, ycol, xlim=None, ylim=None, ax=None, title=None):

    ax = sns.regplot(
        data=data_frame, 
        x=xcol, 
        y=ycol, 
        fit_reg=False,
        ci=None, 
        scatter=True,
        scatter_kws={"s": 50, "alpha": 0.5, "color":"blue"}, 
        ax=ax
    )
   
    max_lim = max([ax.get_ylim()[1], ax.get_xlim()[1]])
    print max_lim
    ax.plot([0, max_lim], [0, max_lim], 'k-')
 
    if title:
        ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    return ax


