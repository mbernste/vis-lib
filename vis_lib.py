#####################################################################################################
#   My own wrapper around Seaborn and Matplotlib for making plots that I use frequently
#####################################################################################################

import seaborn as sns
import matplotlib.pyplot as plt
import pandas
import numpy as np
from collections import defaultdict

RED = '#ff0000'
BLUE = '#0000ff'
GREEN = '#00ff00'
ORANGE = '#ed741d'
PINK = '#ff00ff'
WHITE = '#ffffff'

NICE_RED = '#ff3633'
NICE_BLUE = '#0066ff'
NICE_GREEN = '#3dcd26'
NICE_ORANGE = '#ee8420'
NICE_PINK = '#d64edd'
NICE_PURPLE = '#8200fc'

MAROON = '#800000'
SHRILL_BLUE = '#33d6ff'
DARK_GREEN = '#004d00'
ROTTING_ORANGE = '#b36b00'
DARK_PINK = '#99004d'

STANDARD_COLORS = [
    RED,
    BLUE,
    GREEN,
    ORANGE,
    PINK
]
NICE_COLORS = [
    NICE_RED, 
    NICE_BLUE, 
    NICE_GREEN, 
    NICE_ORANGE, 
    NICE_PINK
]
AGAINST_NICE_COLORS = [
    MAROON,
    SHRILL_BLUE,
    DARK_GREEN,
    ROTTING_ORANGE,
    DARK_PINK
]

MANY_COLORS = [
    "#0652ff", #  electric blue             
    "#e50000", #  red                       
    "#9a0eea", #  violet                    
    "#01b44c", #  shamrock       
    "#fedf08", #  dandelion                 
    "#00ffff", #  cyan                      
    "#89fe05", #  lime green                
    "#a2cffe", #  baby blue                 
    "#dbb40c", #  gold                      
    "#029386", #  teal                      
    "#ff9408", #  tangerine                 
    "#d8dcd6", #  light grey                
    "#80f9ad", #  seafoam                   
    "#3d1c02", #  chocolate                 
    "#fffd74", #  butter yellow             
    "#536267", #  gunmetal
    "#f6cefc", #  very light purple
    "#650021", #  maroon
    "#020035", #  midnight blue
    "#b0dd16", #  yellowish green
    "#9d7651", #  mocha
    "#c20078", #  magenta
    "#380282", #  indigo
    "#ff796c", #  salmon
    "#874c62", #  dark muave
    "#02ccfe", #  bright sky blue
    "#5fa052", #  muted green
    "#9a3001", #  auburn
    "#fc2647", #  pinky red
    "#d8863b", #  dull orange
    "#7b002c", #  bordeaux
    "#8e82fe", #  periwinkle
    "#ffff14", #  yellow 
    "#ff073a", #  neon red
    "#6ecb3c", #  apple
    "#c45508", #  rust orange
    "#8756e4", #  purpley
    "#8756e4", #  diarrhea
    "#bcecac", #  light sage
    "#5d1451", #  grape purple
    "#028f1e", #  emerald green
    "#ffa62b", #  mango
    "#3a2efe", #  light royal blue
    "#c0022f", #  lipstick red
    "#0485d1", #  cerulean
    "#a57e52", #  puce
    "#380835", #  eggplant
    "#a9f971", #  spring green
    "#fe4b03", #  blood orange
    "#8cff9e", #  baby green
    "#86775f", #  brownish grey
    "#9d0759", #  dark fuchsia
    "#665fd1", #  dark periwinkle
    "#49759c", #  dullblue
    "#fffa86", #  manilla
    "#280137", #  midnight purple
    "#fa4224", #  orangey red
    "#d99b82", #  pinkish tan
    "#152eff", #  vivid blue
    "#f2ab15", #  squash
    "#70b23f", #  nasty green
    "#952e8f", #  warm purple
    "#bcf5a6", #  washed out green
    "#9a6200", #  raw sienna
    "#fb5ffc", #  violet pink
    "#ddd618", #  piss yellow
    "#fe420f", #  orangered
    "#c27e79", #  brownish pink
    "#adf802", #  lemon green
    "#29465b", #  dark grey blue
    "#48c072", #  dark mint
    "#edc8ff"  #  light lilac
]

def black_white_barplot(
        data_frame, 
        xcol, 
        ycol, 
        decimal=False, 
        ax=None,
    ):
    """
    Plot a black and white bar plot. 
    Args:
        data_frame: a pandas DataFrame
        xcol: the categorical column to plot along the x axis
        ycol: the numerical column to plot along the y axis
        decimal: if true, use decimals in the height label
            above each bar
        ax: the matplotlib Axes object on which to draw the
            figure
    """

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

    plt.ylabel(ycol, fontsize=15)
    plt.xlabel(xcol, fontsize=15)
    plt.tick_params(labelsize=13);

    return ax


def black_white_double_barplot(
        data_frame,
        xcol,
        ycol,
        hue,
        decimal=False,
        ax=None,
    ):
    clrs = []
    for i in range(data_frame.shape[0]):
        clrs.append('#E5E7E9')
        clrs.append('#737373')
#    clrs = ['#E5E7E9' for x in range(data_frame.shape[0])]

    sns.set_context(rc = {'patch.linewidth': 1.0})
    sns.set_style("white", {'axes.grid' : False})
    if ax:
        ax = sns.barplot(
            x=xcol,
            y=ycol,
            hue=hue,
            palette=clrs,
            data=data_frame,
            ci=None,
            ax=ax
        )
    else:
        ax = sns.barplot(
            x=xcol,
            y=ycol,
            hue=hue,
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

    plt.ylabel(ycol, fontsize=15)
    plt.xlabel(xcol, fontsize=15)
    plt.tick_params(labelsize=13);

    return ax



def histogram(
        data_points,
        x_label,
        y_label="Counts",
        ax=None,
        label_heights=True
    ):

    sns.distplot(data_points, hist=True, rug=False, kde=False, ax=ax)
    if label_heights:
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
    """
    Stripplots in which each strip contains two points
    comparing two methods. This figure is used to compare
    the a metric across many labels between two partitions
    of the data.
    """

    print "Dataframe shape: %d" % data_frame.shape[0]
    yticks = np.arange(0, data_frame.shape[0]/2, 1)
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


def performance_comparison_stripplot(
        condition_to_label_to_value,
        condition_var_name,
        label_var_name,
        value_var_name,
        ax
    ):

    print "CONDITION_TO_LABEL_TO_VALUE"
    print condition_to_label_to_value
    print

    conditions = list(
        condition_to_label_to_value.keys()
    )
    assert len(conditions) == 2
    c_1 = conditions[0]
    c_2 = conditions[1]

    da = [
        (
            label,
            condition_to_label_to_value[c_1][label],
            c_1
        )
        for label in condition_to_label_to_value[c_1]
    ]

    # Sort the data by the difference in value between 
    # the first and second condition. This needs to be 
    # done here because it seems Seaborn will order the 
    # label stripplots in the order in which it finds 
    # them in this list
    da = sorted(
        da, 
        key=lambda x: abs(
            x[1] - condition_to_label_to_value[c_2][x[0]]
        ), 
        reverse=True
    )
    labels_order = [
        #x[0]
        ""
        for x 
        in da
    ]

    da += [
        (
            label,
            condition_to_label_to_value[c_2][label],
            c_2
        )
        for label in condition_to_label_to_value[c_2]
    ]

    print "DA:\n%s" % da

    df = pandas.DataFrame(
        data=da,
        columns=[
            label_var_name, 
            value_var_name,
            condition_var_name
        ]
    )

    print "DF:\n%s" % df

    mult_var_strip_plot(
        df,
        value_var_name,
        label_var_name,
        condition_var_name,
        ax=ax
    )
    
    #ticks = ax.get_yticks().tolist()
    #for tick_i in range(len(ticks)):
    #    ticks[tick_i] = labels_order[tick_i]
    #ax.set_yticklabels(ticks)
    #ax.set_yticklabels(labels_order)

    return labels_order

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

    plt.tight_layout() 
    return fig, axarr


def side_by_side_single_var_strip_plots(
        variable_to_label_to_value,
        label_variable,
        axarr,
        variables_order=None,
        x_lims=None
    ):
    """
    Plot multiplie single-variable strip-plots side-by-side each
    sharing the y-axis labels.
    Args:
        variable_to_label_to_value: a dictionary of dictionaries. The
            keys of the outer dictionary are the variables being 
            plotted in each strip-plot (e.g. precision or recall).
            The keys of the inner dictionary are the labels and the
            values are the values for that label and variable.
        label_variable: the name of the variable of the labels
            (e.g. cell-type)
        axarr: an array of matplotlib Axes objects on which to draw
            the strip plots. The length must match the length of the
            variable_to_label_to_value dictionary.
        x_lims: an array of tuples corresponding the x-axis limits
            of each strip-plot
    """

    assert len(axarr) == len(variable_to_label_to_value)

    if not variables_order:
        variables_order = sorted(
            list(variable_to_label_to_value.keys()) 
        )

    order_by_label_to_values = variable_to_label_to_value[variables_order[0]]
    labels_order = sorted(
        [
            label
            for label in order_by_label_to_values.keys()
        ],
        key=lambda x: order_by_label_to_values[x],
        reverse=True
    )

    variable_to_da = defaultdict(lambda: [])
    for variable in variables_order:
        for label in labels_order:
            value = variable_to_label_to_value[variable][label]
            variable_to_da[variable].append((
                label,
                value
            ))

    variable_to_df = {}
    for variable, da in variable_to_da.iteritems():
        variable_to_df[variable] = pandas.DataFrame(
            data=da,
            columns=[
                label_variable,
                variable
            ]
        )

    for var_i, variable in enumerate(variables_order):
        df = variable_to_df[variable]
        if x_lims:
            single_var_strip_plot(
                df,
                variable,
                label_variable,
                x_lims=x_lims[var_i],
                ax=axarr[var_i]
            )
        else:
            single_var_strip_plot(
                df,
                variable,
                label_variable,
                ax=axarr[var_i]
            )
    
    
def performance_comparison_scatterplot(
        data_frame, 
        xcol, 
        ycol, 
        xlim=None, 
        ylim=None, 
        ax=None, 
        title=None,
        dot_size=50
    ):

    ax = sns.regplot(
        data=data_frame, 
        x=xcol, 
        y=ycol, 
        fit_reg=False,
        ci=None, 
        scatter=True,
        scatter_kws={
            "s": dot_size, 
            "alpha": 0.5, 
            "color":"blue"
        }, 
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


def grid_performance_comparison_scatterplots(
        variable_to_label_to_value_x,
        variable_to_label_to_value_y,
        x_name,
        y_name,
        variable_grid,
        axarr,
        x_lim_grid=None,
        y_lim_grid=None
    ):

    x_vars = frozenset(variable_to_label_to_value_x.keys())
    y_vars = frozenset(variable_to_label_to_value_y.keys())
    assert x_vars == y_vars

    variables = x_vars
    labels = frozenset(
        variable_to_label_to_value_x[list(variables)[0]].keys()
    )
    for var in variables:
        labels_x = frozenset(variable_to_label_to_value_x[var])
        labels_y = frozenset(variable_to_label_to_value_y[var])
        assert labels_x == labels
        assert labels_y == labels


    variable_to_da = defaultdict(lambda: [])
    for variable in variables:
        for label in labels:
            val_x = variable_to_label_to_value_x[variable][label]
            val_y = variable_to_label_to_value_y[variable][label]
            variable_to_da[variable].append((val_x, val_y))

    variable_to_df = {}
    for variable, da in variable_to_da.iteritems():
        variable_to_df[variable] = pandas.DataFrame(
            data=da,
            columns=[x_name, y_name]
        )

    for r_i, row in enumerate(variable_grid):
        for c_i, variable in enumerate(row):
            if variable is None:
                continue
            ax = axarr[r_i][c_i]
            df = variable_to_df[variable]
            if x_lim_grid and y_lim_grid:
                x_lim = x_lim_grid[r_i][c_i]
                y_lim = x_lim_grid[r_i][c_i]
                performance_comparison_scatterplot(
                    df,
                    xcol=x_name,
                    ycol=y_name,
                    title=variable,
                    ax=ax,
                    xlim=x_lim,
                    ylim=y_lim   
                )
            else:
                performance_comparison_scatterplot(
                    df,
                    xcol=x_name,
                    ycol=y_name,
                    title=variable,
                    ax=ax
                )
        






