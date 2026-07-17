import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

def set_plot_style():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],

        "axes.labelsize": 15,
        "axes.titlesize": 18,
        "legend.fontsize": 13,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,

        "grid.color": "0.85",
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,

        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "black",

        # default colour cycle
        "axes.prop_cycle": cycler(
            color=[
                "darkgoldenrod",
                "teal",
                "firebrick",
                "gold",
                "olive",
                "darkkhaki",
                "sienna"
            ]
        ),
    })


# custom heatmap colour map
heatmap_cmap = LinearSegmentedColormap.from_list(
    "epidemic_heatmap",
    [
        "firebrick",
        "darkgoldenrod",
        "darkkhaki",
        "teal"
    ]
)