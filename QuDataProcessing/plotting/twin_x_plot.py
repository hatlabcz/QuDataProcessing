from matplotlib import pyplot as plt


def twin_x_plot(x, y1, y2, x_label=None, y1_label=None, y2_label=None, plot_ax=None):
    if plot_ax is None:
        fig, ax1 = plt.subplots(figsize=(8, 5))
    else:
        ax1 = plot_ax
        fig = None
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y1_label, color="C0")
    ax1.tick_params(axis='y', labelcolor="C0")
    ax1.plot(x, y1)
    ax2 = ax1.twinx()
    ax2.plot(x, y2, color="C1")
    ax2.set_ylabel(y2_label, color="C1")
    ax2.tick_params(axis='y', labelcolor="C1")
    return fig, ax1, ax2