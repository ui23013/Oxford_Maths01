import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Oxford_Maths01.plotting_style import set_plot_style, heatmap_cmap

set_plot_style()


df_tau34 = pd.read_csv('Hybrid_Network_Model/hybrid_sweep_results_tau34.csv')
df_tau56 = pd.read_csv('Hybrid_Network_Model/hybrid_sweep_results_tau56.csv')

df = pd.concat([df_tau34, df_tau56], ignore_index=True)
df.columns = df.columns.str.strip()


# ======================================================
# 2. PARAMETER ORDER
# ======================================================
tau_order = sorted(df['tau'].unique())
delay_order = sorted(df['delay'].unique(), reverse=True)

print(f"Tau order: {tau_order}")
print(f"Delay order (top → bottom): {delay_order}")


# ======================================================
# 3. CREATE FIGURE
# ======================================================
n_tau = len(tau_order)
n_delay = len(delay_order)

fig = plt.figure(
    figsize=(3 * n_tau + 1, 3 * n_delay)
)

gs = fig.add_gridspec(
    n_delay,
    n_tau,
    right=0.92,
    wspace=0.3,
    hspace=0.45
)


# Global colour scale
vmin = df['std_final_epidemic_fraction'].min()
vmax = df['std_final_epidemic_fraction'].max()

first_im = None


# ======================================================
# 4. HEATMAP GRID
# ======================================================
for i, delay in enumerate(delay_order):
    for j, tau in enumerate(tau_order):

        ax = fig.add_subplot(gs[i, j])

        subset = df[
            (df['tau'] == tau) &
            (df['delay'] == delay)
        ]

        if subset.empty:
            ax.text(
                0.5,
                0.5,
                f"No data\nτ={tau}, δ={delay}",
                ha="center",
                va="center",
                transform=ax.transAxes
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue


        # -------------------------------------------------
        # Create p-q grid
        # -------------------------------------------------
        pq_grid = subset.pivot(
            index='q',
            columns='p',
            values='std_final_epidemic_fraction'
        )

        # q decreases bottom → top
        pq_grid = pq_grid.sort_index(ascending=False)

        # p increases left → right
        pq_grid = pq_grid.sort_index(axis=1, ascending=True)


        im = sns.heatmap(
            pq_grid,
            cmap=heatmap_cmap,
            annot=False,
            cbar=False,
            vmin=vmin,
            vmax=vmax,
            ax=ax
        )

        if first_im is None:
            first_im = im


        # -------------------------------------------------
        # Only show selected ticks
        # -------------------------------------------------

        p_ticks = [0.0, 0.5, 1.0]
        q_ticks = [0.0, 0.25, 0.5]

        p_positions = []
        p_labels = []

        for p in p_ticks:
            if p in pq_grid.columns:
                p_positions.append(
                    np.where(pq_grid.columns == p)[0][0] + 0.5
                )
                p_labels.append(f"{p:g}")

        q_positions = []
        q_labels = []

        for q in q_ticks:
            if q in pq_grid.index:
                q_positions.append(
                    np.where(pq_grid.index == q)[0][0] + 0.5
                )
                q_labels.append(f"{q:g}")


        ax.set_xticks(p_positions)
        ax.set_xticklabels(
            p_labels,
            fontsize=9
        )

        ax.set_yticks(q_positions)
        ax.set_yticklabels(
            q_labels,
            fontsize=9,
            rotation=0
        )

        # Edge-only axis labels

        if j == 0:
            ax.set_ylabel(
                rf"$\delta = {delay}$" "\n" r"$q$",
                fontsize=10
            )
        else:
            ax.set_ylabel("")


        if i == n_delay - 1:
            ax.set_xlabel(
                r"$p$" "\n" rf"$\tau = {tau}$",
                fontsize=10
            )
        else:
            ax.set_xlabel("")

        max_std = subset['std_final_epidemic_fraction'].max()

        ax.set_title(rf"Max Std = {max_std:.3f}", fontsize=10, pad=8)

        ax.grid(False)

cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])

cbar = fig.colorbar(first_im.collections[0], cax=cbar_ax)

cbar.set_label("Standard deviation of final epidemic fraction", fontsize=11)

fig.suptitle(
    "Std in Final Epidemic Size\n"
    "Effect of Intervention Time ($\\tau$) and Reporting Delay ($\\delta$)",
    fontsize=13,
    y=0.98)
plt.tight_layout()
plt.show()






