import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

def load_sweep_results(csv_path):
    '''
    load sweep data
    Args:
        csv_path: to data on device
    '''
    return pd.read_csv(csv_path)


def plot_pq_heatmap(sweep_df, tau, val_col='mean_r_inf'):
    '''
    plots 2d heatmap of mean r_inf over p,q space for given tau value
    Args:
        sweep_df: df with columns p, q, tau, mean_inf
        tau: which tau slice to plot
        val_col: which col ot plot as the heatmap colour
    '''

    tau_data = sweep_df[sweep_df['tau']==tau]
    pq_grid = tau_data.pivot(index='q', columns='p', values=val_col)
    pq_grid = pq_grid.sort_index(ascending=False)          # q increasing upward
    pq_grid = pq_grid.sort_index(axis=1, ascending=True)  # p increasing rightward

    fig, ax = plt.subplots(figsize=(6, 5))
    sb.heatmap(pq_grid, annot=False, cmap= 'viridis', cbar_kws={'label': val_col}, ax=ax)
    ax.set_xlabel('p')
    ax.set_ylabel('q')
    ax.set_title(f'{val_col} (tau = {tau})')
    plt.tight_layout()
    plt.show()
    return fig,ax

trial_csv = load_sweep_results('/Users/joycewilliamslt/Documents/GitHub/Oxford_Maths01/trial4_sweep_results.csv')

tau_vals = [5, 10]

for tau in tau_vals:
    trial_heatmap = plot_pq_heatmap(trial_csv, tau, val_col='mean_r_inf')
    print(trial_heatmap)