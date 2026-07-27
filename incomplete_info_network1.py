'''
For the incomplete information model alone (no delay), this py does:

    - a full param sweep across (p, q) across multiple tau values
    - a param sweep across q with a constant p value
    - records all stochastic runs for a given (p,q) combination and plots its histogram
'''

import networkx as nx
import matplotlib.pyplot as plt
import EoN as eon
import random
import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial
from plotting_style import set_plot_style, heatmap_cmap

set_plot_style()

def create_rrg(k, N):
    graph = nx.random_regular_graph(k, N)
    return graph


def run_first_sim(graph, beta, gamma, rho, tau):
    '''
    runs simulation until first intervention time and returns the infected
    and recovered nodes at time tau
    Args:
        beta: transmission rate
        gamma: recovery rate
        rho: proportion of population that's initially infected
        tau: intervention time
    '''
    # simulate until first intervention
    sim = eon.Gillespie_SIR(graph, beta, gamma, rho=rho, tmax=tau, return_full_data=True)

    # and then extract states at intervention time, tau
    first_states = sim.get_statuses(time=tau)

    # retrieve infected and recovered nodes at time tau
    infected_nodes = [node for node, status in first_states.items() if status == 'I']
    recovered_nodes = [node for node, status in first_states.items() if status == 'R']

    return infected_nodes, recovered_nodes, first_states


def edge_removal(graph, p, q,  i_nodes):
    '''
    removes a proportion of edges from the graph
    with probability p, each removal preferentially chooses a edge adjacent to an infected node,
    otherwise (so with probabilty 1-p OR if there are no such edges) a random edge is removed
    (in the second instance this alters p?)
    Args:
        graph: simualtion graph
        i_nodes: infected nodes from simulation
    '''
    randgen = random.Random()

    # define total edges in the graph and thus how many should be removed
    total_edges = graph.number_of_edges()
    removal_count = round(q * total_edges)

    if removal_count == 0: # remove nothing case
        return graph

    # define set of all infected nodes
    infected_nodes = set(i_nodes)

    for e in range(removal_count):
        # and list of all edges in the graph
        current_edges = list(graph.edges())

        if not current_edges:
            break

        # create set and list of informed edges
        informed_edges = [edge for edge in current_edges
                        if edge[0] in infected_nodes or edge[1] in infected_nodes]

        # preferentially choose edge adjacent to infected node
        if informed_edges and randgen.random() < p:
            removed_edge = randgen.choice(informed_edges)
        else:
            removed_edge = randgen.choice(current_edges)

        graph.remove_edge(*removed_edge)

        # print('Informed edges:', len(informed_edges)) # debug
    return graph


def run_second_sim(graph, beta, gamma, infected_nodes, recovered_nodes,tmin, tmax):
    '''
    run simultation from time tau to tmax
    '''
    try:
        sim = eon.Gillespie_SIR(graph, beta, gamma, initial_infecteds=infected_nodes, initial_recovereds=recovered_nodes,
                            tmin=tmin, tmax=tmax, return_full_data=True)

        return sim
    except ZeroDivisionError:

        return None # if no transmission is possible, return current state


def sim_single_intervention(N, k, beta, gamma, rho, tau, p, q, sim_duration):
    '''
    simulate the SIR model which models having incomplete information
    '''

    # create rrg graph
    sim_g = create_rrg(k, N)
    initial_edges = sim_g.number_of_edges()

    # simulate until time tau and extract the infected nodes
    infected_nodes, recovered_nodes, first_states = run_first_sim(sim_g, beta, gamma, rho, tau)

    # debug
    # print('Infected nodes', len(infected_nodes))

    # obtain graph after edge removal
    modified_graph = edge_removal(sim_g, p, q, infected_nodes)
    edges_removed = initial_edges - modified_graph.number_of_edges()

    # and run a second sum to see results of intervention
    second_sim = run_second_sim(modified_graph, beta, gamma, infected_nodes, recovered_nodes, tmin=tau, tmax=sim_duration)

    if second_sim is None:  # Simulation couldn't run (no transmission possible)
        final_infected = len(infected_nodes)
        final_recovered = len(recovered_nodes)
    else:
        final_states = second_sim.get_statuses(time=sim_duration)
        final_infected = sum(1 for status in final_states.values() if status == 'I')
        final_recovered = sum(1 for status in final_states.values() if status == 'R')

    # print(final_infected) # debug to ensure epidemic is in fact over

    total_infected = final_infected + final_recovered

    return modified_graph, edges_removed, final_infected, final_recovered, total_infected, len(infected_nodes)


# github copilot
def _run_single_sim(_, N, k, beta, gamma, rho, tau, p, q, sim_duration):
    '''Helper function for parallelization'''
    return sim_single_intervention(N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau,
                                   p=p, q=q, sim_duration=sim_duration)


def run_repeated_sims(N, k, beta, gamma, rho, tau, p, q, sim_duration, n_runs, n_workers=None):
    '''
    runs same simulation multiple times
    Args:
        sim_duration: how long the sim should be run from
        seed: graph no.
        n_runs: how many repeats
        n_works: number of parallel works
    '''

    run_sim = partial(_run_single_sim, N=N, k=k, beta=beta, gamma=gamma, rho=rho,
                      tau=tau, p=p, q=q, sim_duration=sim_duration)

    with Pool(n_workers) as pool:
        results = pool.map(run_sim, range(n_runs))

    records = [
        {
            'seed_infected': seed_infected,
            'edges_removed': edges_removed,
            'final_infected': final_infected,
            'final_recovered': final_recovered,
            'total_infected': total_infected
        }
        for modified_graph, edges_removed, final_infected, final_recovered, total_infected, seed_infected in results
    ]

    # for _ in range(n_runs):
    #     (modified_graph, edges_removed, final_infected, final_recovered, total_infected, seed_infected) = (
    #         sim_single_intervention(N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau, p=p, q=q,
    #                                 sim_duration=sim_duration))
    #
    #     records.append({'seed_infected': seed_infected, 'edges_removed': edges_removed, 'final_infected':final_infected,
    #                     'final_recovered': final_recovered, 'total_infected': total_infected,})

    return pd.DataFrame(records)


def compare_interventions(N, k, beta, gamma, rho, tau, sim_duration, n_runs):
    '''
    compare NI, RI and PI on a graph
    Args:
        n_runs: how many comparisons should be done
    '''

    df_no = run_repeated_sims(N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau,
                               p=1, q=0.0, sim_duration=sim_duration,n_runs=n_runs, n_workers=4)
    df_random = run_repeated_sims(N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau,
                                   p=0, q=0.1, sim_duration=sim_duration, n_runs=n_runs, n_workers=4)
    df_perfect = run_repeated_sims(N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau,
                                    p=1, q=0.1, sim_duration=sim_duration, n_runs=n_runs, n_workers=4)

    return pd.Series({
        'no_intervention': (df_no['final_recovered'] / N).mean(),
        'random_intervention': (df_random['final_recovered'] / N).mean(),
        'perfect_intervention': (df_perfect['final_recovered'] / N).mean(),
        }, name='mean_r_inf')



def parameter_sweep(N, k_vals, beta, gamma, rho, sim_duration, n_runs, p_vals, q_vals, tau_vals,
                    metric='final_recovered', save_path=None, n_workers=None):
        '''
        does a parameter sweep across (p, q, tau) and records mean r_inf for a given parameter combination
        Args:
            k_vals: list of possible degree values for graph
            beta_range: min and max beta values for sampling of beta
            gamma_range: same as above but for gamma
            p_vals: list of values for probability of informed edge removal, p
            q_vals: list of values for proportion of edges removed at intervention, q
            tau_vals: list of values for intervention time, tau
            metric: the metric that we find a mean value for, in this instance r_inf
            save_path: to save to device
        '''
        # to optimise, precompute all param combos
        param_combos = [(tau, p_val, q_val) for tau in tau_vals for p_val in p_vals for q_val in q_vals]
        total_combos = len(param_combos
                           )
        records = [] # initialise empty list to store data
        combo_num = 0

        for combo_num, (tau, p_val, q_val) in enumerate(param_combos, 1):

            # # randomly sample beta and gamma
            # beta = np.random.uniform(beta_range[0], beta_range[-1])
            # gamma = np.random.uniform(gamma_range[0], gamma_range[-1])
            k = np.random.choice(k_vals)

            # for given combination, repeatedly run an sir sim
            df = run_repeated_sims(N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau, p=p_val, q=q_val
                                    ,sim_duration=sim_duration, n_runs=n_runs, n_workers=n_workers)

            # and get average r_inf value (as % of population) vectorised
            mean_r_inf = (df[metric]/N).mean()
            # append param combo and corresponding mean r_inf value plus beta, gamma and k used
            records.append({'p': p_val, 'q': q_val, 'tau': tau,  'k': k, 'beta': beta, 'gamma': gamma,
                            'mean_r_inf': mean_r_inf})

            print(f'[{combo_num}/{total_combos}] p={p_val}, q={q_val}, tau={tau}, k={k}, beta={beta:.4f}, gamma={gamma:.4f} '
                    f'mean_r_inf={mean_r_inf:.4f}')

        sweep_df = pd.DataFrame(records)

        if save_path is not None:
            sweep_df.to_csv(save_path, index=False)
            print(f'Saved sweep results to {save_path}')

        return sweep_df

def plot_q_sweep(
    N, k, beta, gamma, rho, tau_values, sim_duration, q_values, p_values=(0, 0.5, 1), n_runs=1000, n_workers=None,):
    """
    Plot R_inf vs q for multiple intervention times tau.
    Produces one figure with one subplot per tau.
    """
    tau_values = np.atleast_1d(tau_values)

    # Flatten parameter space
    param_combos = [
        (tau, p, q)
        for tau in tau_values
        for p in p_values
        for q in q_values]

    total = len(param_combos)

    # results[tau][p] = list over q
    results = {tau: {p: [] for p in p_values}
        for tau in tau_values}

    for i, (tau, p, q) in enumerate(param_combos, start=1):

        df = run_repeated_sims(
            N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau=tau, p=p, q=q, sim_duration=sim_duration,
            n_runs=n_runs, n_workers=n_workers,)

        mean_r_inf = (df["final_recovered"] / N).mean()
        results[tau][p].append(mean_r_inf)

        print(
            f"[{i}/{total}] "
            f"tau={tau}, p={p:.2f}, q={q:.3f}, "
            f"R_inf={mean_r_inf:.4f}")

    fig, axes = plt.subplots(1,  len(tau_values), figsize=(6 * len(tau_values), 5), sharey=True,)

    if len(tau_values) == 1:
        axes = [axes]

    for ax, tau in zip(axes, tau_values):

        for p in p_values:
            ax.plot(q_values, results[tau][p], marker="o", linewidth=2, label=fr"$p={p}$",)

        ax.set_title(rf"$\tau={tau}$")
        ax.set_xlabel(r"Proportion removed, $q$")
        ax.grid(True)

    axes[0].set_ylabel(r"Mean epidemic size, $R_\infty$")
    axes[-1].legend()

    plt.tight_layout()
    plt.savefig("q_sweep_plot.pdf")
    plt.show()


def plot_histogram_for_params(N, k, beta, gamma, rho, tau, p, q, sim_duration,
                              n_runs, n_workers=None, metric='total_infected',
                              bins=30, normalize=True, save_path=None):
    """
    Run repeated simulations for a single parameter combination and plot
    a histogram of the chosen metric.
    """
    # Run the simulations
    df = run_repeated_sims(N=N, k=k, beta=beta, gamma=gamma, rho=rho,
                           tau=tau, p=p, q=q, sim_duration=sim_duration,
                           n_runs=n_runs, n_workers=n_workers)

    # Extract the chosen metric
    values = df[metric].values
    if normalize:
        values = values / N

    # LaTeX labels for the chosen metric
    metric_labels = {
        'total_infected': r'$R_\infty$',      # final epidemic size
        'final_recovered': r'$R_\infty$',     # same as total infected at end
        'final_infected': r'$I_\infty$'       # remaining infected (usually 0)
    }
    base_label = metric_labels.get(metric, metric.replace('_', ' '))
    xlabel = rf'{base_label} / $N$' if normalize else rf'{base_label} (count)'

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(rf'Histogram for $p={p:.2f}$, $q={q:.3f}$, $\tau={tau}$')
    plt.grid(True, alpha=0.3)

    # Vertical line at mean
    mean_val = values.mean()
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                label=f'Mean = {mean_val:.4f}')
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    # # Summary statistics
    # print(f"Summary for p={p:.2f}, q={q:.3f}, tau={tau}, metric={metric}:")
    # print(f"  Mean   : {values.mean():.4f}")
    # print(f"  Std    : {values.std():.4f}")
    # print(f"  Median : {np.median(values):.4f}")
    # print(f"  Min    : {values.min():.4f}")
    # print(f"  Max    : {values.max():.4f}")

    return df


def plot_histogram_multiple(N, k, beta, gamma, rho, sim_duration, n_runs,
                            param_combos, metric='total_infected', normalize=True,
                            bins=30, n_workers=None, save_fig_path=None, save_csv_path=None):
    """
    Plot histograms for multiple (p, q, tau) combinations in a grid,
    and optionally save a CSV summary.

    Parameters
    ----------
    param_combos : list of tuples
        Each tuple is (p, q, tau).
    save_fig_path : str or None
        Path to save the figure (e.g., 'histogram_grid.pdf').
    save_csv_path : str or None
        Path to save the summary CSV.
    Other parameters same as before.

    Returns
    -------
    pd.DataFrame
        Summary statistics for each combination.
    """
    n_combos = len(param_combos)
    if n_combos == 0:
        print("No combinations provided.")
        return pd.DataFrame()

    # Run simulations for each combo
    results = []   # list of (p, q, tau, values_array)
    summary_records = []

    for idx, (p, q, tau) in enumerate(param_combos, 1):
        print(f"Running combo {idx}/{n_combos}: p={p:.2f}, q={q:.3f}, tau={tau}")
        df = run_repeated_sims(N=N, k=k, beta=beta, gamma=gamma, rho=rho,
                               tau=tau, p=p, q=q, sim_duration=sim_duration,
                               n_runs=n_runs, n_workers=n_workers)
        values = df[metric].values
        if normalize:
            values = values / N

        # Summary stats
        rec = {'p': p, 'q': q, 'tau': tau, 'mean': values.mean(), 'std': values.std(), 'median': np.median(values),
            'min': values.min(), 'max': values.max()}
        summary_records.append(rec)
        results.append((p, q, tau, values))

    summary_df = pd.DataFrame(summary_records)

    # Save CSV if requested
    if save_csv_path is not None:
        summary_df.to_csv(save_csv_path, index=False)
        print(f"Summary CSV saved to {save_csv_path}")

    # LaTeX labels for axis
    metric_labels = {
        'total_infected': r'$R_\infty$',
        'final_recovered': r'$R_\infty$',
        'final_infected': r'$I_\infty$'
    }
    base_label = metric_labels.get(metric, metric.replace('_', ' '))
    xlabel = rf'{base_label} / $N$' if normalize else rf'{base_label} (count)'

    # Create subplot grid
    ncols = int(np.ceil(np.sqrt(n_combos)))
    nrows = int(np.ceil(n_combos / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))

    # Flatten axes if more than one, else put in list
    if n_combos > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for ax, (p, q, tau, vals) in zip(axes_flat, results):
        ax.hist(vals, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_title(rf'$p={p:.2f}, q={q:.3f}, \tau={tau}$')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes_flat[n_combos:]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_fig_path is not None:
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_fig_path}")
    plt.show()

    return summary_df

if __name__ == '__main__':

    # test a single combination and plot histogram
    # plot_histogram_for_params(
    #     N=1000, k=3, beta=0.4, gamma=0.2, rho=0.05,
    #     tau=3, p=0.6, q=0.2, sim_duration=75,
    #     n_runs=1000, n_workers=6,
    #     metric='total_infected', normalize=True,
    #     bins=50, save_path='histogram_tau3_p08_q03.pdf')

    # (p, q, tau) combinations
    combos = [(0.9, 0.45, 3), (0.8, 0.4, 3), (0.6, 0.425, 3), # red region
              (1.0, 0.25, 3), (0.8, 0.3, 3), (0.6, 0.35, 3), (0.4, 0.4, 3), (0.2, 0.45, 3), (0, 0.5, 3), # STD line
              (1.0, 0.15, 3), (0.8, 0.2, 3), (0.6, 0.25, 3), (0.4, 0.3, 3), (0.2, 0.35, 3), (0, 0.4, 3), # Yellow to blue
              (1.0, 0, 3), (0.8, 0.05, 3), (0.6, 0.1, 3), (0.4, 0.15, 3), (0.2, 0.2, 3), (0, 0.25, 3), # bluish yellow
              (0.1, 0.05, 3), (0.2, 0.125, 3), (0.4, 0.025, 3),] # blue region

    summary = plot_histogram_multiple(
        N=1000, k=3, beta=0.4, gamma=0.2, rho=0.05,
        sim_duration=75, n_runs=1000, param_combos=combos,
        metric='total_infected', normalize=True, bins=50,
        n_workers=6,
        save_fig_path='histogram_grid.pdf',
        save_csv_path='histogram_summary.csv')


    # p_values = list(np.linspace(0, 1, 10))
    # q_values = list(np.linspace(0, 0.5, 10))
    # tau_values = list(range(9, 16, 2))  # [9, 11, 13, 15]
    # print(tau_values)
    #
    # # min and max values for beta and gamma
    # beta = 0.4
    # gamma = 0.2
    # k_range = [3]
    # num_workers = 6
    #
    # for tau in tau_values:
    #     trial_sweep = parameter_sweep(
    #         N=1000, k_vals=k_range, beta=beta, gamma=gamma, rho=0.05,
    #         sim_duration=75, n_runs=1000, p_vals=p_values, q_vals=q_values, tau_vals=[tau],
    #         save_path=f'sweep_results_tau{tau}_2007.csv', n_workers=num_workers)


#
#         N = 1000
#         k = 3
#         beta, gamma = 0.4, 0.3
#         rho = 0.05
#         tau_values = [3, 4, 5, 6, 7]
#         sim_duration = 75
#
#         # q values to investigate
#         q_values = np.linspace(0, 0.5, 31)
#
#         # compare random, partial-information and perfect-information interventions
#         p_values = [0, 0.5, 1]
#
#         # num of stochastic repeats for each q
#         n_runs = 1000
#
#         # produce bifurcation-style plot
#         plot_q_sweep(N=N, k=k, beta=beta, gamma=gamma, rho=rho, tau_values=tau_values, sim_duration=sim_duration, q_values=q_values,
#             p_values=p_values, n_runs=n_runs)