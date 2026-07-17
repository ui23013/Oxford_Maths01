import networkx as nx
import matplotlib.pyplot as plt
import EoN as eon
import random
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation
from plotting_style import set_plot_style

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

    return sim, infected_nodes, recovered_nodes, first_states


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
    removed_edges = []

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
        removed_edges.append(removed_edge)
        # print('Informed edges:', len(informed_edges)) # debug
    return graph, removed_edges


def run_second_sim(graph, beta, gamma, infected_nodes, recovered_nodes,tmin, tmax):
    '''
    run simultation from time tau to tmax
    '''
    sim = eon.Gillespie_SIR(graph, beta, gamma, initial_infecteds=infected_nodes, initial_recovereds=recovered_nodes,
                            tmin=tmin, tmax=tmax, return_full_data=True)

    return sim


def sim_single_intervention(N, k, beta, gamma, rho, tau, p, q, sim_duration):
    '''
    simulate the SIR model which models having incomplete information
    '''

    # create rrg graph
    sim_g = create_rrg(k, N)

    # save original graph
    original_graph = sim_g.copy()
    initial_edges = sim_g.number_of_edges()

    # simulate until time tau and extract the infected nodes
    first_sim, infected_nodes, recovered_nodes, first_states = run_first_sim(sim_g, beta, gamma, rho, tau)

    # debug
    # print('Infected nodes', len(infected_nodes))

    # obtain graph after edge removal
    modified_graph, removed_edges = edge_removal(sim_g, p, q, infected_nodes)
    edges_removed = initial_edges - modified_graph.number_of_edges()

    # and run a second sum to see results of intervention
    second_sim = run_second_sim(modified_graph, beta, gamma, infected_nodes, recovered_nodes, tmin=tau, tmax=sim_duration)
    final_states = second_sim.get_statuses(time=sim_duration) # states at final time

    # outcomes
    final_infected = sum(1 for status in final_states.values() if status == 'I')
    final_recovered = sum(1 for status in final_states.values() if status == 'R')
    total_infected = final_infected + final_recovered

    return {"original_graph": original_graph, "modified_graph": modified_graph, "first_sim": first_sim,
            "second_sim": second_sim, "tau": tau, "edges_removed": edges_removed, "final_states": final_states,
            "total_infected": total_infected, "removed_edges": removed_edges}


def node_colours(states):
    '''
    assigns colour to nodes based on their state
    '''

    colours = []

    for node in sorted(states):
        if states[node] == 'S':
            colours.append('darkgoldenrod')

        elif states[node] == 'I':
            colours.append('firebrick')

        elif states[node] == 'R':
            colours.append('teal')

    return colours


def plot_network_snapshots(results, sim_duration):

    original_graph = results['original_graph']
    modified_graph = results['modified_graph']

    first_sim = results['first_sim']
    second_sim = results['second_sim']

    tau = results['tau']
    removed_edges = results['removed_edges']
    # for fixed layout
    pos = nx.spring_layout(original_graph, seed=0)

    fig, axes = plt.subplots (1, 4, figsize = (16,4))

    snapshots = [(original_graph, first_sim.get_statuses(time=0), r"$t=0$", []), # t=0
                 (original_graph, first_sim.get_statuses(time=tau), r"$t=\tau^-$", removed_edges), # t=tau-
                 (modified_graph, first_sim.get_statuses(time=tau), r"$t=\tau^+$", []),  # t=tau+
                 (modified_graph, second_sim.get_statuses(time=sim_duration), r"$t=t_f$", [])] # t=t_f

    for ax, (graph, states, title, highlight_edges) in zip(axes, snapshots):
        nx.draw_networkx(graph, pos, node_color=node_colours(states), node_size=25, edge_color='lightgrey',
                         with_labels=False, ax=ax)

        if highlight_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=highlight_edges, edge_color='red', width=2, ax=ax)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    return pos


def animate_sim(results, sim_duration, pos):

    original_graph = results["original_graph"]
    modified_graph = results["modified_graph"]

    first_sim = results["first_sim"]
    second_sim = results["second_sim"]

    tau = results["tau"]

    times = np.linspace(0, sim_duration, 100)

    fig, ax = plt.subplots(figsize=(6,6))

    def update(t):

        # clear for 'each frame'
        ax.clear()

        if t <= tau: # original graph with states for all times before tau
            graph = original_graph
            states = first_sim.get_statuses(time=t)

        else: # otherwise use modified graph and continue till t_f
            graph = modified_graph
            states = second_sim.get_statuses(time=t)

        nx.draw_networkx(graph, pos, node_color=node_colours(states), node_size=25, edge_color='lightgrey',
                         with_labels=False, ax=ax)

        ax.set_title(rf"$t={t:.2f}$")

    ani = FuncAnimation(fig, update, frames=times, interval=100)
    plt.close()

    return ani


sim_dur=100
# trial results
trial_results = sim_single_intervention(N=1000, k=3, beta=0.4, gamma=0.2, rho=0.05, tau=15, p=0.8, q=0.3
                                        , sim_duration=sim_dur)

# trial figures
trial_figs = plot_network_snapshots(trial_results, sim_duration=sim_dur)

# trial_animation
trial_ani = animate_sim(trial_results, sim_duration=sim_dur, pos=trial_figs)

# display in notebook
# from IPython.display import HTML
# HTML(trial_ani.to_jshtml())

output_path = "/Users/joycewilliamslt/Documents/GitHub/Oxford_Maths01/trial_animation.gif"
trial_ani.save(output_path, writer="pillow")
print(f"Saved animation to {output_path}")
# to view the gif you can drag the .gif to your safari browser icon/chrome and it should 'animate' there


# file path: "/Users/joycewilliamslt/Documents/GitHub/Oxford_Maths01/trial_animation.gif"
