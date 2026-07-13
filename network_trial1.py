import networkx as nx
import matplotlib.pyplot as plt
import EoN as eon

N, k= 500, 4 # number of nodes, node degree
beta, gamma = 0.5, 0.05 # transmission, recovery rates respectively
# not setting initial infected as it raises error if both this and rho are set
rho = 0.1 # fraction of initially infected
tmin, tmax = 0, 1000 # start and end time of sim

g = nx.random_regular_graph(k, N) # create graph

nx.draw(g) # see what graph looks like

t, S, I, R = eon.fast_SIR(g, beta, gamma, initial_infecteds=None, initial_recovereds=None, rho=rho,
            tmin=tmin, tmax=tmax)


def plot_fast_sir(g, t, s, i, r):
    plt.plot(t, s, label='Susceptible')
    plt.plot(t, i, label='Infected')
    plt.plot(t, r, label='Recovered')

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.show()


plot_fast_sir(g, t, S, I, R)