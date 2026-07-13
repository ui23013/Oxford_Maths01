import networkx as nx
import matplotlib.pyplot as plt
import EoN as eon

N, k= 500, 4 # number of nodes, node degree
beta, gamma = 0.4, 0.1 # transmission, recovery rates respectively
# not setting initial infected as it raises error if both this and rho are set
rho = 0.05 # fraction of initially infected
tmin, tmax = 0, 1000 # start and end time of sim

g = nx.random_regular_graph(k, N, seed=18) # create graph

# visualise graph
plt.figure(figsize=(6, 6))
nx.draw(g, node_size=20)
plt.title('Random Regular Graph')
plt.show()

# get arrays for epidemic dynamic plots from graph g
t, S, I, R = eon.fast_SIR(g, beta, gamma, initial_infecteds=None, initial_recovereds=None, rho=rho,
            tmin=tmin, tmax=tmax)


# plot function
def plot_fast_sir(g, t, s, i, r):
    plt.plot(t, s, label='Susceptible')
    plt.plot(t, i, label='Infected')
    plt.plot(t, r, label='Recovered')

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.show()


# plot fast sir dynamics for graph g
plot_fast_sir(g, t, S, I, R)

# run fast_SIR again, but this time to obtain simulation_investigation object,
# https://epidemicsonnetworks.readthedocs.io/en/latest/functions/EoN.fast_SIR.html
sim = eon.fast_SIR(g, beta, gamma, initial_infecteds=None, initial_recovereds=None, rho=rho,
            tmin=tmin, tmax=tmax, return_full_data=True)
ani=sim.animate(ts_plots=['I', 'SIR'], node_size=4)
# https://epidemicsonnetworks.readthedocs.io/en/latest/examples/SIR_display.html
plt.show()
