import matplotlib.pyplot as plt
import numpy as np
# import sim function from relevant py
from incomplete_info_network1 import sim_single_intervention
from plotting_style import set_plot_style

set_plot_style()

# plt.style.use('bmh')
#
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
#     "axes.labelsize": 12,
#     "axes.titlesize": 14,
#     "legend.fontsize": 12,
#     "grid.color": "0.85",  # light grey grid lines
#     "axes.facecolor": "white",
#     "figure.facecolor": "white",
#     "axes.edgecolor": "black",
#     "grid.linestyle": ":",
#     "grid.linewidth": 0.7
# })

# define test params
N=2500
k=4 # middle of ranges we are currently testing
beta, gamma = 0.4, 0.25 # middle of ranges we are currently testing
rho = 0.05

tau = 10
p, q = 0.75, 0.25
sim_duration = 250

# num of repeats to test
num_repeats = [5, 25, 50, 75, 100, 150, 200, 500]

# first test: num_repeats = [5, 10, 25, 50, 75, 100, 125, 150, 200]
# initialise lists for stats
means = []
stds = []
std_errors = []

for val in num_repeats:
    # initialise list to store measured quantity
    epidemic_sizes = []

    infected_end = []
    for seed in range(val):
        _, _, final_infected, _, total_infected, _ = sim_single_intervention(N=N, k=k, beta=beta, gamma=gamma, rho=rho,
            tau=tau, p=p, q=q, sim_duration=sim_duration)

        epidemic_sizes.append(total_infected)
        infected_end.append(final_infected)

    print("Mean final infected:", np.mean(infected_end))

    epidemic_sizes = np.array(epidemic_sizes)

    # stats to append to lists
    mean = np.mean(epidemic_sizes)
    means.append(mean)
    sigma = np.std(epidemic_sizes, ddof=1)
    stds.append(sigma)
    std_err = sigma / np.sqrt(val)
    std_errors.append(std_err)

    print(f"{val:3d} repeats, Mean = {mean:.2f}, SD = {sigma:.2f}, SE = {std_err:.3f}")

# plot that convergence graph!
ref_line = std_errors[0]* np.sqrt(num_repeats[0])/np.sqrt(num_repeats)
plt.figure(figsize=(6,4))
plt.plot(num_repeats, std_errors, 'o-', color='gold', label='Measured SE')
plt.plot(num_repeats, ref_line, '--', color='cadetblue', label=r'Reference $\propto 1/\sqrt{n}$')
plt.xlabel('Number of repeats')
plt.ylabel('SEM')
plt.title(r'Convergence of Mean Final Epidemic Size, $R_\infty$')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()






