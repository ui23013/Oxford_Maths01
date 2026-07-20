from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
from incomplete_info_network1 import sim_single_intervention
from plotting_style import set_plot_style

set_plot_style()

N, k, beta, gamma, rho = 1000, 3, 0.4, 0.2, 0.05
tau, p, q = 5, 0.75, 0.25
sim_duration = 150
num_repeats = [25, 50, 75, 100, 150, 200, 500, 1000, 2000, 2500]


def run_sim(seed):
    """Wrapper function for parallel execution"""
    np.random.seed(seed)
    _, _, final_infected, _, total_infected, _ = sim_single_intervention(
        N=N, k=k, beta=beta, gamma=gamma, rho=rho,
        tau=tau, p=p, q=q, sim_duration=sim_duration)
    return total_infected, final_infected


if __name__ == '__main__':
    means, stds, std_errors = [], [], []

    for val in num_repeats:
        with Pool() as pool:
            results = pool.map(run_sim, range(val))

        epidemic_sizes, infected_end = zip(*results)
        epidemic_sizes = np.array(epidemic_sizes)
        infected_end = np.array(infected_end)

        print(f"Mean final infected: {np.mean(infected_end)}")

        mean = np.mean(epidemic_sizes)
        sigma = np.std(epidemic_sizes, ddof=1)
        std_err = sigma / np.sqrt(val)

        means.append(mean)
        stds.append(sigma)
        std_errors.append(std_err)

        print(f"{val:3d} repeats, Mean = {mean:.2f}, SD = {sigma:.2f}, SE = {std_err:.3f}")

    # Plotting code
    ref_line = std_errors[0] * np.sqrt(num_repeats[0]) / np.sqrt(num_repeats)
    plt.figure(figsize=(6, 5))
    plt.plot(num_repeats, std_errors, 'o-', color='goldenrod', label='Measured SEM')
    plt.plot(num_repeats, ref_line, '--', color='cadetblue', label=r'Reference $\propto 1/\sqrt{n}$')
    plt.xlabel(r'Number of repeats, $n_{\mathrm{rep}}$')
    plt.ylabel('SEM')
    # plt.title(r'Convergence of Mean Final Epidemic Size, $R_\infty$')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()








