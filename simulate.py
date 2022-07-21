import epnets as ep
import numpy as np
from multiprocessing import Pool, cpu_count

THREADS = cpu_count() - 2

def run_weatherall_oconnor_2021_fig_3(m):
    filename = 'results_and_figures/weatherall_oconnor_2021/fig_3/results.csv'
    n_repetitions = 100
    for i in range(n_repetitions):
        print(f'weatherall_oconnor_2021/fig_3: Mistrust {m} run {i}')
        # Agents start out already polarized on their first credence
        agents = [ep.Agent(n_pulls=50,initial_credences=[0.0, np.random.rand()]) for _ in range(10)]
        agents += [ep.Agent(n_pulls=50,initial_credences=[1.0, np.random.rand()]) for _ in range(10)]
        ep.simulate(agents, m_mistrust=m, results_file=filename, 
                    epsilon=0.01, antiupdating=True)


def run_weatherall_oconnor_2021_fig_9(m):
    filename = 'results_and_figures/weatherall_oconnor_2021/fig_9/results.csv'
    n_repetitions = 100
    for i in range(n_repetitions):
        print(f'weatherall_oconnor_2021/fig_9: Mistrust {m} run {i}')
        agents = ep.make_agents(n_agents=10, n_credences=3, n_pulls=10)
        ep.simulate(agents, m_mistrust=m, results_file=filename,
                    epsilon=0.2, antiupdating=True)


def weatherall_oconnor_2021_fig_3():
    with Pool(THREADS) as p:
        p.map(run_weatherall_oconnor_2021_fig_3, np.linspace(0.1,4.0,num=50))

def weatherall_oconnor_2021_fig_9():
    with Pool(THREADS) as p:
        p.map(run_weatherall_oconnor_2021_fig_9, np.linspace(0.1,4.0,num=50))

def main():
    #weatherall_oconnor_2021_fig_3()
    weatherall_oconnor_2021_fig_9()

if __name__ == '__main__':
    main()

