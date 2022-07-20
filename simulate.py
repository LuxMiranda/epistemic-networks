import numpy as np
import epnets as ep
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count, Lock

# Leave a couple of threads for system functions
N_THREADS = cpu_count() - 2
MUTEX = Lock()

def make_prepolarized_agents(n_agents, n_credences, epsilon, n_pulls, m_mistrust):
    half = int(float(n_agents)/2.0)
    initAgents = lambda x : [
                     ep.Agent(
                         initial_credences=[x, np.random.rand()],
                         epsilon=epsilon,
                         n_pulls=n_pulls,
                         n_credences=n_credences,
                         m_mistrust=m_mistrust
                     ) for _ in range(half)
    ]

    return initAgents(0.0) + initAgents(1.0)

def log_result(net, results_file):
    with MUTEX:
        with open(results_file, 'a') as f:
            f.write(net.to_bare_csv_line())


class Weatherall_OConnor_2021_Fig_3():
    def __init__(self):
        timestamp = datetime.now().strftime('%Y%m%d-%H:%M:%S')
        self.results_file = \
            f'results_and_figures/weatherall_oconnor_2021_fig_3/results_{timestamp}.csv'
        self.n_agents      = 10
        self.n_credences   = 2
        self.epsilon       = 0.01
        self.n_pulls       = 50

    def run(self, m_mistrust):
        print(f'Simulating Weatherall & O\'Connor (2021) Fig. 3 with m_mistrust={m_mistrust}')
        agents = make_prepolarized_agents(
                self.n_agents, 
                self.n_credences,
                self.epsilon, 
                self.n_pulls, 
                m_mistrust)
        net = ep.simulate(agents=agents, net_structure='cycle')
        log_result(net, self.results_file)

    def replicate(self):
        with Pool(N_THREADS) as p:
            ms = [1, 2, 3, 4, 5] * 10
            p.map(self.run, ms)


def run_replications():
    sim  = Weatherall_OConnor_2021_Fig_3()
    sim.replicate()
    return

def main():
    run_replications()
    return


if __name__ == '__main__':
    main()

