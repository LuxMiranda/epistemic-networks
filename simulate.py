import numpy as np
import epnets as ep
import pandas as pd
from multiprocessing import Pool, cpu_count

# Leave a couple of threads for system functions
N_THREADS = cpu_count() - 2

def make_prepolarized_agents(n_agents, epsilon, n_pulls, m_mistrust):
    half = int(float(n_agents)/2.0)
    initAgents = lambda x : [
                     ep.Agent(
                         initial_credences=[x, np.random.rand()],
                         epsilon=epsilon,
                         n_pulls=n_pulls,
                         m_mistrust=m_mistrust
                     ) for _ in range(half)
    ]

    return initAgents(0.0) + initAgents(1.0)


class Weatherall_OConnor_2021_Fig_3():
    def __init__(self):
        self.results_file = 'results_and_figures/weatherall_oconnor_2021_fig_3/results.csv'
        self.n_agents = 20
        self.epsilon  = 0.01
        self.n_pulls  = 50
        self.n_repetitions = 50

    def run(self, m_mistrust):
        print(f'Simulating Weatherall & O\'Connor (2021) Fig. 3 with m_mistrust={m_mistrust}')
        agents = make_prepolarized_agents(
                self.n_agents, 
                self.epsilon, 
                self.n_pulls, 
                m_mistrust)
        net = ep.simulate(agents=agents, net_structure='complete')
        return net.to_dict()

    def replicate(self):
        with Pool(N_THREADS) as p:
            results = p.map(self.run, [1])
        df = pd.DataFrame(results)
        df.index.name = 'Run'
        return df


def run_replications():
    sim  = Weatherall_OConnor_2021_Fig_3()
    data = sim.replicate()
    print(data)
    return

def main():
    run_replications()
    return


if __name__ == '__main__':
    main()

