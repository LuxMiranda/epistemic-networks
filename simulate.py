import numpy as np
import epnets as ep

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

    def run(self, m_mistrust):
        agents = make_prepolarized_agents(
                self.n_agents, 
                self.epsilon, 
                self.n_pulls, 
                m_mistrust)
        net = ep.simulate(agents=agents, net_structure='complete')
        print(net.to_dict())


def run_replications():
    sim = Weatherall_OConnor_2021_Fig_3()
    sim.run(1)
    return

def main():
    run_replications()
    return


if __name__ == '__main__':
    main()

