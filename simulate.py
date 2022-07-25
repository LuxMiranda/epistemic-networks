import epnets as ep
import numpy as np
from multiprocessing import Pool, cpu_count
import os

###########################
### GLOBAL DEFINITIONS ###
#########################

THREADS  = cpu_count() - 2
WOC_FIG3_PATH = 'results_and_figures/weatherall_oconnor_2021/fig_3/results.csv'
WOC_FIG9_PATH = 'results_and_figures/weatherall_oconnor_2021/fig_9/results.csv'

EXP1_PATH = lambda g : 'results_and_figures/experiment1/results-{}.csv'.format(g)
EXP2_PATH = lambda g : 'results_and_figures/experiment2/results-{}.csv'.format(g)

def reset_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'w') as f:
        f.write('m_mistrust,outcome,credences\n')


##########################################
### WEATHERALL & O'CONNOR REPLICATION ###
########################################

def run_weatherall_oconnor_2021_fig_3(m):
    n_repetitions = 100
    for i in range(n_repetitions):
        print(f'weatherall_oconnor_2021/fig_3: Mistrust {m} run {i}')
        # Agents start out already polarized on their first credence
        agents = [ep.Agent(n_pulls=50,initial_credences=[0.0, np.random.rand()]) for _ in range(10)]
        agents += [ep.Agent(n_pulls=50,initial_credences=[1.0, np.random.rand()]) for _ in range(10)]
        ep.simulate(agents, m_mistrust=m, results_file=WOC_FIG3_PATH, 
                    epsilon=0.01, antiupdating=True)

def run_weatherall_oconnor_2021_fig_9(m):
    n_repetitions = 100
    for i in range(n_repetitions):
        print(f'weatherall_oconnor_2021/fig_9: Mistrust {m} run {i}')
        agents = ep.make_agents(n_agents=10, n_credences=3, n_pulls=10)
        ep.simulate(agents, m_mistrust=m, results_file=WOC_FIG9_PATH,
                    epsilon=0.2, antiupdating=True)

def weatherall_oconnor_2021_fig_3():
    reset_file(WOC_FIG3_PATH)
    with Pool(THREADS) as p:
        p.map(run_weatherall_oconnor_2021_fig_3, np.linspace(0.1,4.0,num=50))

def weatherall_oconnor_2021_fig_9():
    reset_file(WOC_FIG9_PATH)
    with Pool(THREADS) as p:
        p.map(run_weatherall_oconnor_2021_fig_9, np.linspace(0.1,4.0,num=50))

#####################
### EXPERIMENT 1 ###
###################

def run_experiment_1(group, m):
    n_repetitions = 100
    structure = 'recommender_only' if group == 'rec' else 'complete'
    for i in range(n_repetitions):
        print(f'experiment 1 {group}: Mistrust {m} run {i}')
        agents = ep.make_agents(
                    n_agents=10, 
                    n_credences=3, 
                    n_pulls=10
                    )
        ep.simulate(agents, 
                m_mistrust=m,
                results_file=EXP1_PATH(group),
                epsilon=0.1,
                antiupdating=True,
                n_recommendations=3,
                network_structure=structure,
                recommend='similar'
                )

def run_experiment_1_control(m):
    run_experiment_1('control', m)

def run_experiment_1_rec(m):
    run_experiment_1('rec', m)

def experiment_1_control():
    reset_file(EXP1_PATH('control'))
    with Pool(THREADS) as p:
        p.map(run_experiment_1_control, np.linspace(0.1, 4.0, num=50))

def experiment_1_rec():
    reset_file(EXP1_PATH('rec'))
    with Pool(THREADS) as p:
        p.map(run_experiment_1_rec, np.linspace(0.1, 4.0, num=50))

def experiment_1():
    #experiment_1_control()
    experiment_1_rec()

#####################
### EXPERIMENT 2 ###
###################

def run_experiment_2(group, m):
    n_repetitions = 100
    structure = 'recommender_only' if group == 'rec' else 'complete'
    for i in range(n_repetitions):
        print(f'experiment 2 {group}: Mistrust {m} run {i}')
        agents = ep.make_agents(
                    n_agents=10, 
                    n_credences=3, 
                    n_pulls=10
                    )
        ep.simulate(agents, 
                m_mistrust=m,
                results_file=EXP2_PATH(group),
                epsilon=0.1,
                antiupdating=True,
                n_recommendations=3,
                network_structure=structure,
                recommend='dissimilar'
                )

def run_experiment_2_control(m):
    run_experiment_2('control', m)

def run_experiment_2_rec(m):
    run_experiment_2('rec', m)

def experiment_2_control():
    reset_file(EXP2_PATH('control'))
    with Pool(THREADS) as p:
        p.map(run_experiment_2_control, np.linspace(0.1, 4.0, num=50))

def experiment_2_rec():
    reset_file(EXP2_PATH('rec'))
    with Pool(THREADS) as p:
        p.map(run_experiment_2_rec, np.linspace(0.1, 4.0, num=50))

def experiment_2():
    #experiment_2_control()
    experiment_2_rec()


def main():
    #weatherall_oconnor_2021_fig_3()
    #weatherall_oconnor_2021_fig_9()
    #experiment_1()
    experiment_2()

if __name__ == '__main__':
    main()

