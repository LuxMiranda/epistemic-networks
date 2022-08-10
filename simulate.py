import epnets as ep
import numpy as np
from multiprocessing import Pool, cpu_count
import os
from tqdm import tqdm

###########################
### GLOBAL DEFINITIONS ###
#########################

THREADS  = cpu_count() - 2
WOC_FIG3_PATH = 'results_and_figures/weatherall_oconnor_2021/fig_3/results.csv'
WOC_FIG9_PATH = 'results_and_figures/weatherall_oconnor_2021/fig_9/results.csv'

EXP_PATH  = lambda i,g : 'results_and_figures/experiment{}/results-{}.csv'.format(i,g)
EXP1_PATH = lambda g : EXP_PATH(1,g)
EXP2_PATH = lambda g : EXP_PATH(2,g)
EXP3_PATH = lambda g : EXP_PATH(3,g)
EXP4_PATH = lambda g : EXP_PATH(4,g)


def reset_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'w') as f:
        f.write('m_mistrust,outcome,recommend,n_agents,n_recommendations,n_partial_links,credences\n')


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
    structure = 'complete'
    recommend = 'dissimilar'
    if group == 'rec':
        structure = 'recommender_only'
    elif group == 'cycle':
        structure = 'cycle'
    elif group == 'random_rec':
        structure = 'recommender_only'
        recommend = 'random'
    elif group == 'mixed_rec':
        structure = 'recommender_only'
        recommend = 'one_similar'
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
                recommend=recommend
                )

def run_experiment_2_control(m):
    run_experiment_2('control', m)

def run_experiment_2_rec(m):
    run_experiment_2('rec', m)

def run_experiment_2_cycle(m):
    run_experiment_2('cycle', m)

def experiment_2_control():
    reset_file(EXP2_PATH('control'))
    with Pool(THREADS) as p:
        p.map(run_experiment_2_control, np.linspace(0.1, 4.0, num=50))

def experiment_2_rec():
    reset_file(EXP2_PATH('rec'))
    with Pool(THREADS) as p:
        p.map(run_experiment_2_rec, np.linspace(0.1, 4.0, num=50))

def experiment_2_cycle():
    reset_file(EXP2_PATH('cycle'))
    with Pool(THREADS) as p:
        p.map(run_experiment_2_cycle, np.linspace(0.1, 4.0, num=50))

def run_experiment_2_randomrec(m):
    run_experiment_2('random_rec', m)

def experiment_2_randomrec():
    reset_file(EXP2_PATH('random_rec'))
    with Pool(THREADS) as p:
        p.map(run_experiment_2_randomrec, np.linspace(0.1, 4.0, num=50))

def run_experiment_2_mixed_rec(m):
    run_experiment_2('mixed_rec', m)

def experiment_2_mixed_rec():
    reset_file(EXP2_PATH('mixed_rec'))
    with Pool(THREADS) as p:
        p.map(run_experiment_2_mixed_rec, np.linspace(0.1, 4.0, num=50))
#####################
### EXPERIMENT 3 ###
###################

EXP3_PARAMS = {
    'random' : 
        { 'recommend'  : 'random',
           'structure' : 'recommender_only',
           'n_recs'    : 4,
           'n_agents'  : 20,
           'n_partial_links' : 0 },
    'most-similar' : 
        { 'recommend'  : 'similar',
           'structure' : 'recommender_only',
           'n_recs'    : 4,
           'n_agents'  : 20,
           'n_partial_links' : 0 },
    'partial' :
         { 'recommend'  : 'similar',
           'structure'  : 'partial_recommender',
           'n_recs'     : 2,
           'n_agents'  : 20,
           'n_partial_links' : 2 },
    'partial2' :
         { 'recommend'  : 'similar',
           'structure'  : 'partial_recommender',
           'n_recs'     : 3,
           'n_agents'  : 20,
           'n_partial_links' : 1 },
    'partial3' :
         { 'recommend'  : 'similar',
           'structure'  : 'partial_recommender',
           'n_recs'     : 0,
           'n_agents'   : 20,
           'n_partial_links' : 4 },
    'partial4' :
         { 'recommend'  : 'dissimilar',
           'structure'  : 'partial_recommender',
           'n_recs'     : 2,
           'n_agents'   : 20,
           'n_partial_links' : 2 },
    'pull_similar' :
          { 'recommend'  : 'pull_similar',
           'structure'  : 'recommender_only',
           'n_recs'     : 4,
           'n_agents'   : 20,
           'n_partial_links' : 0 },
    'pull_dissimilar' :
          { 'recommend'  : 'pull_dissimilar',
           'structure'  : 'recommender_only',
           'n_recs'     : 4,
           'n_agents'   : 20,
           'n_partial_links' : 0 },
}

def run_experiment_3(group, m):
    n_repetitions = 100
    for i in range(n_repetitions):
        print(f'experiment 3 {group}: Mistrust {m} run {i}')
        agents = ep.make_agents(
                    n_agents=EXP3_PARAMS[group]['n_agents'], 
                    n_credences=5, 
                    n_pulls=10
                    )
        ep.simulate(agents, 
                m_mistrust=m,
                results_file=EXP3_PATH(group),
                epsilon=0.2,
                antiupdating=True,
                n_recommendations=EXP3_PARAMS[group]['n_recs'],
                network_structure=EXP3_PARAMS[group]['structure'],
                recommend=EXP3_PARAMS[group]['recommend'],
                n_partial_links=EXP3_PARAMS[group]['n_partial_links']
                )

def run_experiment_3_random(m):
    run_experiment_3('random', m)

def experiment_3_control():
    reset_file(EXP3_PATH('random'))
    with Pool(THREADS) as p:
        p.map(run_experiment_3_random, np.linspace(0.1, 4.0, num=50))

def run_experiment_3_MS(m):
    run_experiment_3('most-similar', m)

def experiment_3_MS():
    reset_file(EXP3_PATH('most-similar'))
    with Pool(THREADS) as p:
        p.map(run_experiment_3_MS, np.linspace(0.1, 4.0, num=50))

def run_experiment_3_partial(m):
    run_experiment_3('partial', m)

def experiment_3_partial():
    reset_file(EXP3_PATH('partial'))
    with Pool(THREADS) as p:
        p.map(run_experiment_3_partial, np.linspace(0.1, 4.0, num=50))

def run_experiment_3_partial2(m):
    run_experiment_3('partial2', m)

def experiment_3_partial2():
    reset_file(EXP3_PATH('partial2'))
    with Pool(THREADS) as p:
        p.map(run_experiment_3_partial2, np.linspace(0.1, 4.0, num=50))


def run_experiment_3_partial3(m):
    run_experiment_3('partial3', m)

def experiment_3_partial3():
    reset_file(EXP3_PATH('partial3'))
    with Pool(THREADS) as p:
        p.map(run_experiment_3_partial3, np.linspace(0.1, 4.0, num=50))

def run_experiment_3_partial4(m):
    run_experiment_3('partial4', m)


def experiment_3_partial4():
    reset_file(EXP3_PATH('partial4'))
    with Pool(THREADS) as p:
        p.map(run_experiment_3_partial4, np.linspace(0.1, 4.0, num=50))


def run_experiment_3_pull_similar(m):
    run_experiment_3('pull_similar',m)

def experiment_3_pull_similar():
    reset_file(EXP3_PATH('pull_similar'))
    with Pool(THREADS) as p:
        p.map(run_experiment_3_pull_similar, np.linspace(0.1, 4.0, num=50))



def run_experiment_3_pull_dissimilar(m):
    run_experiment_3('pull_dissimilar',m)

def experiment_3_pull_dissimilar():
    reset_file(EXP3_PATH('pull_dissimilar'))
    with Pool(THREADS) as p:
        p.map(run_experiment_3_pull_dissimilar, np.linspace(0.1, 4.0, num=50))


##################
## EXPERIMENT 4 ##
##################

EXP4_PBAR = None

EXP4_PARAMS = {
    'vary_n_recs' : 
    {      'structure' : 'recommender_only',
           'n_recs'    : 4,
           'n_agents'  : 20,
           'n_partial_links' : 0 },
}



def run_experiment_4(group, m):
    n_repetitions = 100
    for recommend in ['random','similar']:
        for n_recs in [2,4,6,8,10]:
            for i in range(n_repetitions):
                EXP4_PBAR.update(1)
                agents = ep.make_agents(
                            n_agents=EXP4_PARAMS[group]['n_agents'], 
                            n_credences=5, 
                            n_pulls=10
                            )
                ep.simulate(agents, 
                        m_mistrust=m,
                        results_file=EXP4_PATH(group),
                        epsilon=0.2,
                        antiupdating=True,
                        n_recommendations=n_recs,
                        network_structure=EXP4_PARAMS[group]['structure'],
                        recommend=recommend,
                        n_partial_links=EXP4_PARAMS[group]['n_partial_links']
                        )


def run_experiment_4_vary_n_recs(m):
    run_experiment_4('vary_n_recs',m)

def experiment_4_vary_n_recs():
    reset_file(EXP4_PATH('vary_n_recs'))
    with Pool(THREADS) as p:
        p.map(run_experiment_4_vary_n_recs, np.linspace(0.1, 3.0, num=30))

def run_experiment_4_vary_n_agents(m):
    n_repetitions = 100
    for recommend in ['random','similar']:
        for n_agents in [4,12,20,32,56]:
            for i in range(n_repetitions):
                EXP4_PBAR.update(1)
                agents = ep.make_agents(
                            n_agents=n_agents,
                            n_credences=5, 
                            n_pulls=10
                            )
                ep.simulate(agents, 
                        m_mistrust=m,
                        results_file=EXP4_PATH('vary_n_agents'),
                        epsilon=0.2,
                        antiupdating=True,
                        n_recommendations=n_agents/4,
                        network_structure='recommender_only',
                        recommend=recommend,
                        n_partial_links=0
                        )



def experiment_4_vary_n_agents():
    reset_file(EXP4_PATH('vary_n_agents'))
    with Pool(THREADS) as p:
        p.map(run_experiment_4_vary_n_agents, np.linspace(0.1, 3.0, num=30))


############
### MAIN ###
############

def experiment_1():
    #experiment_1_control()
    experiment_1_rec()

def experiment_2():
    #experiment_2_control()
    #experiment_2_rec()
    #experiment_2_cycle()
    #experiment_2_randomrec()
    experiment_2_mixed_rec()

def experiment_3():
    #experiment_3_control()
    #experiment_3_MS()
    #experiment_3_partial()
    #experiment_3_partial2()
    #experiment_3_partial3()
    #experiment_3_partial4()
    #experiment_3_pull_similar()
    experiment_3_pull_dissimilar()

def experiment_4():
    global EXP4_PBAR
    EXP4_PBAR = tqdm(total=2*5*100*30,ascii=" ▖▘▝▗▚▞█")
    #exepriment_4_vary_n_recs()
    experiment_4_vary_n_agents()
    EXP4_PBAR.close()

# Note: Dear user,
#       I haven't actually tried running multiple experiments (or even sub-
#       experiments) at once. Rather embarrassingly, I'm afraid it wouldn't
#       work out given epnets.py's usage of global variables and this file's
#       usage of parallelism. When I tried to re-implement epnets.py to not use
#       global variables, I maddeningly got a x10 slowdown in simulation time
#       that I could not for the life of me figure out how to remedy. So, dear
#       user, this is simply a tragic case of me choosing to value performance
#       over code readability and usability. Please only run one sub-experiment
#       at a time (e.g., experiment_3_control()), lest your results go totally
#       whacky and non-deterministic.
#       Sincerely, ya girl,
#       Lux
def main():
    #weatherall_oconnor_2021_fig_3()
    #weatherall_oconnor_2021_fig_9()
    #experiment_1()
    #experiment_2()
    #experiment_3()
    experiment_4()

if __name__ == '__main__':
    main()

