import numpy as np
from heapq import nlargest
from operator import itemgetter
from multiprocessing import Pool, Lock

EPSILON = 0.1

# Strength of mistrust
MISTRUST = 2

# Number of arm pulls per round (n)
N_PULLS = 1

# Number of different credences/beliefs
N_CREDENCES = 2

# Success rate of A
pA = 0.5
# Success rate of B
pB = 0.5 + EPSILON

ANTIUPDATING = True

MUTEX = Lock()

# TODO: Some serious notation rectification
# P_f(~E) is misleading (it should just be the complement of P_f(E) 
# The notation between Bayes' rule and Weatherall & O'Connors notation is
# also incompatible (e.g. Bayes' P(E|H) is the same as W&O's P_i(E))

# Simple belief-updating conditionalization using Bayes' rule
#                 P(E|H)P(H)
# P(H|E) = -------------------------
#          P(E|H)P(H) + P(E|~H)P(~H)
def strictConditionalization(credence, result):
    # Update our credence for B in light of the new evidence with Bayes rule
    pEH    = (pB if result == 'success' else 1-pB) # P(E|H)
    pH     = credence                              # P(H)
    pEnotH = 1 - pEH                               # P(E|~H)
    pnotH  = 1 - pH                                # P(~H)
    return (pEH*pH)/(pEH*pH + pEnotH*pnotH)

def other(result):
    return 'fail' if result == 'success' else 'success'

# Equation (1) in Weatherall & O'Connor 2021
# P_f(E)(d) = max({1 - d * m * (1 - P_i(E)), 0})
def mistrust(result, diff):
    PiE = (pB if result == 'success' else 1-pB) # P_i(E)
    minimum = 0.0 if ANTIUPDATING else PiE
    return np.max([1.0 - (diff * MISTRUST * (1.0 - PiE)), minimum])

# Jeffrey's rule:
# P_f(H) = P_i(H|E) * P_f(E) + P_i(H|~E) * P_f(~E)
def jeffreyConditionalization(credence, result, diff=0):
    piHE    = strictConditionalization(credence, result)        # P_i(H|E)
    piHnotE = strictConditionalization(credence, other(result)) # P_i(H|~E)
    pfE     = mistrust(result, diff)                            # P_f(E)
    pfnotE  = 1 - pfE                                           # P_f(~E)
    return (piHE*pfE) + (piHnotE*pfnotE)


# Base Agent class
class Agent:
    def __init__(self, initial_credences=None,
                       n_credences=2,
                       n_pulls=50,
                       conditionalization='jeffrey'):
        global N_CREDENCES, N_PULLS
        N_CREDENCES = n_credences
        N_PULLS = n_pulls
        # Credence := Agent's belief of the probability that B is correct
        if initial_credences:
            # Set it to this if specified
            self.credences = initial_credences
            N_CREDENCES = len(initial_credences)
        else:
            # Otherwise randomly initialize along a uniform distribution
            self.credences = [np.random.rand() for _ in range(N_CREDENCES)]

        # Belief-updating function
        if conditionalization == 'strict':
            self.conditionalization = strictConditionalization
        elif conditionalization == 'jeffrey':
            self.conditionalization = jeffreyConditionalization

        # Most recent pull results, used for sharing with other agents 
        self.pullResults = [
            [(None,None) for _ in range(N_PULLS)] for _ in range(N_CREDENCES)]
        # List of neighbors
        self.neighbors = []
        # List of permanent neighbors (for use in partial_recommender networks)
        self.permanent_neighbors = []
        # Agents know their own number (unless they're not in a network,
        # in which case they are just an orphan)
        self.number = '(orphan)'
        self.type   = 'AGENT'
        # By default, don't pull A. Unless it's important to for recommendations!
        self.pullA  = False

    def __str__(self):
        return f'AGENT {self.number}: {self.type}\n'+\
               f'Credences: {self.credences}\n' +\
               f'Neighbors: {self.neighbors}\n' +\
               f'Last pulls: {self.pullResults}'

    # Update credence given the result of a B arm pull
    def update_credence(self, i, result, diff, verbose):
        self.credences[i] = self.conditionalization(
                                    self.credences[i],
                                    result,
                                    diff=diff)
        if verbose:
            print(f'\tNew credence for hypothesis {i}: {self.credences[i]}')

    # Choose an arm to pull, pull it, and update credence if B was pulled
    def update_on_self(self, verbose):
        # Reset pull results
        self.pullResults = []
        # For each credence
        for i in range(N_CREDENCES):
            self.pullResults.append([])
            # Test the arm N_PULLS times
            for _ in range(N_PULLS):
                # Select arm based on credence
                arm = 'A' if self.credences[i] < 0.5 else 'B'
                # Pull the arm
                if arm == 'B' or self.pullA:
                    armSuccessRate = pA if arm == 'A' else pB
                    result = 'success' if np.random.rand() < armSuccessRate else 'fail'
                    # Update pullResults to share with other agents
                    self.pullResults[i].append((arm,result))
                    # Update credence if B was pulled
                    if arm == 'B':
                        self.update_credence(i, result, 0, verbose)


# Makes n_agents default agents
def make_agents(n_agents=50, n_credences=2, n_pulls=50):
    return [Agent(n_credences=n_credences, n_pulls=n_pulls) for _ in range(n_agents)]

class EpistemicNetwork:
    # Initialize with a list of agents and specified structure
    def __init__(self, 
            agents, 
            edges=[], 
            structure=None, 
            n_recommendations=3,
            n_partial_links=0,
            recommend='similar'
            ):
        self.agents    = agents
        self.n_agents  = len(agents)
        self.outcome   = None
        self.edges     = edges
        self.structure = structure
        self.recommender = 'recommender' in self.structure
        self.n_recommendations = n_recommendations
        self.n_partial_links   = n_partial_links
        self.recommend = recommend
        if recommend == 'similar':
            self.score = self.score_similarity
            self.pickScore = self.pickTop
        elif recommend == 'dissimilar':
            self.score = self.score_dissimilarity
            self.pickScore = self.pickTop
        elif recommend == 'random':
            self.score = self.score_random
            self.pickScore = self.pickTop
        elif recommend == 'one_similar':
            self.score = self.score_similarity
            self.pickScore = self.pickMixed
        elif recommend == 'pull_similar':
            self.score = self.score_pull_similarity
            self.pickScore = self.pickTop
            self.tellAgentsToPullA()
        elif recommend == 'pull_dissimilar':
            self.score = self.score_pull_similarity
            self.pickScore = self.pickTop
            self.tellAgentsToPullA()
        elif self.recommender:
            raise Exception('Invalid recommendation strategy. Try: similar, dissimilar, random, one_similar, pull_similar, pull_dissimilar')
        self.buildNetwork()

    def __str__(self):
        netStr = '========[NETWORK STATE]=======\n'
        for agent in self.agents:
            netStr += str(agent) + '\n---\n'
        return netStr
 

    # Construct the network
    def buildNetwork(self):
        # If a structure variable is defined
        if self.structure:
            # Build the network according to the specified structure
            self.buildFromStructure()
        else:
            # Otherwise, build the network from the manual list of edges
            self.buildFromEdges()

    # Construct the network according to a specified structure
    def buildFromStructure(self):
        if self.structure == 'complete':
            # Construct a complete list of all possible edges
            self.edges = [(i,j) for i in range(self.n_agents)\
                                for j in range(self.n_agents)\
                                if i!=j ]
        elif self.structure == 'cycle':
            highestIndex = self.n_agents - 1
            forward = [(i,i+1) for i in range(highestIndex)] +[(highestIndex,0)]
            backward = [(r,l) for (l,r) in forward]
            self.edges = forward + backward
        elif self.structure == 'recommender_only':
            # Init with recommended links
            self.addRecommendedLinks()
        elif self.structure == 'partial_recommender':
            self.addPartialLinks()
            self.addRecommendedLinks()
        else:
            raise f'Undefined structure: {self.structure}. Try: \
                    complete, cycle, recommender_only, cycle_recommender'
       # Build it out!
        self.buildFromEdges()
            

    # Construct the network from a manual list of edges
    def buildFromEdges(self):
        # For every agent
        for i in range(self.n_agents):
            # Tell the agent its number
            self.agents[i].number = i
            # Tell it about its neighbors
            for node,neighbor in self.edges:
                if node == i:
                    self.agents[i].neighbors.append(neighbor)


    # Have agents pull their arms and update themselves
    def selfUpdates(self, verbose):
        if verbose:
            print('========[SELF-UPDATING]============')
        for i in range(self.n_agents):
            if verbose:
                print(f'AGENT {i}:')
            self.agents[i].update_on_self(verbose)
        return

    # Have agents use neighbor's results to update their credences
    def neighborUpdates(self, verbose):
        if verbose:
             print('========[NEIGHBOR-UPDATING]============')
        for i in range(self.n_agents):
             if verbose:
                 print(f'AGENT {i}:')
             for neighbor in self.agents[i].neighbors:
                self.neighborUpdate(i, neighbor, verbose)

    def tellAgentsToPullA(self):
        for i in range(self.n_agents):
            self.agents[i].pullA = True

    # Compute the trust distance between agents i and j
    # as the euclidean distance between their credence vectors
    def distance(self, i, j):
        x = np.array(self.agents[i].credences)
        y = np.array(self.agents[j].credences)
        return np.linalg.norm(x-y)

    # Have agent i update their credence using neighbor's evidence
    def neighborUpdate(self, i, neighbor, verbose):
        if verbose:
            print(f"\tUpdating credence on Agent {neighbor}'s results")
        for c in range(N_CREDENCES):
            for nPull, nResult in self.agents[neighbor].pullResults[c]:
                if nPull == 'B':
                    diff = self.distance(i, neighbor)
                    self.agents[i].update_credence(c, nResult, diff, verbose)

    # Scores higher for agents with more similar pull scores
    def score_pull_similarity(self, agent_i,  agent_j):
        # If the last result is None, it's the first step. Return random.
        if self.agents[agent_i].pullResults[0][0][0] == None:
            return np.random.rand()

        # Get a point for each result that is the same
        total = 0
        for belief in range(N_CREDENCES):
            count  = lambda agent, arm, res : \
                len([(a,r) for (a,r) in self.agents[agent].pullResults[belief] if (a,r) == (arm,res)])
            counts = [(count(agent_i, a, r), count(agent_j, a, r))\
                    for a in 'AB' for r in ['success','failure']]
            total += np.sum([np.min([i,j]) for (i,j) in counts])
        return total

    # Opposite of pull similarity
    def score_pull_dissimilarity(self, agent_i, agent_j):
        return -1*self.score_pull_similarity(agent_i, agent_j)

    # Scores higher for agents more similar
    def score_similarity(self, agent_i, agent_j):
        return -1*self.distance(agent_i, agent_j)

    # Scores higher for agents more dissimilar
    def score_dissimilarity(self, agent_i, agent_j):
        return self.distance(agent_i, agent_j)

    # Random scores
    def score_random(self, agent_i, agent_j):
        return np.random.rand()

    # Create a list of agents scored with the recommendation criterion
    def scoreAgentsFor(self, agent_i):
        return [(agent_j, self.score(agent_i, agent_j)) 
                for agent_j in range(self.n_agents) 
                if agent_j != agent_i
                and agent_j not in self.agents[agent_i].permanent_neighbors]
                # ^ This last line is important for only recommending agents
                #   who are not already permanent neighbors in partial 
                #   recommender networks
    
    # Pick the top n_recommendations of those 
    def pickTop(self, scoredAgents):
        return [agent_j for agent_j,score in 
            nlargest(int(self.n_recommendations), scoredAgents, key=itemgetter(1))]

    # Pick the top scored agent, and n_recommendations-1 lowest scored agents
    def pickMixed(self, scoredAgents):
        top = max(scoredAgents, key=itemgetter(1))
        reversedScores = [(agent_j, -1*score) for agent_j, score in scoredAgents]
        bottoms = self.pickTop(reversedScores)[:-1]
        return [top[0]] + bottoms

    def getRecommendationsFor(self, agent_i):
        # Create a list of agents scored with the recommendation criterion
        scoredAgents = self.scoreAgentsFor(agent_i)
        # Pick n_recommendations of those 
        return self.pickScore(scoredAgents)

    # Get n_partial_links random neighbors of agent_i, who are not agent_i
    def getRandomNeighbors(self, agent_i):
        possible_neighbors = [j for j in range(self.n_agents) if j!=agent_i]
        return [possible_neighbors[np.random.randint(low=0, high=len(possible_neighbors))]
                for i in range(self.n_partial_links)]


    def addPartialLinks(self):
        # For every agent
        for agent_i in range(self.n_agents):
            # Randomly select n_partial_links other agents to add 
            # as permanent neighbors
            newNeighbors = self.getRandomNeighbors(agent_i)
            # Set these as the agent's permanent neighbors
            self.agents[agent_i].neighbors = [n for n in newNeighbors]
            self.agents[agent_i].permanent_neighbors = [n for n in newNeighbors]

    def addRecommendedLinks(self):
        # For every agent
        for agent_i in range(self.n_agents):
            # Select the n_recommendations recommended agents
            recommendations = self.getRecommendationsFor(agent_i)
            # Create links to them
            self.agents[agent_i].neighbors += recommendations
        return

    def removeLinks(self):
        # For every agent
        for agent_i in range(self.n_agents):
            # Reset their list of neighbors to the permanent ones
            # (If it is not a partial recommender network, the list of
            # permanent neighbors is simply empty [] )
            self.agents[agent_i].neighbors = [n for n in self.agents[agent_i].permanent_neighbors]
        return

    # Run an update round
    def update(self, verbose=False):
        # If a recommender is managing the network links
        if self.recommender:
            # Regenerate links
            self.removeLinks()
            self.addRecommendedLinks()
        # Have every agent run their self updates
        self.selfUpdates(verbose)
        # Then, have agents incorporate neighbors' results into their credence
        self.neighborUpdates(verbose)
        if verbose:
            print('')

    def to_bare_csv_line(self):
        return '{},{},{},{},{},{},"{}"\n'.format(MISTRUST, self.outcome,
                self.recommend,
                self.n_agents, self.n_recommendations, self.n_partial_links,
                [self.agents[i].credences for i in range(self.n_agents)])

# If any agent has a 0.5 <= credence <= 0.99, the simulation is not finished
def unfinished(net):
    for agent in net.agents:
        for credence in agent.credences:
            if 0.5 <= credence and credence <= 0.99:
                return True
    return False

def true_consensus(net):
    for agent in net.agents:
        for credence in agent.credences:
            if credence <= 0.99:
                return False
    return True

def false_consensus(net):
    for agent in net.agents:
        for credence in agent.credences:
            if credence >= 0.5:
                return False
    return True

def value_from_credence(credence):
    if credence > 0.99:
        return 'true'
    elif credence < 0.5:
        return 'false'
    else:
        raise 'Evaluating consensus on unfinished simulation'
        exit(1)

def mixed_consensus(net):
    # For each belief
    for belief in range(N_CREDENCES):
        # Look at the first agent's value
        agent_0_value = value_from_credence(net.agents[0].credences[belief])
        # For all other agents
        for agent in net.agents:
            # If even one other agent does not share that value, there is not
            # consensus.
            agent_i_value = value_from_credence(agent.credences[belief])
            if agent_0_value != agent_i_value:
                return False
    # If we made it here, all agents have formed a consensus on all beliefs
    # (whether or not they chose the true or false hypothesis)
    return True


def polarized(net):
    # By the time this function is called, we can count on:
    # 1. All agents having credences < 0.5 or > 0.99
    # 2. No consensus having been reached
    # All we need to do is check to see if the A agents have credences below
    # the proper polar boundary
    polar_boundary = 0.01 if ANTIUPDATING else 0.5
    for agent in net.agents:
        for credence in agent.credences:
            if credence < 0.5 and credence >= polar_boundary:
                return False
    return True

def check_if_finished(net):
    # First, do a quick check to see if credences are stable yet
    if unfinished(net):
        return False, None
    # Now, credences are possibly stable; evaluate which kind of outcome it is
    if true_consensus(net):
        return True, 'True consensus'
    if false_consensus(net):
        return True, 'False consensus'
    if mixed_consensus(net):
        return True, 'Mixed consensus'
    if polarized(net):
        return True, 'Polarization'
    # This false is called if agents haven't formed a consensus, but are still 
    # above the polar boundary
    return False, 'Transient polarization'

def simulate(agents, epsilon=0.01, m_mistrust=2, n_recommendations=2,
        network_structure='complete', recommend='similar',
        n_partial_links=0,
        results_file='results.csv', antiupdating=True):
    global EPSILON, MISTRUST, ANTIUPDATING
    EPSILON  = epsilon
    MISTRUST = m_mistrust
    ANTIUPDATING = antiupdating
    net = EpistemicNetwork(agents, structure=network_structure,
                           n_recommendations=n_recommendations,
                           recommend=recommend,
                           n_partial_links=n_partial_links)

    polarized_steps = 0
    max_polarized_steps = 5
    finished, outcome = False, None
    while not finished:
        net.update()
        finished, outcome = check_if_finished(net)
        if outcome == 'Transient polarization':
            polarized_steps += 1
        if outcome == 'Transient polarization' and polarized_steps >= max_polarized_steps:
            finished, outcome = True, 'Polarization'

    net.outcome = outcome

    with MUTEX:
        with open(results_file,'a') as f:
            f.write(net.to_bare_csv_line())
    
