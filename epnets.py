import numpy as np

# TODO: Some serious notation rectification
# P_f(~E) is misleading (it should just be the complement of P_f(E) 
# The notation between Bayes' rule and Weatherall & O'Connors notation is
# also incompatible (e.g. Bayes' P(E|H) is the same as W&O's P_i(E))

# Simple belief-updating conditionalization using Bayes' rule
#                 P(E|H)P(H)
# P(H|E) = -------------------------
#          P(E|H)P(H) + P(E|~H)P(~H)
def strictConditionalization(credence, result, diff, pA, pB, m_mistrust):
    # Update our credence for B in light of the new evidence with Bayes rule
    pEH    = (pB if result == 'success' else 1-pB) # P(E|H)
    pH     = credence                              # P(H)
    pEnotH = (pA if result == 'success' else 1-pA) # P(E|~H)
    pnotH  = 1 - pH                                # P(~H)
    return (pEH*pH)/(pEH*pH + pEnotH*pnotH)

def other(result):
    return 'fail' if result == 'success' else 'success'

# Equation (1) in Weatherall & O'Connor 2021
# P_f(E)(d) = max({1 - d * m * (1 - P_i(E)), 0})
def mistrust(result, diff, pB, m_mistrust):
    PiE = (pB if result == 'success' else 1-pB) # P_i(E)
    return np.max(1 - (diff * m_mistrust * (1 - PiE)), 0)

# Jeffrey's rule:
# P_f(H) = P_i(H|E) * P_f(E) + P_i(H|~E) * P_f(~E)
def jeffreyConditionalization(credence, result, diff, pA, pB,m_mistrust):
    piHE    = strictConditionalization(credence, result, diff, pA, pB, m_mistrust)        # P_i(H|E)
    piHnotE = strictConditionalization(credence, other(result), diff, pA, pB, m_mistrust) # P_i(H|~E)
    pfE     = mistrust(result, diff, pB, m_mistrust)                                      # P_f(E)
    pfnotE  = 1 - pfE                                                                     # P_f(~E)
    return (piHE*pfE) + (piHnotE*pfnotE)


# Base Agent class
class Agent:
    def __init__(self, 
            initial_credences=None,
            conditionalization='jeffrey',
            n_credences=2,
            epsilon=0.01,
            n_pulls=50,
            m_mistrust=2,
            pA=0.5):
        # Credence := Agent's belief of the probability that B is correct
        if initial_credences:
            # Set it to this if specified
            self.credences = initial_credences
        else:
            # Otherwise randomly initialize along a uniform distribution
            self.credences = [np.random.rand() for _ in range(n_credences)]

        # Belief-updating function
        if conditionalization == 'strict':
            self.conditionalization = strictConditionalization
        elif conditionalization == 'jeffrey':
            self.conditionalization = jeffreyConditionalization

        # Most recent pull results, used for sharing with other agents 
        self.pullResults = [
            [(None,None) for _ in range(n_pulls)] for _ in range(n_credences)]
        # List of neighbors
        self.neighbors = []
        # Agents know their own number (unless they're not in a network,
        # in which case they are just an orphan)
        self.number = '(orphan)'
        self.type   = 'AGENT'
        # Parameters
        self.epsilon    = epsilon
        self.n_pulls    = n_pulls
        self.m_mistrust = m_mistrust
        self.pA         = pA
        self.pB         = pA + epsilon
        self.n_pulls    = n_pulls
        self.n_credences= n_credences

    def __str__(self):
        return f'AGENT {self.number}: {self.type}\n'+\
               f'Credences: {self.credences}\n' +\
               f'Neighbors: {self.neighbors}\n' +\
               f'Epsilon: {self.epsilon}\n'
               #f'Last pulls: {self.pullResults}'

    # Update credence given the result of a B arm pull
    def update_credence(self, i, result, diff, verbose):
        self.credences[i] = self.conditionalization(
                                    self.credences[i],
                                    result,
                                    diff,
                                    self.pA,
                                    self.pB,
                                    self.m_mistrust)
        if verbose:
            print(f'\tNew credence for hypothesis {i}: {self.credences[i]}')

    # Choose an arm to pull, pull it, and update credence if B was pulled
    def update_on_self(self, verbose):
        # Reset pull results
        self.pullResults = []
        # For each credence
        for i in range(self.n_credences):
            self.pullResults.append([])
            # Test the arm N_PULLS times
            for _ in range(self.n_pulls):
                # Select arm based on credence
                arm = 'A' if self.credences[i] < 0.5 else 'B'
                if arm == 'B':
                # Pull the arm
                    armSuccessRate = self.pA if arm == 'A' else self.pB
                    result = 'success' if np.random.rand() < armSuccessRate else 'fail'
                    self.update_credence(i, result, 0, verbose)
                    # Update pullResults to share with other agents
                    self.pullResults[i].append((arm,result))


# Scientist behavior is implemented in the default agent class
class Scientist(Agent):
    def __init__(self, initial_credence=None):
        Agent.__init__(self, initial_credence=initial_credence)
        self.type = 'Scientist'


# Policymakers work the same as scientists, but don't collect their own
# evidence.
class Policymaker(Agent):
    def __init__(self, initial_credence=None):
        Agent.__init__(self, initial_credence=initial_credence)
        self.type = 'Policymaker'

    # Policymakers don't gather evidence
    def update_on_self(self):
        pass
 

class EpistemicNetwork:
    # Initialize with a list of agents and specified structure
    def __init__(self, 
            agents, 
            edges=None, 
            structure='complete'):
        # Agents
        self.agents    = agents
        self.n_agents  = len(agents)
        # Network
        self.edges     = edges
        self.structure = structure
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
        else:
            raise f'Undefined structure: {self.structure}. Try: complete, cycle'
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

    # Compute the trust distance between agents i and j
    # as the euclidean distance between their credence vectors
    def distance(self, i, j):
        x = np.array(self.agents[i].credences)
        y = np.array(self.agents[j].credences)
        return np.linalg.norm(x-y)

    # Have agent i update their credence using neighbor's evidence
    def neighborUpdate(self, i, neighbor, verbose):
        if verbose:
            print(f"\tUpdating credence on Agent {n}'s results")
        for c in range(self.agents[i].n_credences):
            for nPull, nResult in self.agents[neighbor].pullResults[c]:
                if nPull == 'B':
                    diff = self.distance(i, neighbor)
                    self.agents[i].update_credence(c, nResult, diff, verbose)

    # Run an update round
    def update(self, verbose=False):
        # First, have every agent run their self updates
        self.selfUpdates(verbose)
        # Then, have agents incorporate neighbors' results into their credence
        self.neighborUpdates(verbose)
        if verbose:
            print('')

    def to_dict(self):
        return {
            'epsilon'    : self.agents[0].epsilon,
            'n_pulls'    : self.agents[0].n_pulls,
            'n_agents'   : self.n_agents,
            'm_mistrust' : self.agents[0].m_mistrust,
            'agent_credences' : [self.agents[i].credences for i in range(self.n_agents)]
        }

    def to_bare_csv_line(self):
        data = self.to_dict()
        return '{},{},{},{},{}\n'.format(
            data['epsilon'],
            data['n_pulls'],
            data['n_agents'],
            data['m_mistrust'],
            data['agent_credences']
        )


def decided(agent):
    return False not in [c < 0.5 or c > 0.99 for c in agent.credences]

def finished(net):
    return False not in [decided(a) for a in net.agents]

def simulate(epsilon=0.01, # Uncertainty of hypothesis B's superiority
             n_pulls=50,   # Number of arm pulls per round
             n_agents=20,  # Number of agents
             m_mistrust=2, # Mistrust parameter
             agents=None,  # Manual list of agents (overrides n_agents)
             net_structure='complete'): # Network structure

    # If agents list isn't manually passed, create n_agents default agents
    if not agents:
        agents = [Agent(epsilon=epsilon,n_pulls=n_pulls,m_mistrust=m_mistrust)
                for _ in range(n_agents)]

    net = EpistemicNetwork(agents, structure=net_structure)

    while not finished(net):
        net.update()

    return net

