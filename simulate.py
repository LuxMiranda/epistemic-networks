import numpy as np

EPSILON = 0.1

# Strength of mistrust
MISTRUST = 2

# Number of arm pulls per round (n)
N_PULLS = 1

# Success rate of A
pA = 0.5
# Success rate of B
pB = 0.5 + EPSILON


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
    pEnotH = (pA if result == 'success' else 1-pA) # P(E|~H)
    pnotH  = 1 - pH                                # P(~H)
    return (pEH*pH)/(pEH*pH + pEnotH*pnotH)

def other(result):
    return 'fail' if result == 'success' else 'success'

# Equation (1) in Weatherall & O'Connor 2021
# P_f(E)(d) = max({1 - d * m * (1 - P_i(E)), 0})
def mistrust(result, diff):
    PiE = (pB if result == 'success' else 1-pB) # P_i(E)
    return np.max(1 - (diff * MISTRUST * (1 - PiE)), 0)

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
    def __init__(self, initial_credence=None, conditionalization='jeffrey'):
        # Credence := Agent's belief of the probability that B is correct
        if initial_credence:
            # Set it to this if specified
            self.credence = initial_credence
        else:
            # Otherwise randomly initialize along a uniform distribution
            self.credence = np.random.rand()

        # Belief-updating function
        if conditionalization == 'strict':
            self.conditionalization = strictConditionalization
        elif conditionalization == 'jeffrey':
            self.conditionalization = jeffreyConditionalization

        # Most recent pull results, used for sharing with other agents 
        self.pullResults = [(None,None) for i in range(N_PULLS)]
        # List of neighbors
        self.neighbors = []
        # Agents know their own number (unless they're not in a network,
        # in which case they are just an orphan)
        self.number = '(orphan)'
        self.type   = 'AGENT'

    def __str__(self):
        return f'AGENT {self.number}: {self.type}\n'+\
               f'Credence: {self.credence}\n' +\
               f'Neighbors: {self.neighbors}\n'
               #f'Last pulls: {self.pullResults}'

    # Update credence given the result of a B arm pull
    def update_credence(self, result, diff, verbose):
        self.credence = self.conditionalization(self.credence,result,diff=diff)
        if verbose:
            print(f'\tNew credence: {self.credence}')

    # Choose an arm to pull, pull it, and update credence if B was pulled
    def update_on_self(self, verbose):
        # Reset pull results
        self.pullResults = []
        for i in range(N_PULLS):
            # Select arm based on credence
            arm = 'A' if self.credence < 0.5 else 'B'
            # Pull the arm
            armSuccessRate = pA if arm == 'A' else pB
            result = 'success' if np.random.rand() < armSuccessRate else 'fail'
            # If we pulled arm B, we have evidence to update our credence for B
            if arm == 'B':
                self.update_credence(result, 0, verbose)
            # Update pullResults to share with other agents
            self.pullResults.append((arm,result))


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
    def __init__(self, agents, edges=None, structure=None):
        self.agents    = agents
        self.n_agents  = len(agents)
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

    
    # Have agent i update their credence using neighbor's evidence
    def neighborUpdate(self, i, neighbor, verbose):
        for nPull, nResult in self.agents[neighbor].pullResults:
            if nPull == 'B':
                if verbose:
                    print(f"\tUpdating credence on Agent {n}'s result")
                diff = np.abs(
                        self.agents[i].credence - self.agents[neighbor].credence
                        )
                self.agents[i].update_credence(nResult, diff, verbose)

    # Run an update round
    def update(self, verbose=False):
        # First, have every agent run their self updates
        self.selfUpdates(verbose)
        # Then, have agents incorporate neighbors' results into their credence
        self.neighborUpdates(verbose)
        if verbose:
            print('')



def main():
    global N_PULLS
    N_PULLS = 10
    net = EpistemicNetwork([
            Agent(),
            Agent(),
            Agent(),
            Agent(),
            Agent()
          ], structure='cycle')
    print(net)
    for i in range(10):
        net.update()
    print(net)

if __name__ == '__main__':
    main()
