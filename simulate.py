import numpy as np

# Unknown value
epsilon = 0.3

# Success rate of A
pA = 0.5
# Success rate of B
pB = 0.5 + epsilon

# Simple belief-updating conditionalization using Baye's rule
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

def jeffreyConditionalization():
    return

# Base Agent class
class Agent:
    def __init__(self, 
            initial_credence = None,
            conditionalization = 'strict'
        ):
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
        # Most recent pull result, used for sharing with other agents 
        self.pullResult = (None,None)
        # List of neighbors
        self.neighbors = []
        # Agents know their own number (unless they're not in a network,
        # in which case they are just an orphan)
        self.number = '(orphan)'

    def __str__(self):
        return f'AGENT {self.number}\n'+\
               f'Credence: {self.credence}\n' +\
               f'Neighbors: {self.neighbors}\n' +\
               f'Last pull: {self.pullResult}'

    # Update credence given the result of a B arm pull
    def update_credence(self, result, verbose):
        self.credence = self.conditionalization(self.credence, result)
        if verbose:
            print(f'\tNew credence: {self.credence}')

    # Choose an arm to pull, pull it, and update credence if B was pulled
    def update_on_self(self, verbose):
        # Select arm based on credence
        arm = 'A' if self.credence < 0.5 else 'B'
        # Pull the arm
        armSuccessRate = pA if arm == 'A' else pB
        result = 'success' if np.random.rand() < armSuccessRate else 'fail'
        if verbose:
            print(f'\tPulling arm {arm}')
            print(f'\tResult: {result}')
        # If we pulled arm B, we have evidence to update our credence for B
        if arm == 'B':
            self.update_credence(result, verbose)
        # Update pullResult to share with other agents
        self.pullResult = (arm,result)

    # Fetch pull results from all neighbors and update credence
    def update_on_neighbors(self):
        # Get the round's pull results from each neighbor
        for neighbor in self.neighbors:
            arm,result = neighbor.pullResult
            # Update self credence if neighbor pulled B
            if arm == 'B':
                self.update_credence(result)

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
            # Build it out!
            self.buildFromEdges()
        else:
            raise f'Undefined structure: {self.structure}. Try: complete'
            

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

    # Run an update round
    def update(self, verbose=False):
        if verbose:
            print('========[SELF-UPDATING]============')

        # First, have every agent run their self updates
        for i in range(self.n_agents):
            if verbose:
                print(f'AGENT {i}:')
            self.agents[i].update_on_self(verbose)
        if verbose:
            print('========[NEIGHBOR-UPDATING]============')
        # Then, have agents incorporate neighbors results into their credence
        for i in range(self.n_agents):
            if verbose:
                print(f'AGENT {i}:')
            for n in self.agents[i].neighbors:
                nPull, nResult = self.agents[n].pullResult
                if nPull == 'B':
                    if verbose:
                        print(f"\tUpdating credence on Agent {n}'s result")
                    self.agents[i].update_credence(nResult, verbose)
        if verbose:
            print('')



def main():
    net = EpistemicNetwork([
            Agent(),
            Agent(),
            Agent(),
            Agent(),
            Agent(),
            Agent()
          ], structure='complete')
    print(net)
    for i in range(100):
        net.update()
    print(net)


if __name__ == '__main__':
    main()
