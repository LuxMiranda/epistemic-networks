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
            # Randomly initialize credence along a uniform distribution
            initial_credence=np.random.rand(),
            conditionalization = 'strict'
        ):
        # Credence := Agent's belief of the probability that B is correct
        self.credence = initial_credence
        # Belief-updating function
        if conditionalization == 'strict':
            self.conditionalization = strictConditionalization
        elif conditionalization == 'jeffrey':
            self.conditionalization = jeffreyConditionalization
        # Most recent pull result, used for sharing with other agents 
        self.pullResult = (None,None)
        print(f'Initial credence: {self.credence}')
            
    # Update credence given the result of a B arm pull
    def update_credence(self, result):
        self.credence = self.conditionalization(self.credence, result)
        print(f'New credence:{self.credence}')

    # Choose an arm to pull, pull it, and update credence if B was pulled
    def update_on_self(self):
        # Select arm based on credence
        arm = 'A' if self.credence < 0.5 else 'B'
        # Pull the arm
        armSuccessRate = pA if arm == 'A' else pB
        result = 'success' if np.random.rand() < armSuccessRate else 'fail'
        print(f'Pulling arm {arm}')
        print(f'Result: {result}')
        # If we pulled arm B, we have evidence to update our credence for B
        if arm == 'B':
            self.update_credence(result)
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

def main():
    a = Agent()
    a.update_on_self()
    a.update_on_self()
    a.update_on_self()
    a.update_on_self()
    a.update_on_self()

if __name__ == '__main__':
    main()
