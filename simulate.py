import numpy as np

# Unknown value
epsilon = 0.3

# Success rate of A
pA = 0.5
# Success rate of B
pB = 0.5 + epsilon

# Simple belief-updating conditionalization using Baye's rule
def strictConditionalization(credence):
    # Select arm based on credence
    arm = 'A' if credence < 0.5 else 'B'
    # Pull the arm
    armSuccessRate = pA if arm == 'A' else pB
    result = 'success' if np.random.rand() < armSuccessRate else 'fail'
    print(f'Pulling arm {arm}')
    print(f'Result: {result}')
    # If we pulled arm A, we have no information to update our credence for B
    if arm == 'A':
        return credence
    # Else, if we pulled arm B
    else:
        # Update our credence for B in light of the new evidence with Bayes rule
        pEH    = (pB if result == 'success' else 1-pB) # P(E|H)
        pH     = credence                    # P(H)
        pEnotH = (pA if result == 'success' else 1-pA) # P(E|~H)
        pnotH  = 1 - pH                      # P(~H)
        return (pEH*pH)/(pEH*pH + pEnotH*pnotH)

def jefferyConditionalization():
    return

# Base Agent class
class Agent:
    def __init__(self):
        # Credence is randomly initialized along a uniform distribution
        self.credence = np.random.rand()
        self.conditionalization = strictConditionalization

    def update(self):
        self.credence = self.conditionalization(self.credence)

a = Agent()

