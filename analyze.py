import pandas as pd

data = pd.read_csv('results.csv',sep=';')
print(data)

def getOutcome(agents):
    if False not in [a[1] > 0.99 for a in agents]:
        return 'TC'
    if False not in [a[1] < 0.5 for a in agents]:
        return 'FC'
    else:
        return 'P'

for i, row in data.iterrows():
    agents = [eval(c) for c in list(row)[1:]]
    data.at[i, 'Outcome'] = getOutcome(agents)

print(data)
