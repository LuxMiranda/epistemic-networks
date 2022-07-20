import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('results.csv',sep=';')

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

data = data[['m_mistrust','Outcome']]

print(data)

data = data.groupby(['m_mistrust','Outcome']).size().reset_index(name='Count')

TC = data[data['Outcome'].isin(['TC'])]
TC['Percent'] = TC['Count'].apply(lambda x : x / 50)

P = data[data['Outcome'].isin(['P'])]
P['Percent'] = P['Count'].apply(lambda x : x / 50)


print(data)

sns.lineplot(data=TC, x='m_mistrust', y='Percent', marker='o')
sns.lineplot(data=P, x='m_mistrust', y='Percent', marker='o')
plt.show()
