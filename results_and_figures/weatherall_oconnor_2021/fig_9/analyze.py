import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

data = pd.read_csv('results.csv')

def getOutcome(agents):
    agents = eval(agents)
    for a in agents:
        if False in [c > 0.99 for c in a]:
            return 'Other'
    return 'True consensus'

data['outcome'] = data['credences'].apply(getOutcome)
data = data[['m_mistrust','outcome']]


#print(data[data['outcome'].apply(lambda x : x == 'False consensus')])

data = data[data['outcome'].isin(['True consensus','Other'])]

data = data.groupby(['outcome','m_mistrust']).size().reset_index(name='Count')
data['Percent'] = data['Count'].apply(lambda x : x / 100)

sns.lineplot(data=data[data['outcome'].isin(['True consensus'])],
    x='m_mistrust', y='Percent',
    marker='$♥$', markersize=10,label='True consensus')


ax = sns.lineplot(data=data[data['outcome'].isin(['Other'])],
    x='m_mistrust', y='Percent',
    marker='X', markersize=10,label='Other')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.ylim(top=1.05)
plt.xlabel('Mistrust ($m$)')
plt.ylabel('Percentage of outcomes')
plt.legend()
plt.show()
