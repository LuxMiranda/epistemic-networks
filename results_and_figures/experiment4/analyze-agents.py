import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

n_repetitions = 25

#data = pd.read_csv('results-vary_n_agents.csv')
#
#data = data[data['recommend'].isin(['random'])]
#
#data = data.groupby(['outcome','m_mistrust','n_agents']).size().reset_index(name='Count')
#data['Percent'] = data['Count'].apply(lambda x : x / n_repetitions)
#
#data = data[data['outcome'].isin(['Polarization'])]
#
#ax = sns.lineplot(data=data, x='m_mistrust',y='Percent',hue='n_agents', style='n_agents', markers=True, dashes=False, linewidth=4)
#ax.set_xlim((0,3))
#plt.show()
#

###


data = pd.read_csv('results-vary_n_agents.csv')

data = data[data['recommend'].isin(['similar'])]

data = data.groupby(['outcome','m_mistrust','n_agents']).size().reset_index(name='Count')
data['Percent'] = data['Count'].apply(lambda x : x / n_repetitions)

data = data[data['outcome'].isin(['Polarization'])]

ax = sns.lineplot(data=data, x='m_mistrust',y='Percent',hue='n_agents', style='n_agents', markers=True, dashes=False, linewidth=4)
ax.set_xlim((0,3))
plt.show()
