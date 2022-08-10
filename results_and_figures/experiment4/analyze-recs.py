import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import numpy as np

n_repetitions = 100


data = pd.read_csv('results-vary_n_recs.csv')

data = data[data['recommend'].isin(['random'])]

data = data.groupby(['outcome','m_mistrust','n_recommendations']).size().reset_index(name='Count')
data['Percent'] = data['Count'].apply(lambda x : x / n_repetitions)

data = data[data['outcome'].isin(['Polarization'])]


null_data = [['Polarization',m,a] for m in np.linspace(0.1,1.0,num=10) for a in [2,4,6,8,10]]
null_data = pd.DataFrame(null_data, columns=['outcome','m_mistrust','n_recommendations'])

data = pd.merge(data, null_data, how='outer')
data = data.fillna(0)

sns.set_theme(style='white', font='serif')
fig, ax = plt.subplots(figsize=(6,3))
ax = sns.lineplot(data=data, x='m_mistrust',y='Percent',hue='n_recommendations',linewidth=4)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_xlim((0,3))

ax.legend(title="# of recs")

plt.xlabel('Mistrust (m)')
plt.ylabel('Probability of polarization')
plt.subplots_adjust(left=0.15,bottom=0.174)
sns.despine()
plt.savefig('RandomVariedRecs.pdf')

###

data = pd.read_csv('results-vary_n_recs.csv')

data = data[data['recommend'].isin(['similar'])]

data = data.groupby(['outcome','m_mistrust','n_recommendations']).size().reset_index(name='Count')
data['Percent'] = data['Count'].apply(lambda x : x / n_repetitions)

data = data[data['outcome'].isin(['Polarization'])]

sns.set_theme(style='white', font='serif')
fig, ax = plt.subplots(figsize=(6,3))
ax = sns.lineplot(data=data, x='m_mistrust',y='Percent',hue='n_recommendations',linewidth=4)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_xlim((0,3))


ax.legend(title="# of recs")

plt.xlabel('Mistrust (m)')
plt.ylabel('Probability of polarization')
plt.subplots_adjust(left=0.15,bottom=0.174)
sns.despine()
plt.savefig('MostSimilarVariedRecs.pdf')
