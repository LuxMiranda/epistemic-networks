import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

data = pd.read_csv('results.csv')
data = data[['m_mistrust','outcome']]


data = data.groupby(['outcome','m_mistrust']).size().reset_index(name='Count')
data['Percent'] = data['Count'].apply(lambda x : x / 100)

sns.lineplot(data=data[data['outcome'].isin(['True consensus'])],
    x='m_mistrust', y='Percent',
    marker='$♥$', markersize=10,label='True consensus')


ax = sns.lineplot(data=data[data['outcome'].isin(['Polarization'])],
    x='m_mistrust', y='Percent',
    marker='X', markersize=10,label='Polarization')

ax = sns.lineplot(data=data[data['outcome'].isin(['Mixed consensus'])],
    x='m_mistrust', y='Percent',
    marker='o', markersize=10,label='Mixed consensus')


ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.ylim(top=1.05)
plt.xlabel('Mistrust ($m$)')
plt.ylabel('Percentage of outcomes')
plt.legend()
plt.show()
