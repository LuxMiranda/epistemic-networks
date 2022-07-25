import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

TC_rec = '#19fc00'
OC_rec = '#fce702'
PZ_rec = '#fc0202'

TC_ctl = '#b1fc9f'
OC_ctl = '#fcf49f'
PZ_ctl = '#fca39f'

plt.figure(figsize=(10,5))

####################
### CONTROL DATA ###
####################


data_control = pd.read_csv('results-control.csv')
data_control = data_control[['m_mistrust','outcome']]

n_repetitions = 100

data_control = data_control.groupby(['outcome','m_mistrust']).size().reset_index(name='Count')
data_control['Percent'] = data_control['Count'].apply(lambda x : x / n_repetitions)

sns.lineplot(data=data_control[data_control['outcome'].isin(['True consensus'])],
    x='m_mistrust', y='Percent', color=TC_ctl, linestyle='dashed',
    marker='$♥$', markersize=7,label='Complete: True consensus')

ax = sns.lineplot(data=data_control[data_control['outcome'].isin(
    ['Mixed consensus','False consensus'])],
    x='m_mistrust', y='Percent', color=OC_ctl, linestyle='dashed',
    marker='^', markersize=7,label='Complete: Other consensus')

ax = sns.lineplot(data=data_control[data_control['outcome'].isin(['Polarization'])],
    x='m_mistrust', y='Percent', color=PZ_ctl, linestyle='dashed',
    marker='X', markersize=7,label='Complete: Polarization')

#########################
### RECOMMENDER DATA ###
#######################


data_rec = pd.read_csv('results-rec.csv')
data_rec = data_rec[['m_mistrust','outcome']]

n_repetitions = 100

data_rec = data_rec.groupby(['outcome','m_mistrust']).size().reset_index(name='Count')
data_rec['Percent'] = data_rec['Count'].apply(lambda x : x / n_repetitions)


sns.lineplot(data=data_rec[data_rec['outcome'].isin(['True consensus'])],
    x='m_mistrust', y='Percent', color=TC_rec,
    marker='$♥$', markersize=7,label='LS Recommender: True consensus')

ax = sns.lineplot(data=data_rec[data_rec['outcome'].isin(
    ['Mixed consensus','False consensus'])],
    x='m_mistrust', y='Percent', color=OC_rec,
    marker='^', markersize=7,label='LS Recommender: Other consensus')

ax = sns.lineplot(data=data_rec[data_rec['outcome'].isin(['Polarization'])],
    x='m_mistrust', y='Percent', color=PZ_rec,
    marker='X', markersize=7,label='LS Recommender: Polarization')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.ylim(top=1.05)
plt.xlabel('Mistrust ($m$)')
plt.ylabel('Percentage of outcomes')
plt.title('Least-similar recommender vs. Complete network')
plt.show()
