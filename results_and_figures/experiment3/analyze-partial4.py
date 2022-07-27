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


data_control = pd.read_csv('results-random.csv')
data_control = data_control[['m_mistrust','outcome']]

n_repetitions = 100

data_control = data_control.groupby(['outcome','m_mistrust']).size().reset_index(name='Count')
data_control['Percent'] = data_control['Count'].apply(lambda x : x / n_repetitions)

sns.lineplot(data=data_control[data_control['outcome'].isin(['True consensus'])],
    x='m_mistrust', y='Percent', color=TC_ctl, linestyle='dashed',
    marker='$♥$', markersize=7,label='Random: True consensus')

ax = sns.lineplot(data=data_control[data_control['outcome'].isin(
    ['Mixed consensus','False consensus'])],
    x='m_mistrust', y='Percent', color=OC_ctl, linestyle='dashed',
    marker='^', markersize=7,label='Random: Other consensus')

ax = sns.lineplot(data=data_control[data_control['outcome'].isin(['Polarization'])],
    x='m_mistrust', y='Percent', color=PZ_ctl, linestyle='dashed',
    marker='X', markersize=7,label='Random: Polarization')

#########################
### RECOMMENDER DATA ###
#######################


data_rec = pd.read_csv('results-partial4.csv')
data_rec = data_rec[['m_mistrust','outcome']]

n_repetitions = 100

data_rec = data_rec.groupby(['outcome','m_mistrust']).size().reset_index(name='Count')
data_rec['Percent'] = data_rec['Count'].apply(lambda x : x / n_repetitions)


sns.lineplot(data=data_rec[data_rec['outcome'].isin(['True consensus'])],
    x='m_mistrust', y='Percent', color=TC_rec,
    marker='$♥$', markersize=7,label='50/50 Least-similar: True consensus')

ax = sns.lineplot(data=data_rec[data_rec['outcome'].isin(
    ['Mixed consensus','False consensus'])],
    x='m_mistrust', y='Percent', color=OC_rec,
    marker='^', markersize=7,label='50/50 Least-similar: Other consensus')

ax = sns.lineplot(data=data_rec[data_rec['outcome'].isin(['Polarization'])],
    x='m_mistrust', y='Percent', color=PZ_rec,
    marker='X', markersize=7,label='50/50 Least-similar: Polarization')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.ylim(top=1.05)
plt.xlabel('Mistrust ($m$)')
plt.ylabel('Percentage of outcomes')
plt.title('50/50 Least-similar recommender vs. Random recommender')
plt.show()
