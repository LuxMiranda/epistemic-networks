import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

def getDist(df, name):
    n_repetitions = 100
    df = df.groupby(['outcome','m_mistrust']).size().reset_index(name='Count')
    df['Percent'] = df['Count'].apply(lambda x : x / n_repetitions)
    return df[df['outcome'].isin([name])]['m_mistrust']

def test_set(test):
    data_random = pd.read_csv('results-random.csv')
    data_test   = pd.read_csv(test)

    print(f'### Testing {test} ###')

    print('Test: True consensus')

    tc_random = getDist(data_random, 'True consensus')
    tc_test   = getDist(data_test, 'True consensus')

    U1, p = mannwhitneyu(tc_random, tc_test)
    print(f'p: {p}')
    U1, p = mannwhitneyu(tc_random, tc_test, method='exact')
    print(f'p: {p}')


    print('Test: Polarization')

    pol_random = getDist(data_random, 'Polarization')
    pol_mse    = getDist(data_test, 'Polarization')

    U1, p = mannwhitneyu(pol_random, pol_mse)
    print(f'p: {p}')
    U1, p = mannwhitneyu(pol_random, pol_mse, method='exact')
    print(f'p: {p}')

test_set('results-pull_similar.csv')
test_set('results-pull_dissimilar.csv')
test_set('results-most-similar.csv')
test_set('results-least-similar.csv')
test_set('results-partial.csv')
test_set('results-partial2.csv')
test_set('results-partial4.csv')
