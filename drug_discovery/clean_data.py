import pandas as pd 
import numpy as np 

fname = 'pkis2_matrix_1um.csv'
# target = 'CAMK2B'         # kinase
target = 'ACVRL1'

def read_in(fname):
    return pd.read_csv(fname, sep=',')

def select_target(df, target):
    return df[target].to_numpy()

def get_compound_names(df):
    return df['Pubchem_CID'].to_numpy()

def remove_missing(data, names):
    ''' 'data' from the actual percent inhibitions
        'names' is the corresponding pubchem CIDs 
    '''
    nonz_inds = np.argwhere(data != 0).flatten()
    return data[nonz_inds], names[nonz_inds]

def remove_duplicates(data, names):
    d2, n2 = [], []
    for dat, name in zip(data, names): 
        inds = np.flatnonzero(names == name)
        if len(inds) > 1:
            continue
        if not name in n2:
            d2.append(float(data[inds]))
            n2.append(name)
    return d2, n2

def is_promiscuous(df, CID):
    '''
    Is promiscuous if for 50% of compounds has 50% control
    '''
    all_names = df['Pubchem_CID'].to_numpy()
    above = 75      # 50%
    n_kinases = len(list(df))-1 # -1 to ignore pubchem CID
    row = np.where(all_names == CID)[0]
    count = 0
    for kinase in list(df):
        if kinase is not 'Pubchem_CID':
            for r in row:
                count += int(df[kinase].iloc[r] > above)
    return count > above

def transform_data(data):
    '''Original data chosen such that -log(value) 
    had standard deviation 1. The stored data is percent
    inhibition. 
    1) Threshold to 99.9 first
    2) Divide by 100 to remove x100 in storage.
    3) Compute 1-data to get percent control instead of inhibited.
    highest -log(values) are the most active compounds instead of min.
    4) Compute -log(data) for each point
    '''
    for i in range(len(data)):  # see doc string for explanation of steps. 
        if data[i] == 100:
            data[i] -= 0.1
    data = np.array(data)
    data /= 100     
    data = 1 - data 
    data = -np.log(data)
    return data

def get_data():
    df = read_in(fname)
    data = select_target(df, target)
    names = get_compound_names(df)
    data, names = remove_missing(data, names)
    data = transform_data(data)
    return data, names 

def compute_alpha_beta(means, thresh):
    means = sorted(means)[::-1]
    above = [m for m in means if m >= thresh]
    below = [m for m in means if m < thresh]
    alpha = above[-1] - thresh 
    beta  = thresh - below[0]
    return alpha, beta 

if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    data, names = get_data()
    plt.plot(sorted(data))
    plt.show()