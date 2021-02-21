import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def invert(string):
    """
    A function which invert a string.
    Parameters
    ----------
    string : string
        the string to invert.

    Returns
    -------
    string
        An inverted string.

    Required libraries
    ------------------
    None.
    """
    return string[::-1]

def weighted_median(data, weights, interpolate = False):
    """
    A function that calculates the weighted median of a given series of values 
    by using a series of weights.
    
    Parameters
    ----------
    data : Iterable
        The data which the function calculates the median for.
    weights : Iterable
        The weights the function uses to calculate an weighted median.

    Returns
    -------
    numpy.float64
        The function return the weighted median.
        
    Required libraries
    ---------
    Numpy.
    """
    #Forcing the data to a numpy array.
    data = np.array(data)
    weights = np.array(weights)
    
    #Sorting the data and the weights.
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
   
    #Calculating the cumulative sum of the weights.
    sn = np.cumsum(sorted_weights)
    
    #Calculating the threshold.
    threshold = sorted_weights.sum()/2
   
    #Interpolating the median and returning it.
    if interpolate == True:
        return np.interp(0.5, (sn - 0.5 * sorted_weights) / np.sum(sorted_weights), sorted_data)
    
    #Returning the first value that equals or larger than the threshold.
    else:
        return sorted_data[sn >= threshold][0]
    
#Importing the countries csv and creating an empty DataFrame to contain the results.    
df_countries = pd.read_csv(r'')
results = pd.DataFrame()

for country in df_countries['Country']:
    #Excluding countries that don't have profession data.
    if country in ['Austria', 'Canada', 'Estonia', 'Finland', 'United Kingdoms']:
        continue
    
    #Reading a file and dropping irrelevant rows.
    df = pd.read_csv(r''+str(country)+'.csv', low_memory = False)
    if country in ['France', 'Italy', 'Spain']:
        df.dropna(subset = ['PVLIT1', 'PVNUM1'], inplace = True)
    else:
        df.dropna(subset = ['PVLIT1', 'PVNUM1', 'PVPSL1'], inplace = True)
    
    #Slicing the teachers.
    dfe = df[df['ISCO2C'] == '23'].copy()
    dfe = dfe[dfe['B_Q01a'].isin(['12'])]
    dfn = df[df['ISCO2C'] != '23'].copy()
    dfn = dfn[dfn['B_Q01a'].isin(['12'])]
    
    #If the sample is too small, skip.
    if len(dfe) < 50:
        continue
    
    #Saving the median of each skill.
    results.loc[country, 'Lit Median'] = weighted_median(dfe['PVLIT1'], dfe['SPFWT0'], interpolate = True) 
    results.loc[country, 'Num Median'] = weighted_median(dfe['PVNUM1'], dfe['SPFWT0'], interpolate = True) 
    results.loc[country, 'PSL Median'] = weighted_median(dfe['PVPSL1'], dfe['SPFWT0'], interpolate = True)

#Creating the figure.
results.loc[:, 'Color'] = 'tab:blue'
results.loc['Israel', 'Color'] = 'tab:orange'
fig, axes = plt.subplots(3, 1, figsize = (10,20), tight_layout = True)
results.sort_values(by = ['Lit Median'], inplace = True)
axes[0].bar(results.index, results['Lit Median'], color = results['Color'])
axes[0].set_xticklabels(labels = results.index, rotation = 'vertical', fontsize = 14)
results.sort_values(by = ['Num Median'], inplace = True)
axes[1].bar(results.index, results['Num Median'], color = results['Color'])
axes[1].set_xticklabels(labels = results.index, rotation = 'vertical', fontsize = 14)
results.sort_values(by = ['PSL Median'], inplace = True)
results.drop(['Spain', 'Italy', 'France'], inplace = True)
axes[2].bar(results.index, results['PSL Median'], color = results['Color'])
axes[2].set_xticklabels(labels = results.index, rotation = 'vertical', fontsize = 14)
fig.suptitle(invert('ביצועי מורים במבחני CAAIP ביחס לאנשים בעלי השכלה זהה'), x = 0.5, y = 1.02, fontsize = 20)
for i in range(0,3):
    axes[i].set_ylabel(invert('יחס הציונים'), fontsize = 14)
axes[0].set_title(invert('אורינות מילולית'), fontsize = 14)
axes[1].set_title(invert('אורינות כמותית'), fontsize = 14)
axes[2].set_title(invert('פתרון בעיות'), fontsize = 14)