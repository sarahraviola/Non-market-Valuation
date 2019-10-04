#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-Market Valuation - Problem Set 1
Author: Sarah Raviola

"""
# Import Necessary Packages
import numpy as np
import scipy as sp
import pandas as pd #to handle databases
import os
from tabulate import tabulate

"""
Task 1: Data Loading and Summary Statistics
"""
#set working directory
os.chdir('/Users/sarah/Box/Non_mkt_valuation/PS1_Hedonics/')


variables = ['house_ID', 'price', 'county', 'year_built', 'sq_feet', 'bath', 
             'bed', 'rooms', 'stories', 'violent_crime', 'property_crime', 'year_sale']

#sf_txt = pd.read_csv('sf_data.txt', delimiter = '\t')
#la_txt = pd.read_csv('la_data.txt', delimiter = '\t')

la = pd.read_csv('la_data.csv')
sf = pd.read_csv('sf_data.csv')

la.columns = sf.columns = variables

delete = [41, 55, 95, 97] #select sf counties to delete
for i in delete:
    sf = sf[sf.county != i]
    
mean_sf = sf.groupby(['county']).mean()
var_sf = sf.groupby(['county']).std()
mean_sf = mean_sf.drop(columns = 'house_ID')
var_sf = var_sf.drop(columns = 'house_ID')

mean_la = la.groupby(['county']).mean()
var_la = la.groupby(['county']).std()
mean_la = mean_la.drop(columns = 'house_ID')
var_la = var_la.drop(columns = 'house_ID')

#Print the result in Markdown form
print(tabulate(mean_sf, tablefmt="pipe", headers="keys"))
print(tabulate(var_sf, tablefmt="pipe", headers="keys"))
print(tabulate(mean_la, tablefmt="pipe", headers="keys"))
print(tabulate(var_la, tablefmt="pipe", headers="keys"))

#la_group =  la.groupby(['county'])
#la_stats = la_group.describe()

"""
Task 2: Bootstrapped Hedonic Price Function
"""
from sklearn.linear_model import LinearRegression
n_la = la.shape[0]
r = 500
[begin, end] = [1993, 2008]

#prepare the database
la["prop_crime_sq"] = la["property_crime"]**2
la["year_built_sq"] = la["year_built"]**2
la["sq_feet_sq"] = la["sq_feet"]**2
la["rooms_sq"] = la["rooms"]**2
la["violent_crime_sq"] = la["violent_crime"]**2

#county dummies creation  ("long version of dummy creation)
la['orange']= (la['county'] == 59).astype(int)
la['riverside']= (la['county'] == 65).astype(int)
la['sbernardino']= (la['county'] == 71).astype(int)
la = la.drop('county', axis = 1)

#year dummies (fast version)
la = pd.concat([la.drop('year_sale', axis=1), pd.get_dummies(la['year_sale'])], axis=1)


# Select the X and Y
la_columns = la.columns.tolist()
la_columns = [c for c in la_columns if c not in ["house_ID","price", "county", 1999]]
Y = "price"

ols_la = LinearRegression().fit(la[la_columns], la[Y])
results_la = pd.DataFrame(la_columns)
results_la.columns = ['variables']
results_la['coefficients'] = ols_la.coef_

inter = [] 
inter.insert(0, {'variables': 'intercept', 'coefficients': ols_la.intercept_})
results_la = pd.concat([pd.DataFrame(inter), results_la], ignore_index=True, sort=False)

# Now the bootstrap!

results_la_boot = np.zeros((len(ols_la.coef_)+1, r))

#aggiungere ciclo per r
for j in range(r):
    print('bootstrap iteration:', j)
    index = np.random.choice(n_la, size = n_la, replace = True)
    pd.value_counts(pd.Series(index)) #check the frequencies to be sure about replacement
    
    la_boot = pd.DataFrame()
    la_boot = la_boot.append(la.iloc[index,:]) #avoids looping over index
        
    
    ols_la_boot = LinearRegression().fit(la_boot[la_columns], la_boot[Y])
    results_la_boot[:,j] = np.append(ols_la_boot.intercept_, ols_la_boot.coef_)

std_la_boot = np.std(results_la_boot, axis = 1)

results_la['bootstrapped_SE'] = std_la_boot

print(tabulate(results_la, tablefmt="pipe", headers="keys"))






