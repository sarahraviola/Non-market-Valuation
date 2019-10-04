#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-Market Valuation - Problem Set 1 -San Francisco
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
#n_la = la.shape[0]
#n_la = 10
n_sf = sf.shape[0]
r = 500
[begin, end] = [1993, 2008]


#prepare the database
sf["prop_crime_sq"] = sf["property_crime"]**2
sf["year_built_sq"] = sf["year_built"]**2
sf["sq_feet_sq"] = sf["sq_feet"]**2
sf["rooms_sq"] = sf["rooms"]**2
sf["violent_crime_sq"] = sf["violent_crime"]**2

#county dummies creation  ("long version of dummy creation)
sf['contra_costa']= (sf['county'] == 13).astype(int)
sf['sfrancisco']= (sf['county'] == 75).astype(int)
sf['smateo']= (sf['county'] == 81).astype(int)
sf['sclara']= (sf['county'] == 85).astype(int)
sf = sf.drop('county', axis = 1)

#year dummies (fast version)
sf = pd.concat([sf.drop('year_sale', axis=1), pd.get_dummies(sf['year_sale'])], axis=1)

# Select the X and Y
sf_columns = sf.columns.tolist()
sf_columns = [c for c in sf_columns if c not in ["house_ID","price", "county", 1999]]
#Y = "price"
#
#ols_la = LinearRegression().fit(la[la_columns], la[Y])
#results_la = pd.DataFrame(la_columns)
#results_la.columns = ['variables']
#results_la['coefficients'] = ols_la.coef_
#
#inter = [] 
#inter.insert(0, {'variables': 'intercept', 'coefficients': ols_la.intercept_})
#results_la = pd.concat([pd.DataFrame(inter, columns = ['variables', 'coefficients']), results_la], ignore_index=True, sort=False)

#la_columns = ['bath', 'bed','stories', 'property_crime', 'prop_crime_sq', 'year_built', 'sq_feet']
X = sf[sf_columns]
Y = sf['price']
X = sm.add_constant(X) # adding a constant

model = sm.OLS(Y, X).fit()
print_model = model.summary()
print(print_model)
results_sf = pd.DataFrame()
results_sf['variables'] = model.params


# Now the bootstrap!
results_sf_boot = np.zeros((len(model.params), r))

#aggiungere ciclo per r
for j in range(r):
    print('bootstrap iteration:', j)
    index = np.random.choice(n_sf, size = n_sf, replace = True)
    pd.value_counts(pd.Series(index)) #check the frequencies to be sure about replacement
    
    sf_boot = pd.DataFrame()
    sf_boot = sf_boot.append(sf.iloc[index,:]) #avoids looping over index
    X = sf_boot[sf_columns]
    X = sm.add_constant(X)
    Y = sf_boot['price']
    
    model_boot = sm.OLS(Y, X).fit()
    results_sf_boot[:,j] = np.asarray(model_boot.params)

std_sf_boot = np.std(results_sf_boot, axis = 1)

results_sf['bootstrapped_SE'] = std_sf_boot
round(results_sf, 2).to_latex()
#print(tabulate(results_la, tablefmt="pipe", headers="keys"))

#store coefficients for violent crime (need to uncomment last line if needed)
violent_crime = pd.DataFrame(results_sf_boot.transpose(), columns =  list(results_sf.index.values.tolist()))
violent_crime= violent_crime[['violent_crime', 'violent_crime_sq']]
violent_crime.to_csv ('violent_crime_sf1.csv', index = None, header=True)
del violent_crime
del sf_boot







