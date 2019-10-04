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
import time
from tabulate import tabulate
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
plt.rcParams['agg.path.chunksize'] = 10000

"""
Task 1: Data Loading and Summary Statistics
"""
#set working directory
os.chdir('/Users/sarah/Box/Non_mkt_valuation/PS1_Hedonics/')


variables = ['house_ID', 'price', 'county', 'year_built', 'sq_feet', 'bath', 
             'bed', 'rooms', 'stories', 'violent_crime', 'property_crime', 'year_sale']

#sf_txt = pd.read_csv('sf_data.txt', delimiter = '\t')
#la_txt = pd.read_csv('la_data.txt', delimiter = '\t')

la = pd.read_csv('la_data1.csv')
sf = pd.read_csv('sf_data1.csv')

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

# Print the results in latex table
round(mean_la, 2).to_latex()
round(var_la, 2).to_latex()
round(mean_sf, 2).to_latex()
round(var_sf, 2).to_latex()


#Print the result in Markdown form
#print(tabulate(mean_sf, tablefmt="pipe", headers="keys"))
#print(tabulate(var_sf, tablefmt="pipe", headers="keys"))
#print(tabulate(mean_la, tablefmt="pipe", headers="keys"))
#print(tabulate(var_la, tablefmt="pipe", headers="keys"))

#la_group =  la.groupby(['county'])
#la_stats = la_group.describe()

"""
Task 2: Bootstrapped Hedonic Price Function
"""
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
la['ventura'] = (la['county'] == 111).astype(int)
la = la.drop('county', axis = 1)

#year dummies (fast version)
la = pd.concat([la.drop('year_sale', axis=1), pd.get_dummies(la['year_sale'])], axis=1)


# Select the X and Y
la_columns = la.columns.tolist()
la_columns = [c for c in la_columns if c not in ["house_ID","price", "county", 1999]]
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
X = la[la_columns]
Y = la['price']
X = sm.add_constant(X) # adding a constant

model = sm.OLS(Y, X).fit()
print_model = model.summary()
print(print_model)
results_la = pd.DataFrame(model.params)


# Now the bootstrap!

results_la_boot = np.zeros((len(model.params), r))

#aggiungere ciclo per r
for j in range(r):
    print('bootstrap iteration:', j)
    index = np.random.choice(n_la, size = n_la, replace = True)
    pd.value_counts(pd.Series(index)) #check the frequencies to be sure about replacement
    
    la_boot = pd.DataFrame()
    la_boot = la_boot.append(la.iloc[index,:]) #avoids looping over index
    X = la_boot[la_columns]
    X = sm.add_constant(X)
    Y = la_boot['price']
    
    model_boot = sm.OLS(Y, X).fit()
    results_la_boot[:,j] = np.asarray(model_boot.params)

std_la_boot = np.std(results_la_boot, axis = 1)

results_la['bootstrapped_SE'] = std_la_boot
round(results_la, 2).to_latex()
#print(tabulate(results_la, tablefmt="pipe", headers="keys"))

#store coefficients for violent crime (need to uncomment last line if needed)
violent_crime = pd.DataFrame(results_la_boot.transpose(), columns = results_la['variables'])
violent_crime= violent_crime[['violent_crime', 'violent_crime_sq']]
violent_crime.to_csv ('violent_crime_la1.csv', index = None, header=True)
del violent_crime
del la_boot

"""
Task 3: Bootstrapped Multimarket Rosen Estimates
"""
buyers = pd.read_csv('buyer_data_sf_la.csv', header=None)
buyers = pd.concat([buyers.drop(4, axis=1), pd.get_dummies(buyers[4])], axis=1)


buyers.columns = ['buyer_ID', 'price', 'violent_crime', 'prop_crime', 'income',
                  'LA', 'asian', 'black', 'hisp', 'white']

la_crime = pd.read_csv('violent_crime_la1.csv')
sf_crime = pd.read_csv('violent_crime_sf.csv')


violent_crime = pd.concat([la_crime, sf_crime], axis=1)
violent_crime.columns = [['violent_crime_la', 'violent_crime_sq_la', 
                          'violent_crime_sf', 'violent_crime_sq_sf']]

del [la_crime, sf_crime] #not to have too many databases in memory


#Using estimated parameters -> SAVE THE PARAMETERS AS FIRST LINE IN THE LACRiME
a_la, b_la, a_sf, b_sf = -202.275, 0.074, -225.167, 0.112

#confusing syntax. Useful to learn. Assigns implicit price variable ad the
#dericative of the hedonid function with respect to  violent crime
buyers['implicit_price'] = np.where(buyers['LA'] == 0, 
      a_sf + 2*b_sf*buyers['violent_crime'], a_la + 2*b_la*buyers['violent_crime'])

buyers_columns = buyers.columns.tolist()
buyers_columns = [c for c in buyers_columns if c not in ["buyer_ID","price", 
                                            "prop_crime", 'implicit_price', 'white']]

X = buyers[buyers_columns]
Y = buyers['implicit_price']
X = sm.add_constant(X) # adding a constant

model = sm.OLS(Y, X).fit()
print_model = model.summary()
print(print_model)
results_buy = pd.DataFrame(model.params)


#Y = "implicit_price"
#ols_buy = LinearRegression(normalize = True).fit(buyers[buyers_columns], buyers[Y])
#results_buy = pd.DataFrame()
#results_buy['variable'] = buyers_columns
#results_buy['coefficient'] = ols_buy.coef_
#results_buy.rename(columns={0:''}, inplace=True)
#inter = [] 
#inter.insert(0, {'coefficient': ols_buy.intercept_, 'variable': 'intercept' })
#results_buy = pd.concat([pd.DataFrame(inter, columns = ['variable', 'coefficient']), results_buy], ignore_index=True, sort=False)

"""
Bootstrap -> run 1 regression for each set of parameters estimated before
"""
n_b = len(buyers)
results_rosen_boot = np.zeros((len(results_buy), len(violent_crime)))

for j in range(len(violent_crime)):
    print('bootstrap iteration:', j)
    index = np.random.choice(n_b, size = n_b, replace = True)
    
    buyers_boot = pd.DataFrame()
    buyers_boot = buyers_boot.append(buyers.iloc[index,:]) #avoids looping over index
    
    a_la = violent_crime.iloc[j,0] 
    b_la = violent_crime.iloc[j,1]
    a_sf = violent_crime.iloc[j,2]
    b_sf = violent_crime.iloc[j,3]
    
    X = buyers_boot[buyers_columns]
    X = sm.add_constant(X) # adding a constant

    

    buyers_boot['implicit_price'] = np.where(buyers_boot['LA'] == 0, 
      a_sf + 2*b_sf*buyers_boot['violent_crime'], a_la + 2*b_la*buyers_boot['violent_crime'])
    buyers_boot_columns = buyers.columns.tolist()
    buyers_boot_columns = [c for c in buyers_boot_columns if c not in ["buyer_ID","price", 
                                            "prop_crime", 'implicit_price', 'white']]
    Y = buyers_boot['implicit_price']
    
    model = sm.OLS(Y, X).fit()
    results_rosen_boot[:,j] = np.array(model.params)
    
    #ols_ros_la = LinearRegression(normalize = True).fit(buyers_boot[buyers_columns], buyers_boot[Y])
    #results_rosen_boot[:,j] = np.append(ols_ros_la.intercept_, ols_ros_la.coef_)

std_rosen = np.std(results_rosen_boot, axis = 1)

results_buy['bootstrapped_SE'] = std_rosen
round(results_buy, 4).to_latex()


""" 
Task 4 - Non-parametric Hedonic Price Function
"""
n = 2000
bk = pd.read_csv('la_data1.csv', header=None)
bk.columns = ['house_ID', 'price', 'county', 'year_built', 'sq_feet', 'bath', 
             'bed', 'rooms', 'stories', 'violent_crime', 'property_crime', 'year_sale']


xi = np.arange(0, n) #nb: last argument is the step
vc = np.asarray(bk['violent_crime'])
P = np.asarray(bk['price'])
sigma = np.std(vc)

def gauss_kernel(VC, xi, h, sigma):
    return(1/(h* sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((VC - xi)/(h*sigma))**2))

beta_1 = np.zeros(n)
beta_3 = np.zeros(n)
beta_10 = np.zeros(n)
beta_1000 = np.zeros(n)

start = time.time()
for i in range(len(xi)):
    apply = np.vectorize(gauss_kernel, excluded = ['xi', 'h', 'sigma']) #apply function to the vector and avoid loop
    k_1 = apply(vc, i, 1, sigma)
    k_3 = apply(vc, i, 3, sigma)
    k_10 = apply(vc, i, 10, sigma)
    k_1000 = apply(vc, i, 1000, sigma)
    
    #Instead of working with the huge matrices, let's define:
    A_1 = np.sum(np.multiply(k_1,P))
    B_1 = np.sum(np.sum(k_1))
    C_1 = np.multiply(vc, k_1)
    D_1 = np.sum(np.multiply(C_1,P))
    C_1 = np.sum(C_1)
    F_1 = np.sum(np.multiply(vc**2, k_1))
    #alpha_1[j] = (C_1*D_1 - A_1*F_1)/(C_1**2 - B_1*F_1)
    beta_1[int(i)] = (A_1*C_1 - B_1*D_1)/(C_1**2 - B_1*F_1)

  
    #h = 3
    A_3 = np.sum(np.multiply(k_3,P))
    B_3 = np.sum(np.sum(k_3))
    C_3 = np.multiply(vc, k_3)
    D_3 = np.sum(np.multiply(C_3,P))
    C_3 = np.sum(C_3)
    F_3 = np.sum(np.multiply(vc**2, k_3))
    #alpha_3[j] = (C_3*D_3 - A_3*F_3)/(C_3**2 - B_3*F_3)
    beta_3[int(i)] = (A_3*C_3 - B_3*D_3)/(C_3**2 - B_3*F_3)
    
    #h = 10
    A_10 = np.sum(np.multiply(k_10,P))
    B_10 = np.sum(np.sum(k_10))
    C_10 = np.multiply(vc, k_10)
    D_10 = np.sum(np.multiply(C_10,P))
    C_10 = np.sum(C_10)
    F_10 = np.sum(np.multiply(vc**2, k_10))
    #alpha_10[j] = (C_10*D_10 - A_10*F_10)/(C_10**2 - B_10*F_10)
    beta_10[int(i)] = (A_10*C_10 - B_10*D_10)/(C_10**2 - B_10*F_10)
    
    #h = 100
    A_1000 = np.sum(np.multiply(k_1000,P))
    B_1000 = np.sum(np.sum(k_1000))
    C_1000 = np.multiply(vc, k_1000)
    D_1000 = np.sum(np.multiply(C_1000,P))
    C_1000 = np.sum(C_1000)
    F_1000 = np.sum(np.multiply(vc**2, k_1000))
    #alpha_100[j] = (C_100*D_100 - A_100*F_100)/(C_100**2 - B_100*F_100)
    beta_1000[int(i)] = (A_1000*C_1000 - B_1000*D_1000)/(C_1000**2 - B_1000*F_1000)      

    print(int(i))
end = time.time()
print(end - start)


fig = plt.figure()
ax = fig.add_subplot(1, 1,1)
ax.set_ylim([-370,0])
ax.plot(xi, beta_1, color='tab:blue')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1,1)
ax.set_ylim([-370,0])
ax.plot(xi, beta_3, color='tab:blue')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1,1)
ax.set_ylim([-370,0])
ax.plot(xi, beta_10, color='tab:blue')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1,1)
ax.set_ylim([-370,0])
ax.plot(xi, beta_1000, color='tab:blue')
plt.show()

Beta = pd.DataFrame([beta_1, beta_3, beta_10, beta_1000])
Beta.to_csv ('beta_task1.csv', index = None, header=True)

# Plotting Gaussian Kernels for xi = 2000
#fig = plt.figure()
#ax = fig.add_subplot(1, 1,1)
#ax.set_ylim([0,0.00175])
#ax.scatter(vc, k_1, color='tab:blue')
#plt.show()
#fig = plt.figure()
#ax = fig.add_subplot(1, 1,1)
#ax.set_ylim([0,0.0007])
#ax.scatter(vc, k_3, color='tab:blue')
#plt.show()
#fig = plt.figure()
#ax = fig.add_subplot(1, 1,1)
#ax.set_ylim([0,0.0002])
#ax.scatter(vc, k_10, color='tab:blue')
#plt.show()
#fig = plt.figure()
#ax = fig.add_subplot(1, 1,1)
#ax.set_ylim([0,0.00002])
#ax.scatter(vc, k_1000, color='tab:blue')
#plt.show()



