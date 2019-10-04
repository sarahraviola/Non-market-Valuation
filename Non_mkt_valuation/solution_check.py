#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 21:00:44 2019

@author: sarah
"""

beta_1 = np.zeros(n)
beta_3 = np.zeros(n)
beta_10 = np.zeros(n)
beta_1000 = np.zeros(n)

xi = 1
h = 1
n = len(vc)

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
    alpha_1 = (C_1*D_1 - A_1*F_1)/(C_1**2 - B_1*F_1)
    beta_1 = (A_1*C_1 - B_1*D_1)/(C_1**2 - B_1*F_1)

  
N = np.zeros(n)
M = np.zeros(n)
for i in range(n):
    N[i] = np.float128((P[i] - alpha_1 - beta_1*vc[i])*k_1[i])
    M[i] = np.float128(vc[i]*N[i])