# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:21:25 2021

@author: Lenovo
"""

import cvxpy as cp
import numpy as np

w = cp.Variable((2,1))
X = np.array([[2,0],[0,1],[0,0]])
y = np.array([[3],[2],[2]])

obj_func = cp.Minimize( cp.norm2(X @ w - y) ** 2 )

# Question 4(b): Lasso 
print("Question 4(b)")
for i in [1,10]:
    constraints = [cp.norm1(w) <= i]
    prob = cp.Problem(obj_func, constraints)
    prob.solve()
    print("Lasso with t =",i,":")
    print("status: ", prob.status)
    print("optimum value: ",prob.value)
    print("optimum var: \n[",w.value,"]",end="\n\n")

# Question 4(c): Ridge Regression
print("Question 4(c)")
for i in [1,100]:
    constraints = [cp.norm2(w) <= i]
    prob = cp.Problem(obj_func, constraints)
    prob.solve()
    print("Ridge Regression with t =",i,":")
    print("status: ", prob.status)
    print("optimum value: ",prob.value)
    print("optimum var: \n[",w.value,"]",end="\n\n")
