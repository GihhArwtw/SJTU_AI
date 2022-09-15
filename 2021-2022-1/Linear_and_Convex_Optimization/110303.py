# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:14:39 2021

@author: Lenovo
"""

import cvxpy as cp
import numpy as np

# Question 3(b)
x = cp.Variable((2,1))
A = np.array([[2,1],[1,-3],[1,2]])
b = np.array([[5],[10],[-5]])

constraints = [ cp.norm_inf(x) <= 1 ]

obj_func = cp.Minimize( cp.norm1(A @ x - b) )

prob = cp.Problem(obj_func, constraints)
prob.solve()

print("Question 3(b)")
print("status: ", prob.status)
print("optimum value: ",prob.value)
print("optimum var: \n[",x.value,"]",end="\n\n")

# Question 3(c)
y = cp.Variable((3,1))
constraints = [ x <= 1, -x <= 1, -y <= A@x -b, -y <= -A@x +b ]
c = np.array([[1],[1],[1]])

prob = cp.Problem( cp.Minimize(c.T@y), constraints)
prob.solve()

print("Question 3(c)")
print("status: ", prob.status)
print("optimum value: ",prob.value)
print("optimum var: \n[",x.value,"]",end="\n\n")
