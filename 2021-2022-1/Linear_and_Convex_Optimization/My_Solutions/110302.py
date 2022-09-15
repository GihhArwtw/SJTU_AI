# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 20:03:07 2021

@author: Lenovo
"""

# Question 2
import cvxpy as cp

x = cp.Variable()
y = cp.Variable()

constraints = [ 2*x + y >= 1, x + 3*y >= 1, x >= 0, y >= 0]

obj_funcs = [ cp.Minimize( x+y ),
              cp.Minimize( -x-y ),
              cp.Minimize( x ),
              cp.Minimize( cp.maximum(x,y) ),
              cp.Minimize( x**2+9*(y**2) ) ]

ch = 'a'

for obj in obj_funcs:
    print("Question 2("+ch+")")
    prob = cp.Problem(obj, constraints)
    prob.solve()
    ch = chr(ord(ch) + 1)
    print("status: ", prob.status)
    print("optimum value: ",prob.value)
    print("optimum var: [",x.value,",",y.value,"]",end="\n\n")
