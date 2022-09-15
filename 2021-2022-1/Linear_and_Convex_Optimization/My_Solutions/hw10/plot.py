# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 19:58:12 2021

@author: Lenovo
"""

import matplotlib.pyplot as plt
import numpy as np

def f_2d(x1, x2):
    return (x1*x1+x2*x2)

x_traces = [[-0.5,-1.],[2.,1.]]

fig = plt.figure(figsize=(4,3))

plt.plot([1.],[0.], '-o', color='#d0514e')

x1, x2 = zip(*x_traces)
x1 = np.arange(min(x1)-.2, max(x1)+.2, 0.01)
x2 = np.arange(min(x2)-.2, max(x2)+.2, 0.01)
x1, x2 = np.meshgrid(x1,x2)
plt.contour(x1, x2, f_2d(x1, x2), 20, colors='#b6d4f3')
plt.xlabel('x1')
plt.ylabel('x2')
plt.tight_layout(pad=.1)

plt.rc('font', family='serif')

fig.savefig("p2.svg")

for i in range(3):
    print(i)