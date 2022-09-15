# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import random


x_0 = 0.3
y_0 = 0.7

x_pnt = [0, 0.83, 0.97, 0.2, 0.3333]
y_pnt = [0.6, 0.8, 0.1, 0.0, 0.9]

"""for i in range(5):
    x_pnt = x_pnt + [random.random()]
    y_pnt = y_pnt + [random.random()]
"""
    
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16,16))

plt.scatter(x_pnt, y_pnt, s=500)
plt.scatter(x_0, y_0, s=500, cmap="#9e1d1d")

for i in range(5):
    perp = (-y_pnt[i]+y_0, x_pnt[i]-x_0)
    
    x_m = (x_0+x_pnt[i])/2
    y_m = (y_0+y_pnt[i])/2
    
    x_min = 0
    x_max = 1
    y_min = 0
    y_min = 1
    
    if (perp[0]==0):
        x_min = x_m
        y_min = 0
        x_max = x_m
        y_max = 1
    else:
        if (perp[1]==0):
            x_min = 0
            y_min = y_m
            x_max = 1
            y_max = y_m
        else:
            y_min = y_m - x_m*perp[1]/perp[0]
            if (y_min<0):
                y_min = 0
                x_min = x_m - y_m*perp[0]/perp[1]
            if (y_min>1):
                y_min = 1
                x_min = x_m + (1-y_m)*perp[0]/perp[1]
                
            y_max = y_m + (1-x_m)*perp[1]/perp[0]
            if (y_max<0):
                y_max = 0
                x_max = x_m - y_m*perp[0]/perp[1]
            if (y_max>1):
                y_max = 1
                x_max = x_m + (1-y_m)*perp[0]/perp[1]
                
    plt.plot([x_min,x_max],[y_min,y_max],'#637ca1',linewidth=6)
          
plt.xticks([])  #去掉横坐标值
plt.yticks([])  #去掉横坐标值
                  
fig.show()