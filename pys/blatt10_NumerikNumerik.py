#!/usr/bin/env python
# coding: utf-8

# In[1]:


#------------------------------------------------------------------------------#

# Numerik WiSe 2021/22 - Blatt 10
# Aufgabe: 4
# Autor: Jakob Sachs
# Gruppe: NumerikNumerik

#------------------------------------------------------------------------------#

# Importiere Module #

import numpy as np
import matplotlib.pyplot as plt
import time

# Lade StyleSheet #

plt.style.use('fivethirtyeight')


# In[48]:


def bisektion(f , a, b):
  assert(a < b)
  assert(f(a)*f(b) < 0)

  u = [a]
  v = [b]

  x = (a+b)/2
  x_last = 0

  while True:
    x = (u[-1] + v[-1])/2

    if abs(f(x)) < 1e-12: break
    if abs(x-x_last) <= 1e-9: break
    if f(x) == 0: break
    
    if f(x) * f(v[-1]) < 0:
      u.append(x)
      v.append(v[-1])
    else:
      v.append(x)
      u.append(u[-1])

    x_last = x
  return x

def newton(f, df, x0):
  assert(df(x0) != 0)

  x = x0
  x_last = x

  while True:
    x = x_last - f(x_last)/df(x_last)

    if abs(f(x)) < 1e-12: break
    if abs(x-x_last) <= 1e-9: break
    if f(x) == 0: break

    x_last = x

  return x

def f(x): return x**2 - 2
def df(x): return 2*x

print(bisektion(f,1,2))
print(newton(f,df, (1+2)/2 ))


# In[ ]:




