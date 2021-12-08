#!/usr/bin/env python
# coding: utf-8

# In[180]:


#------------------------------------------------------------------------------#

# Numerik WiSe 2021/22 - Blatt 8
# Aufgabe: 3
# Autor: Jakob Sachs
# Gruppe: NumerikNumerik

#------------------------------------------------------------------------------#

# Importiere Module #

import numpy as np
import matplotlib.pyplot as plt
import scipy  as sp
import time

# Lade StyleSheet #

plt.style.use('fivethirtyeight')


# In[181]:


# Vorbereitung
np.random.seed(2021)
N = 500

A = np.random.rand(N,N)
(_,L,R) = sp.linalg.lu(A)

for i in range(N):
  L[i,i] = 1

y = np.ones(N)
b = np.dot(L,y)


# In[182]:


# Messung Gauss

start = time.time_ns()
for _ in range(N):
  _ = np.linalg.solve(L,b)
duration = time.time_ns() - start
duration /= 1e9

print(f"Gauss:\t\t{duration:.5} seconds total")
print(f"\t\t{duration/N:.5} seconds per iteration")

# Messung Substitution

start = time.time_ns()
for _ in range(N):
  _ = sp.linalg.solve_triangular(L,b,lower=True)
duration = time.time_ns() - start
duration /= 1e9

print(f"Substitution:\t{duration:.5} seconds total")
print(f"\t\t{duration/N:.5} seconds per iteration")


# In[183]:


#------------------------------------------------------------------------------#
# Aufgabe 4
#------------------------------------------------------------------------------#

def vorwaerts(L,b):
  n = b.size
  x = np.zeros(n)

  x[0] = b[0]/L[0,0]

  for i in range(1,n):
    x[i] = b[i]
    for j in range(i):
      x[i] -= L[i,j]*x[j]
    x[i] /= L[i,i]
  return x

def rueckwaerts(R,b):
  n = b.size
  x = np.zeros(n)

  x[n-1] = b[n-1]/R[n-1,n-1]

  for i in range(n-2,0-1,-1):
    x[i] = b[i]
    for j in range(i+1,n):
      x[i] -= R[i,j]*x[j]
    x[i] /= R[i,i]
  return x


# In[184]:


a = np.matrix('1 4 5; 1 6 11; 2 14 31')
l = np.matrix('1 0 0; 1 1 0 ; 2 3 1')
r = np.matrix('1 4 5; 0 2 6 ; 0 0 3')


# In[185]:


y = vorwaerts(l,np.array([17,31,82]))
x = rueckwaerts(r,y)
print(x)


# In[ ]:




