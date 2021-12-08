#!/usr/bin/env python
# coding: utf-8

# In[16]:


# Importiere Module #

import numpy as np
import matplotlib.pyplot as plt
import time 

# Lade StyleSheet #

plt.style.use('fivethirtyeight')


# 

# In[17]:


def cramer(A,b):
  n = np.linalg.det(A)
  x = []
  for i in range(A.shape[1]):
    H = np.hstack( [ A[:,:i], b.reshape(-1,1), A[:,i+1:] ] )
    x.append(np.linalg.det(H)/n)
  return x


# In[18]:





for i in [50,100,500]:
  A = np.random.rand(i,i)
  x = np.ones(i)
  b = np.dot(A,x)


  print(f"{i}x{i} Matrix:")
  
  if i <= 100:

    start = time.time_ns()
    for j in range(i):
      _ = cramer(A,b)
    duration = time.time_ns() - start
    duration /= 1e9
    print(f"Cramer:  {duration:.5} seconds")

  start = time.time_ns()
  for _ in range(i):
    A_inv = np.linalg.inv(A)
    _ = np.dot(A_inv,b)
  duration = time.time_ns() - start
  duration /= 1e9

  print(f"Inverse:  {duration:.5} seconds")

  start = time.time_ns()
  for _ in range(i):
    _ = np.linalg.solve(A,b)
  duration = time.time_ns() - start
  duration /= 1e9

  print(f"np-Solve:  {duration:.5} seconds")



# In[ ]:





# In[ ]:




