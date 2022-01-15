#!/usr/bin/env python
# coding: utf-8

# In[46]:


#------------------------------------------------------------------------------#

# Numerik WiSe 2021/22 - Blatt 10
# Aufgabe: 4
# Autor: Jakob Sachs
# Gruppe: NumerikNumerik

#------------------------------------------------------------------------------#

# Importiere Module #
import numpy as np


# In[47]:


def bisektion(f , a, b):
  assert(a < b)
  assert(f(a)*f(b) < 0)

  u = a
  v = b

  x = (a+b)/2
  x_last = 0

  while True:
    x = (u-v) / 2

    if abs(f(x)) < 1e-12: break
    if abs(x-x_last) <= 1e-9: break
    if f(x) == 0: break
    
    if f(x) * f(v) < 0:
      u = x
      v = v
    else:
      v = x
      u = x
    x_last = x

  return x

def newton(f, df, x0):
  assert(df(x0) != 0)

  x = x0
  x_last = x
  while True:
    x = x_last - f(x_last) / df(x_last)
    
    if f(x) == 0: break
    if abs(f(x)) < 1e-12: break
    if abs(x-x_last) <= 1e-9: break
    
    x_last = x

  return x

def dekker_brent(f,a,b):
  assert(a < b)
  assert(f(a)*f(b) < 0)

  c_last = 0  
  while True:
    c = (a+b)/2

    if f(c) == 0: break
    if abs(f(c)) < 1e-12: break
    if abs(c-c_last) <= 1e-9: break
   
    xi = c 
    xi -= f(c)*((b-c)*(f(b)-f(c))) 
    xi += (f(c) * f(b) * (((a-b)/(f(a)-f(b))-(b-c)/(f(b)-f(c)))/(a-c)))

    if  a <= xi  <= c:
      a = a
      b = c
    else:
      a = c
      b = b
    c_last = c
  return c


# In[48]:


# Wir testen unsere drei Verfahren fuer f(x) = x^2 - 2 auf [1,2]

def f(x): return x**2 - 2
def df(x): return 2*x # Ableitung fuer Newton-Verfahren

print("[Bisektion]\tx = ",bisektion(f,1,2),"\tf(x) = ",f(bisektion(f,1,2)))
print("[Newton]\tx = ",newton(f,df, (1+2)/2 ),"\tf(x) = ",f(newton(f,df, (1+2)/2 )))
print("[Dekker-Brent]\tx = ",dekker_brent(f,1,2),"\tf(x) = ",f(dekker_brent(f,1,2)))
print("[Original]\tx = ", np.sqrt(2),"\tf(x) = ",f(np.sqrt(2)))


# In[ ]:




