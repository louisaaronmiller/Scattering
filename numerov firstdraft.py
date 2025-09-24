import numpy as np
import scipy.constants as sc
import math as math
import matplotlib.pyplot as plt


R = 1 # radius of the potential 1e-15
V_0 = 5 #-10e6 * 1.6e-19
E = 15 # should eventually become an array
rmax = 10
eps=1e-5

# taking hbar^2 / 2m == 1

def V(r):
    if r <= R:
        return V_0
    else:
        return 0
    
    
    '''
def k(E:float):
    return math.sqrt(2 * m * E)/sc.hbar

def F(l:int, r: float, E:float):
    func_val = V(r) + (sc.hbar * l * (l_1))/(2 * m * r **2) - E
    '''
def F(l,r,E):
    if r == 0:
        return 0
    else:
        func_val = V(r) + (l *(l+1))/(r**2) - E
        return func_val
    
def k(E):
    return math.sqrt(E)
    

def Numerov(l:int, E:float):
    #answer = [] # placeholder
    h = np.pi/k(E) # step size (half the wavelength as recommended in book)
    rvals = np.arange(0,rmax,h)
    n = len(rvals)

    u = [0.0,eps]
    w0 = (1-((h**2)/12) * F(l,rvals[0],E)) * u[0]
    w1 = (1-((h**2)/12) * F(l,rvals[1],E)) * u[1]
    w = [w0,w1]
    
    for i in (range(1, n- 1)):
        fval = F(l,rvals[i],E)
        fval_p1 = F(l,rvals[i+1],E)
        
        wnp1 = 2*w[i] - w[i-1] + (h**2) * fval * u[i] #w_{n+1} 
        unp1 =  wnp1/(1-((h**2)/(12)) * fval_p1)    #u_{n+1} 
        
        w.append(wnp1)
        u.append(unp1)
        
    return u, rvals

u, r = Numerov(l=0,E=15)

plt.plot(r, u)
plt.title('Numerov')
plt.minorticks_on()
plt.grid()
plt.show()
        
    
    