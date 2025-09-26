import numpy as np
#import scipy.constants as sc
import math as math
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn

R = 1 # radius of the potential 1e-15
V_0 = 5 #-10e6 * 1.6e-19
E = 6 # should eventually become an array
rmax = 10
eps=1e-5

# taking hbar^2 / 2m == 1

def V(r):
    if r <= R:
        return V_0
    else:
        
        return 0
    
def F(l,r,E):
    if r == 0:
        return 0
    else:
        func_val = V(r) + (l *(l+1))/(r**2) - E
        return func_val
    
def k(E,r):
    if r <= R:
        return math.sqrt(E)
    else:
        return math.sqrt(V_0)
    

def Numerov(l:int, E:float):
    #answer = [] # placeholder
    h = 0.001 # step size (np.pi/k(E) half the wavelength as recommended in book)
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


def outside_vals(rvals,uvals):
    '''
    returns a tuple where the u and r vals start just when the potential "turns off"
    '''
    rvals_dict = dict(enumerate(rvals))
    for key,value in rvals_dict.items():
        if V(value) != 0:
            continue
        else:
            index = key
            break
    r = rvals[index:]
    u = uvals[index:]
    return u, r
            

def K(rvals,uvals):
    '''
    rvals here are starting from when there is no potential r > R
    
    The l value the uvals possesses will determine the l value/subscript
    that the phase shift delta_l will have.
    '''
    K_array = []
    for i in range(len(rvals)-1):
        K = (rvals[i] * uvals[i+1])/(rvals[i+1] * uvals[i])
        K_array.append(K)
    return K_array


def delta_l(l, rvals,kvals):
    '''
    similar to K, rvals here are starting from when there is no potential r > R
    '''
    deltavals = []
    k_0 = math.sqrt(E)
    for i in range(len(rvals)- 1):
        j_l_i = spherical_jn(l,k_0 * rvals[i])
        n_l_i = spherical_yn(l,k_0 * rvals[i])
        
        j_l_ip1 = spherical_jn(l,k_0 * rvals[i+1])
        n_l_ip1 = spherical_yn(l,k_0 * rvals[i+1])
        
        tandelta = (kvals[i] * j_l_i - j_l_ip1)/(kvals[i] * n_l_i - n_l_ip1)
        delta = np.arctan(tandelta)
        
        deltavals.append(delta)
    return deltavals

'''
TODO = implement sigma function, that sums over until hbar * l_max = hbar * k * r_max
or choose an l value to go up to  

= comprise the things below into a function that loops over for an array of Energies   
'''



u, r = Numerov(l=0,E=15)#

u_aug, r_aug = outside_vals(r,u)    # These are u and r vals that start just outside the potential

x = delta_l(l=0,rvals = r_aug,kvals = K(r_aug,u_aug))
print(len(x))


plt.plot(r, u)
plt.title('Numerov')
plt.minorticks_on()
plt.grid()
plt.show()
        
    
    
