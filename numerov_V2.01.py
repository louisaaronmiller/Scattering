# -------------------- Modules and Packages--------------------

# import scipy.constants as sc
import math as math

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from scipy.special import spherical_jn, spherical_yn

# -------------------- Constants and Parameters--------------------

R = 1  # radius of the potential 1e-15
V_0 = -5  # -10e6 * 1.6e-19
E = 13  # should eventually become an array
rmax = 20
# taking hbar^2 / 2m == 1

# -------------------- Potential, Energy, and Schrodinger terms --------------------

def V(r,R=R):
    if r < R:
        return V_0
    else:

        return 0


def F(l, r, E,R=R):
    if r == 0:
        return 0
    else:
        func_val = V(r,R) + (l * (l + 1)) / (r**2) - E
        return func_val


def k(E, r,R=R):
    if r > R:
        return math.sqrt(E)
    else:
        return math.sqrt(E - V_0)


def Numerov(l: int, E: float,rmax = rmax,R=R, h = 0.01):
    rvals = np.arange(0, rmax, h)
    n = len(rvals)

    u = [
        0.000,
        (h ** (l + 1)),
    ]  # Again recommended in the textbook to use this as the second point
    w0 = (1 - ((h**2) / 12) * F(l, rvals[0], E, R)) * u[0]
    w1 = (1 - ((h**2) / 12) * F(l, rvals[1], E, R)) * u[1]
    w = [w0, w1]

    for i in range(1, n - 1):
        fval = F(l, rvals[i], E, R)
        fval_p1 = F(l, rvals[i + 1], E, R)

        wnp1 = 2 * w[i] - w[i - 1] + (h**2) * fval * u[i]  # w_{n+1}
        unp1 = wnp1 / (1 - (((h**2) / (12)) * fval_p1))  # u_{n+1}

        w.append(wnp1)
        u.append(unp1)

    return u, rvals

# -------------------- Functions for getting r and u values starting from outside the potential --------------------

def outside_vals(rvals, uvals,R=R):
    """
    returns a tuple where the u and r vals start just when the potential "turns off"
    """
    rvals_dict = dict(enumerate(rvals))
    for key, value in rvals_dict.items():
        if V(value,R) != 0:
            continue
        else:
            index = key
            break
    r = rvals[index:]
    u = uvals[index:]
    return u, r


def r_1halfr_2(r, u, E, max_points=2):
    """
    Finds r and u values at successive extrema (half-wavelength separation).
    Works by finding local maxima/minima in u(r).
    """

    u = np.array(u)
    r = np.array(r)

    # Finding indices of local maxima and minima
    extrema_idx = (
        argrelextrema(u, np.greater)[0].tolist() + argrelextrema(u, np.less)[0].tolist()
    )
    extrema_idx.sort()

    extrema_idx = extrema_idx[:max_points]

    r_aug = r[extrema_idx]
    u_aug = u[extrema_idx]

    return r_aug, u_aug

# -------------------- Functions for phase shifts (delta) and total cross section (sigma) --------------------

def K(rvals, uvals):
    """
    rvals here are starting from when there is no potential r > R

    The l value the uvals possesses will determine the l value/subscript
    that the phase shift delta_l will have.
    """
    K_array = []
    for i in range(len(rvals) - 1):
        K = (rvals[i] * uvals[i + 1]) / (rvals[i + 1] * uvals[i])
        K_array.append(K)
    return K_array


def delta_l(l, rvals, kvals, E):
    """
    similar to K, rvals here are starting from when there is no potential r > R
    """

    deltavals = []
    k_0 = math.sqrt(E)

    for i in range(len(rvals) - 1):
        j_l_i = spherical_jn(l, k_0 * rvals[i])
        n_l_i = spherical_yn(l, k_0 * rvals[i])

        j_l_ip1 = spherical_jn(l, k_0 * rvals[i + 1])
        n_l_ip1 = spherical_yn(l, k_0 * rvals[i + 1])

        numerator = kvals[i] * j_l_i - j_l_ip1
        denominator = kvals[i] * n_l_i - n_l_ip1

        delta_i = np.arctan(numerator / denominator)

        deltavals.append(delta_i)
    return deltavals


def sigma(l, delta,E):
    k = math.sqrt(E)
    if l == 0:
        sigma_tot = ((4 * np.pi) / (k**2)) * (np.sin(delta)) ** 2
        return sigma_tot
    else:
        return 0  # Set up later, should be a sum from l=0 to infinity, going to some l that stops when terms after aren't as 'strong'

# -------------------- Delta, Sigma, Energies, Momenta simulation --------------------

def PhaseforEnergy(l: int,E_max = 30,rmax = rmax,R=R, h = 0.001):
    E = np.arange(1, E_max, 0.1)
    delta = []
    k = []
    sigmas = []
    for i in E:
        k.append(np.sqrt(i))

        u, r = Numerov(l, i, rmax, R, h)
        u_aug, r_aug = outside_vals(r, u,R)
        r_new, u_new = r_1halfr_2(r_aug, u_aug, i)

        phaseshift = delta_l(l, r_new, K(r_new, u_new), i)
        # print(f"phase shifts: {phaseshift}")
        delta.append(phaseshift[-1])

        sig = sigma(l,phaseshift[-1],i)
        sigmas.append(sig)
    return delta, E, k, sigmas

# -------------------- Analytical Functions --------------------

def analytical_delta0(E,V_0=V_0,a=R):
    '''
        Calculates analytical phase shift values for l=0

        a = radius of the spherical potential

        Taken from B.H.B & C.J.J Quantum mechanics 2nd Edition,
        they define K as k**2 + V_0 and k^2 as 2mE/hbar**2.
        Here k is defined as sqrt(E) and K as sqrt(E+V_0),
        assuming a attractive potential.
    '''
    k = math.sqrt(E)
    K = math.sqrt(E-V_0) # Because i've defined V_0 has negative the minus signs cancels eachother out


    numerator = k * np.tan(K*a) - K * np.tan(k*a)
    denominator = K + (k * np.tan(k*a) * np.tan(K*a))
    delta = np.arctan(numerator/denominator) #math.atan2

    return delta

def analytical_deltal(E,l,V_0=V_0,a=R):
    '''
        The same function as analytical_delta0, but for l>0
    '''
    k = math.sqrt(E)
    K = math.sqrt(E-V_0)
    # -------- k --------

    j_l_k = spherical_jn(l, k * a)
    n_l_k = spherical_yn(l, k * a)

    j_l_k_prime = spherical_jn(l, k * a, derivative=True)
    n_l_k_prime = spherical_yn(l, k * a, derivative=True)

    # -------- K --------

    j_l_K = spherical_jn(l, K * a)
    #n_l_K = spherical_yn(l, K * a)

    j_l_K_prime = spherical_jn(l, K * a, derivative=True)
    #n_l_K_prime = spherical_yn(l, K * a, derivative=True)

    # -------- Calculation --------

    numerator = (k * j_l_k_prime * j_l_K) - (K * j_l_k * j_l_K_prime)
    denominator = (k * n_l_k_prime * j_l_K) - (K * n_l_k * j_l_K_prime)

    delta = np.arctan(numerator/denominator) #math.atan2

    return delta

def analytical_sigma0(delta,E):
    k = math.sqrt(E)

    sig = ((4 * np.pi) / (k **2)) * (np.sin(delta))**2
    
    return sig
# -------------------- Resonance (Bound states) --------------------

def delta_res(l, rvals, kvals, E):
    """
    creates delta values for 
    """

    deltavals = []
    k_0 = math.sqrt(E)

    for i in range(len(rvals) - 1):
        j_l_i = spherical_jn(l, k_0 * rvals[i])
        n_l_i = spherical_yn(l, k_0 * rvals[i])

        j_l_ip1 = spherical_jn(l, k_0 * rvals[i + 1])
        n_l_ip1 = spherical_yn(l, k_0 * rvals[i + 1])

        numerator = kvals[i] * j_l_i - j_l_ip1
        denominator = kvals[i] * n_l_i - n_l_ip1

        delta_i = np.arctan(numerator / denominator)

        deltavals.append(delta_i)
    return deltavals


def findingResonance(l,E=1e-3,V_0=V_0,h=0.005,rmax = 5):
    '''
        this function finds the zero energy resonance of a system (bound states),
        it retuns gamma and alpha which are the resonance relation and scattering
        length respectively, the resonance relation shows that zero-energy resonance
        occurs whenever gamma is an odd multiple of pi/2

        How gamma and alpha are related:

        alpha = (1-tan(gamma)/gamma)*a

        gamma = V_0^{1/2} *a

        where a is the radius of the potential sphere
        E is the energy of the particle
        V_0 is the potential energy -> the reason for this minus is because we use it
        in the equation above, so we are still dealing with an attractive potential.
    '''
    a = np.arange(0.5,4,h)
    delta = []
    sigmas = []
    alpha_arr = []
    gamma_arr= []

    res_indexs = []
    for i in a:
        
        u, r = Numerov(l,E,rmax = rmax,R=i,h=h)
        u_aug, r_aug = outside_vals(r, u,R=i)
        r_new, u_new = r_1halfr_2(r_aug, u_aug, E)

        phaseshift = delta_l(l, r_new, K(r_new, u_new), E)
        delta.append(phaseshift[-1])

        sig = sigma(l,phaseshift[-1],i)
        sigmas.append(sig)
        
        gamma = math.sqrt(abs(V_0)) * i
        gamma_arr.append(gamma)
        alpha = (1- (np.tan(gamma))/(gamma)) * i
        alpha_arr.append(alpha)
    for key, val in enumerate(gamma_arr): # Extremely useless. However, might be useful later?
        if abs(val-np.pi/2) < 1e-2 or abs(val-3*(np.pi/2)) < 1e-2 or abs(val- 5 *(np.pi/2)) < 1e-2:
            res_indexs.append(key)


    return delta, sigmas, alpha_arr, gamma_arr, res_indexs, a

def findingResWave(l,E=1e-3,V_0=V_0,h=0.005,rmax = 5):
    '''
       finding wavefunctions of correpsonding bound states
    '''
    a = np.arange(0.5,4,h)

    first_res_r= []
    second_res_r = []
    third_res_r = []

    first_res_u = []
    second_res_u = []
    third_res_u = []

    _,_,_,_,index,a = findingResonance(l,E,V_0,h=h,rmax=rmax)
    res_array = [a[index[0]],a[index[1]],a[index[2]]]
    j = 0
    for i in res_array:
        u, r = Numerov(l,E,rmax = rmax,R=i,h=h)
        if j == 0:
            first_res_r.append(r)
            first_res_u.append(u)
            j+=1
        if j == 1:
            second_res_r.append(r)
            second_res_u.append(u)
            j+=1
        if j == 2:
            third_res_r.append(r)
            third_res_u.append(u)
    return first_res_r,first_res_u,second_res_r,second_res_u,third_res_r,third_res_u



# -------------------- Result taking --------------------

# -------- Numerical --------
'''
delta0, Elist0, ki0, sigma0 = PhaseforEnergy(0)

delta1, Elist1, ki1, sigma1 = PhaseforEnergy(1)
delta2, Elist2, ki2, sigma2 = PhaseforEnergy(2)
delta3, Elist3, ki3, sigma3 = PhaseforEnergy(3)

# -------- Analytical --------

a_delta0 = [analytical_delta0(i) for i in Elist0]
a_delta1 = [analytical_deltal(i,1) for i in Elist1]
a_delta2 = [analytical_deltal(i,2) for i in Elist2]
a_delta3 = [analytical_deltal(i,3) for i in Elist3]

a_sigma0 = [analytical_sigma0(j,i) for j,i in zip(a_delta0,Elist0)]
'''
# -------- Resonance --------

delta_r_0, sigma_r_0, scattering_length_0, gamma_r_0, res_indexs_0, a_vals_0 = findingResonance(0,E=1e-3,V_0=-V_0,h=0.01,rmax = 200)
r1,u1,r2,u2,r3,u3 = findingResWave(0,E=1e-3,V_0=-V_0,h=0.01,rmax = 200)

# -------------------- Plotting --------------------

'''
fig, axs = plt.subplots(2, 2, figsize=(12, 8))  
axs = axs.ravel()  

# --- Phase shift l=0 ---

axs[0].plot(ki0, delta0, label="$l=0$ - Numerical")
axs[0].plot(ki0, a_delta0, 'k--', label="$l=0$ - Analytical")
axs[0].set_title("Phase shift vs Momentum")
axs[0].set_xlabel("Momentum ($k$, where $\\hbar = 1$)")
axs[0].set_ylabel("Phase shift (Radians)")
axs[0].minorticks_on()
axs[0].grid()
axs[0].legend()

# --- Phase shift l=1 ---
axs[1].plot(ki1, delta1, label="$l=1$ - Numerical")
axs[1].plot(ki1, a_delta1, 'r--', label="$l=1$ - Analytical")
axs[1].set_title("Phase shift vs Momentum")
axs[1].set_xlabel("Momentum ($k$, where $\\hbar = 1$)")
axs[1].set_ylabel("Phase shift (Radians)")
axs[1].minorticks_on()
axs[1].grid()
axs[1].legend()

# --- Phase shift l=2 ---
axs[2].plot(ki2, delta2, label="$l=2$ - Numerical")
axs[2].plot(ki2, a_delta2, 'b--', label="$l=2$ - Analytical")
axs[2].set_title("Phase shift vs Momentum")
axs[2].set_xlabel("Momentum ($k$, where $\\hbar = 1$)")
axs[2].set_ylabel("Phase shift (Radians)")
axs[2].minorticks_on()
axs[2].grid()
axs[2].legend()

# --- Phase shift l=3 ---
axs[3].plot(ki3, delta3, label="$l=3$ - Numerical")
axs[3].plot(ki3, a_delta3, 'g--', label="$l=3$ - Analytical")
axs[3].set_title("Phase shift vs Momentum")
axs[3].set_xlabel("Momentum ($k$, where $\\hbar = 1$)")
axs[3].set_ylabel("Phase shift (Radians)")
axs[3].minorticks_on()
axs[3].grid()
axs[3].legend()

plt.tight_layout()

# -------- Total cross section (l=0) --------
   
plt.plot(ki0,sigma0, label='$l=0$ - Numerical')                                    
plt.plot(ki0,a_sigma0, 'k--', label='$l=0$ - Analytical')

plt.title("Total cross section vs Momentum")
plt.ylabel("Total cross section (m$^2$)")
plt.xlabel("Momentum ($k$, where $\\hbar = 1$)")

'''

# -------------------- Resonance Plotting (l=0) --------------------

# -------- delta vs gamma--------
'''
plt.plot([np.pi/2]*50,np.linspace(-20,20,50),'r--' , linewidth=1)
plt.plot([(3 * np.pi)/2]*50,np.linspace(-20,20,50),'r--', linewidth=1)
plt.plot([(5 * np.pi)/2]*50,np.linspace(-20,20,50),'r--', linewidth=1)

plt.plot(gamma_r_0, delta_r_0)

plt.text(2, 1, '$\\frac{\\pi}{2}$', ha='center', va='top', fontsize=24)
plt.text(5.3, 1, '$\\frac{3\\pi}{2}$', ha='center', va='top', fontsize=24)
plt.text(8.5, 1, '$\\frac{5\\pi}{2}$', ha='center', va='top', fontsize=24)
plt.ylim(-1.25,1.25)

plt.title("Zero-Energy Resonance ($\\delta$ against $\\gamma$ for $l=0$)")
plt.ylabel("Phaseshift $\\delta$")
plt.xlabel("Potential Energy-Radii Relation $\\gamma$")


# -------- sigma vs gamma --------

plt.plot([np.pi/2]*50,np.linspace(-20,20,50),'r--' , linewidth=1)
plt.plot([(3 * np.pi)/2]*50,np.linspace(-20,20,50),'r--', linewidth=1)
plt.plot([(5 * np.pi)/2]*50,np.linspace(-20,20,50),'r--', linewidth=1)

plt.plot(gamma_r_0, sigma_r_0)

plt.text(2, 17, '$\\frac{\\pi}{2}$', ha='center', va='top', fontsize=24)
plt.text(5.3, 6, '$\\frac{3\\pi}{2}$', ha='center', va='top', fontsize=24)
plt.text(8.5, 3, '$\\frac{5\\pi}{2}$', ha='center', va='top', fontsize=24)
plt.ylim(-0.5,17.5)

plt.title("Zero-Energy Resonance ($\\sigma$ against $\\gamma$ for $l=0$)")
plt.ylabel("Total Cross Section $\\sigma$")
plt.xlabel("Potential Energy-Radii Relation $\\gamma$")

'''
# -------- Phaseshift vs momentum for each resonance 0,1,2 --------





# -------- Wavefunction for resonance --------

'''
# plt.scatter(r_augNew,u_augNew, label='half wavelength')
# plt.plot(Elist,deltadeg)
# plt.plot(ki,delta)
# plt.plot(r, u)
# plt.plot(r_aug[:-1],phaseshift_vals)

plt.plot(r1[0],u1[0],label='pi/2')
plt.scatter(r2[0],u2[0],label='3pi/2')
plt.scatter(r3[0],u3[0],label='5pi/2')
'''


plt.legend()
plt.minorticks_on()
plt.grid()
plt.show()
