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

def V(r):
    if r < R:
        return V_0
    else:

        return 0


def F(l, r, E):
    if r == 0:
        return 0
    else:
        func_val = V(r) + (l * (l + 1)) / (r**2) - E
        return func_val


def k(E, r):
    if r > R:
        return math.sqrt(E)
    else:
        return math.sqrt(E - V_0)


def Numerov(l: int, E: float, h = 0.01):
    rvals = np.arange(0, rmax, h)
    n = len(rvals)

    u = [
        0.000,
        (h ** (l + 1)),
    ]  # Again recommended in the textbook to use this as the second point
    w0 = (1 - ((h**2) / 12) * F(l, rvals[0], E)) * u[0]
    w1 = (1 - ((h**2) / 12) * F(l, rvals[1], E)) * u[1]
    w = [w0, w1]

    for i in range(1, n - 1):
        fval = F(l, rvals[i], E)
        fval_p1 = F(l, rvals[i + 1], E)

        wnp1 = 2 * w[i] - w[i - 1] + (h**2) * fval * u[i]  # w_{n+1}
        unp1 = wnp1 / (1 - (((h**2) / (12)) * fval_p1))  # u_{n+1}

        w.append(wnp1)
        u.append(unp1)

    return u, rvals

# -------------------- Functions for getting r and u values starting from outside the potential --------------------

def outside_vals(rvals, uvals):
    """
    returns a tuple where the u and r vals start just when the potential "turns off"
    """
    rvals_dict = dict(enumerate(rvals))
    for key, value in rvals_dict.items():
        if V(value) != 0:
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


def sigma(l, delta):
    k = math.sqrt(E)
    if l == 0:
        sigma_tot = ((4 * np.pi) / (k**2)) * (np.sin(delta)) ** 2
        return sigma_tot
    else:
        return 0  # Set up later, should be a sum from l=0 to infinity, going to some l that stops when terms after aren't as 'strong'

# -------------------- Delta, Sigma, Energies, Momenta simulation --------------------

def PhaseforEnergy(l: int,E_max = 30):
    E = np.arange(1, E_max, 0.1)
    delta = []
    k = []
    sigmas = []
    for i in E:
        k.append(np.sqrt(i))

        u, r = Numerov(l, i)
        u_aug, r_aug = outside_vals(r, u)
        r_new, u_new = r_1halfr_2(r_aug, u_aug, i)

        phaseshift = delta_l(l, r_new, K(r_new, u_new), i)
        # print(f"phase shifts: {phaseshift}")
        delta.append(phaseshift[-1])

        # sig = sigma(l,phaseshift[-1])
        # sigmas.append(sig)
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
    delta = math.atan2(numerator,denominator)

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

    delta = math.atan2(numerator,denominator)

    return delta

def analytical_sigma0(delta,E):
    k = math.sqrt(E)

    sig = ((4 * np.pi) / (k **2)) * (np.sin(delta))**2
    
    return sig

# -------------------- Result taking --------------------

delta0, Elist0, ki0, sigma0 = PhaseforEnergy(0)

delta1, Elist1, ki1, sigma1 = PhaseforEnergy(1)
delta2, Elist2, ki2, sigma2 = PhaseforEnergy(2)
delta3, Elist3, ki3, sigma3 = PhaseforEnergy(3)

a_delta0 = [analytical_delta0(i) for i in Elist0]
a_delta1 = [analytical_deltal(i,1) for i in Elist1]
a_delta2 = [analytical_deltal(i,2) for i in Elist2]
a_delta3 = [analytical_deltal(i,3) for i in Elist3]

# -------- Phase shifts (l=0) --------
plt.plot(ki0, delta0, label="$l=0$ - Numerical")
plt.plot(ki0, a_delta0, 'k--' , label= '$l=0$ - Analytical')

# -------- Phase shifts (l=1) --------
plt.plot(ki1, delta1, label="$l=1$ - Numerical")
plt.plot(ki1, a_delta1, 'r--' , label= '$l=1$ - Analytical')

# -------- Phase shifts (l=2) --------
plt.plot(ki2, delta2, label="$l=2$ - Numerical")
plt.plot(ki2, a_delta2, 'b--' , label= '$l=2$ - Analytical')

# -------- Phase shifts (l=3) --------
plt.plot(ki3, delta3, label="$l=3$ - Numerical")
plt.plot(ki3, a_delta3, 'g--' , label= '$l=3$ - Analytical')


#plt.plot(Elist1, delta1, label="$l=1$")
#plt.plot(Elist2, delta2, label="$l=2$")
#plt.plot(Elist3, delta3, label="$l=3$")

# plt.plot(Elist,sigma0)

# plt.scatter(r_augNew,u_augNew, label='half wavelength')
# plt.plot(Elist,deltadeg)
# plt.plot(ki,delta)
# plt.plot(r, u)
# plt.plot(r_aug[:-1],phaseshift_vals)

plt.title("Phase shift vs Energy")
plt.ylabel("Phase shift (Radians)")
plt.legend()
plt.xlabel("Momentum ($k$, where $\\hbar = 1$)")
plt.minorticks_on()
plt.grid()
plt.show()