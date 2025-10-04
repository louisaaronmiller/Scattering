# import scipy.constants as sc
import math as math

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from scipy.special import spherical_jn, spherical_yn

R = 1  # radius of the potential 1e-15
V_0 = -5  # -10e6 * 1.6e-19
E = 13  # should eventually become an array
rmax = 20
# taking hbar^2 / 2m == 1


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


def Numerov(l: int, E: float):
    h = 0.01
    rvals = np.arange(0, rmax, h)
    n = len(rvals)

    u = [
        0.0,
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


def PhaseforEnergy(l: int):
    E = np.arange(1, 30, 0.1)
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


def radtodeg(array: list):
    deg = []
    for i in array:
        calc = np.rad2deg(i)
        deg.append(calc)
    return deg


# Result taking

delta, Elist, ki, sigma = PhaseforEnergy(0)

delta1, Elist1, ki1, sigma1 = PhaseforEnergy(1)
delta2, Elist2, ki2, sigma2 = PhaseforEnergy(2)
delta3, Elist3, ki3, sigma3 = PhaseforEnergy(3)


plt.plot(Elist, delta, label="$l=0$")
plt.plot(Elist1, delta1, label="$l=1$")
plt.plot(Elist2, delta2, label="$l=2$")
plt.plot(Elist3, delta3, label="$l=3$")

# plt.plot(Elist,sigma)

# plt.scatter(r_augNew,u_augNew, label='half wavelength')
# plt.plot(Elist,deltadeg)
# plt.plot(ki,delta)
# plt.plot(r, u)
# plt.plot(r_aug[:-1],phaseshift_vals)

plt.title("Phase shift vs Energy")
plt.ylabel("Phase shift (Radians)")
plt.legend()
plt.xlabel("Energy")
plt.minorticks_on()
plt.grid()
plt.show()
