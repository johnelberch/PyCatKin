from pycatkin.constants.physical_constants import *
import numpy as np
from typing import List


def karr(T, prefac, barrier):
    """Calculates reaction rate constant from Arrhenius expression.

    Returns rate constant."""

    k = prefac * np.exp(-barrier / (R * T))

    return k


def kads(T, mass, area):
    """Calculates adsorption rate constant from collision theory.

    Returns rate constant."""

    k = area / np.sqrt(2.0 * np.pi * (mass * amutokg) * kB * T)

    return k


# def kdes(T, mass, area, sigma, theta, des_en):
#     """Calculates desorption rate constant from collision theory.

#     Returns rate constant."""

#     k = ((kB ** 2) * area * 2.0 * np.pi * (mass * amutokg) * (T ** 3)) / (
#             (h ** 3) * sigma * theta) * np.exp(-des_en / (R * T))

#     return k

def kdes(T: float, mass: float, area: float, sigma: float, inertia: List[float], des_en: float):
    """Calculates the desorption rate constant. This modified version accounts for non diatomic molecules,
    which have multiple moments of inertia. 

    Args:
        T (float): Temperature [K]
        mass (float): Molecular mass [g/mol]
        area (float): Area of an active site [m^2] 
        sigma (float): Symmetry number [dimensionless]
        inertia (List[float]): Moments of inertia [amu.A^2] (ASE's units)
        des_en (float): Desorption electronic energy adjusted with ZPE [J/mol]

    Returns:
        float: Desorption rate constant [1/s]
    """   
    inertia = list(inertia) #ensure the moments of inertia is a list

    # Check if there are moments of inertia set to zero. If not, compute as non-linear polyatomic
    if len(inertia) == 3 and all([abs(k) > 0.001 for k in inertia]):
        theta = [h**2 / (8 * np.pi**2 * (I*amuA2tokgm2) * kB) for I in inertia]
        coeff = (kB**2 * T**(7/2) * area * 2 * np.pi**(3/2) * (mass*amutokg)) / (h**3 * sigma * np.prod(theta))
    
    # For all other cases, treat as linear diatomic (take the largest rotational temperature)
    else:
        theta = h**2 / (8 * np.pi**2 * (max(inertia)*amuA2tokgm2) * kB) 
        coeff = (kB**2 * T**3 * area * 2 * np.pi * (mass*amutokg)) / (h**3 * sigma * theta)

    return coeff * np.exp(-des_en / (R * T)) 


def keq_kin(ka, kd):
    """Calculates equilibrium constant from kinetics.

    Returns equilibrium rate constant."""

    k = ka / kd

    return k


def keq_therm(T, rxn_en):
    """Calculates equilibrium constant from thermodynamics.

    Returns equilibrium rate constant."""

    k = np.exp(-rxn_en / (R * T))

    return k


def k_from_eq_rel(kknown, Keq, direction='forward'):
    """Calculates unknown forward/reverse rate constant from equilibrium relation.

    Returns unknown rate constant."""

    if direction == 'forward':
        k = kknown / Keq
    else:
        k = kknown * Keq

    return k


def prefactor(T):
    """Calculates prefactor from transition state theory.

    Returns prefactor."""

    prefac = kB * T / h

    return prefac
