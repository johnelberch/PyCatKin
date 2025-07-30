"""Ensuring that the energies are consistent with expected values
PyCatKin's documentation is WAY better than CatMap... but it remains a bit obscure (specially,
what units to use and other details). I am manually veryfying that all my calculations make sense,
i.e., I can replicate what ASE would predict, and my MonkeyPatched functions work
"""
#%% 
import os
import sys

module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(module_dir)
os.chdir(os.path.dirname(__file__))

import numpy as np  # noqa: E402
from ase.build import molecule  # noqa: E402
from ase.thermochemistry import HarmonicThermo, IdealGasThermo  # noqa: E402
from ase.units import invcm  # noqa: E402
from monkey_patching_functions import (new_calc_rate_constants,  # noqa
                                       new_calc_electronic_energy,
                                       new_calc_free_energy)
from pycatkin.classes.state import ScalingState  # noqa: E402
from pycatkin.functions.load_input import read_from_input_file  # noqa: E402
from pycatkin.classes.reaction import Reaction

# Monkey patching the ScalingState class so it accepts multiple descriptor coefficients
ScalingState.calc_free_energy = new_calc_free_energy
ScalingState.calc_electronic_energy = new_calc_electronic_energy
Reaction.calc_rate_constants = new_calc_rate_constants


#%% 
#--------------------------------------------------
## Loading the inputs
sim_system = read_from_input_file("../CH4_input.json")

# States and reactions
rxn = sim_system.reactions["R1"]
sCO = sim_system.states["sCO"]
sTS = sim_system.states["sC-H--OH"]
gO2 = sim_system.states["O2"]

# Note the temperature and pressure
T = sim_system.params['temperature']
p = sim_system.params['pressure']

# Set the descriptors' energies
EC = 1.5
EO = 0.2
sim_system.reactions["C_ads"].dErxn_user = EC
sim_system.reactions["O_ads"].dErxn_user = EO

## Manually added inputs
# CO ADSORPTION
sCO_coeffs = [0.45, 0, 0.51]
sCO_vib_cm = [2040.0, 306.9, 268.2, 261.1, 99.7, 68.7]

# C-H--OH TS
sTS_coeffs = [0.89, 0.46, 0.29]
sTS_vib_cm = [3705.1, 1298.0, 1012.1, 688.3, 613.0, 435.1, 420.5, 358.6, 310.2, 215.0, 12.2] 

# O2 GAS
O2_vib = [1543.5]
O2_sigma = 2
O2_energy = 5.48
O2_mass = 31.998
O2_inertia = [0.0, 12.418474628311035, 12.418474628311035]
O2_atoms = molecule("O2")


#%%
#--------------------------------------------------
## Testing the States
def my_round(n, n_dec=4):
    return np.ceil(n*10**n_dec)/10**n_dec

def test_energy(s, EC, EO, coeff, vib_cm):
    """Tests if the predicted energies of a state matches what is expected
    
    Args:
        s: State or ScalingState from PyCatKin
        EC: Energy of C descriptor
        EO: Energy of O descriptor
        vib_cm: Vib frequencies in cm^-1 (list)
    """
    ## Compute with PyCatKin
    # Electronic energy (Sets Gelec)
    s.calc_electronic_energy()
    # Zero point energy contribution (Sets Gzpe)
    s.calc_zpe()
    # Free energy (electronic, vibrational, translational, rotational). Note, Transl and Rot are only for gases. Sets Gfree
    s.calc_free_energy(T,p)


    ## Compute with ASE
    # Convert vibrational freq units
    vib_eV = [invcm * k for k in vib_cm]
    # Compute Electronic Energy
    E_pred = np.dot(np.array(coeff), np.array([EC, EO, 1]))
    # Set Harmonic Oscillator Object
    ht = HarmonicThermo(vib_eV, E_pred)
    # zpe
    ZPE_pred = ht.get_ZPE_correction()
    # Free energy
    A_pred = ht.get_helmholtz_energy(T, verbose=False)


    ## Make the comparissons
    assert my_round(E_pred, 6) == my_round(s.Gelec, 6), "Electronic energies don't match"
    assert my_round(ZPE_pred, 3) == my_round(s.Gzpe, 3), "ZPE don't match"
    assert my_round(A_pred, 3) == my_round(s.Gfree, 3), "Free energies don't match"
    print("ALL TESTS PASSED")

def test_gas_energy(g, E, vib_cm, sigma, mass, inertia, atoms):
    ## Compute with PyCatKin
    # Free energy (electronic, vibrational, translational, rotational). Note, Transl and Rot are only for gases. Sets Gfree
    g.calc_free_energy(T,p)

    ## Compute with ASE
    # Convert vibrational freq units
    vib_eV = [invcm * k for k in vib_cm]
    it = IdealGasThermo(vib_eV, geometry="linear", potentialenergy=E, natoms=2, 
    atoms=atoms, symmetrynumber=sigma, spin=1)
    G_pred = it.get_gibbs_energy(T, p, verbose=False)

    print(g.Gfree, G_pred)


#%%
test_energy(sCO, EC, EO, sCO_coeffs, sCO_vib_cm)
test_energy(sTS, EC, EO, sTS_coeffs, sTS_vib_cm)
test_gas_energy(gO2, O2_energy, O2_vib, O2_sigma, O2_mass, O2_inertia, O2_atoms)


#%%
#--------------------------------------------------
## Testing the Reaction
# Compute with PyCatKin
rxn.calc_rate_constants(T, p) #sets kfwd, Keq, krev
rxn_energy = rxn.get_reaction_energy(T, p, verbose=False)
rxn_barriers = rxn.get_reaction_barriers(T,p, verbose=False)

state_dictio = { #([coeffs], [vib freq])
    "IS" : {
        "sCH3" : (
            [0.239785047,0,0.136587444], 
            [95.8, 103.5, 226.0, 278.8, 545.5, 547.6, 1166.3, 1400.4, 1403.8, 2944.3, 3014.7, 3016.4],
            )
    },
    "TS" : {
        "sCH2-H" : (
            [0.618905821,0,0.19638489],
            [3080.5, 3007.1, 1406.2, 1361.2, 822.3, 622.7, 534.5, 442.3, 340.2, 222.8, 77.0],
        )
    },
    "FS" : {
        "sCH2" : (
            [0.494635, 0, 0.232988],
            [152.0, 257.5, 305.9, 416.3, 434.8, 643.3, 1329.9, 2947.9, 3008.0],
        ),
        "hH" : (
            [0.219820574, 0, -0.785276035],
            [978.2, 768.0, 764.8],
        ),
    },
}

# Manual calculation
def compute_rxn_energies(state_dictio:dict, return_G = True):
    L_IS = []
    L_TS = []
    L_FS = []

    for state, L in zip(["IS","TS","FS"], [L_IS, L_TS, L_FS]):
        for k,v in state_dictio[state].items():
            vib_eV = [invcm * k for k in v[1]]
            E_pred = np.dot(np.array(v[0]), np.array([EC, EO, 1]))
            ht = HarmonicThermo(vib_eV, E_pred)
            if return_G:
                L.append(ht.get_helmholtz_energy(T, verbose=False))
            else:
                L.append(E_pred + ht.get_ZPE_correction())
    
    return L_IS, L_TS, L_FS

IS, TS, FS = compute_rxn_energies(state_dictio=state_dictio)

# print(f"Reaction energy: PyCatKin {rxn_energy/1000} , ASE {(sum(FS) - sum(IS))*96.485} kJ/mol")
# print(f"Fwd activation: PyCatKin {rxn_barriers[0]/1000} , ASE {(sum(TS) - sum(IS)) * 96.485} kJ/mol")
# print(f"Bkw activation: PyCatKin {rxn_barriers[1]/1000} , ASE {(sum(TS) - sum(FS)) * 96.485} kJ/mol")

def get_k(state_dictio, T):
    IS, TS, FS = compute_rxn_energies(state_dictio, return_G = True)
    kB = 1.38064900e-26 #kJ/K
    R = 8.314e-3 #kJ/mol.K
    h = 6.62607015e-37 #kJ.s
    #fwd
    kfwd = kB*T/h * np.exp(-(sum(TS)-sum(IS))*96.485/(R*T))
    kbkw = kB*T/h * np.exp(-(sum(TS)-sum(FS))*96.485/(R*T))
    return kfwd, kbkw

kf, kb = get_k(state_dictio=state_dictio, T=T)
print(f"PRED: kf:{kf:.3e}, kb:{kb:.3e}\nPyCatKin: kf:{rxn.kfwd.item():.3e}, kb:{rxn.krev.item():.3e}")

# %%
# Test cases that do not get the electronic energies calculated (should only be descriptors)
for name in sim_system.states:
    s = sim_system.states[name]
    try:
        s.calc_electronic_energy()
    except Exception as e:
        print(name)
        print(e)
# %%
