from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import copy

import numpy as np
from pycatkin.constants.physical_constants import (
    JtoeV,
    amuA2tokgm2,
    amutokg,
    eVtokJ,
    h,
    kB,
)
from pycatkin.settings import ScalingStateSettings, StateSettings
from pydantic import TypeAdapter


class BaseState(ABC):
    def __init__(self, Gelec: float|None, state_type: str, name: str):
        """Basic state definition (defines the bare-bones contract all State classes must follow)

        Args:
            Gelec (float | None): Electronic energy (computed with DFT, in eV)
            state_type (str): Descriptor identifying the type of the state
            name (str): Unique nametag for this state
        """
        self.Gelec = Gelec
        self._state_type = state_type
        self._name = name
        self._Gfree = None

    @abstractmethod
    def get_free_energy(self) -> float:
        """Computes the Gibbs Free Energy"""
        pass

    @abstractmethod
    @property
    def name(self) -> str:
        return self._name
    
    @abstractmethod
    @property
    def Gfree(self) -> float:
        return self._Gfree

    @abstractmethod
    @property
    def state_type(self) -> str:
        return self._state_type
    

def _preprocess_frequencies(freq: Sequence[float], truncate_freq:bool = True, cutoff:float = 12.4 * 1e-3 / (h * JtoeV)) -> Sequence[float]:
    """Helping function to preprocess frequencies in frequency-relevant classes

    Args:
        freq (Sequence[float]): Vibrational Frequencies in [1/s]
        truncate_freq (bool, optional): Trim the frequencies below cutoff. Defaults to True.
        cutoff (float, optional): Defaults to 0.0124 eV [100 cm^-1].

    Returns:
        Sequence[float]: processed frequencies
    """
    sorted_freq = np.array(sorted(freq, reverse=True))
    if truncate_freq:
        a = sorted_freq.copy()
        sorted_freq = np.where(a >= cutoff, a, cutoff)
    return sorted_freq


class SurfaceState(BaseState):
    def __init__(self, Gelec: float, state_type: str, name: Literal["surface"]):
        """Clean surface state class.

        Raises:
            ValueError: State type not equal to 'surface'
        """
        if state_type.lower() != "surface":
            raise ValueError(f"`SurfaceState` can only accept `state_type = 'surface'` but `{state_type}` was provided")
        super().__init__(self, Gelec, state_type, name)

    def get_free_energy(self) -> float:
        return self.Gelec

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def Gfree(self) -> float:
        return self._Gelec

    @property
    def state_type(self) -> str:
        return self.surface
    

#TODO
class AdsorbedState(BaseState):
    def __init__(self,
                Gelec: float,
                state_type: Literal["adsorbate", "TS"],
                name:str,
                freq: Sequence[float],
                ifreq: Sequence[float] | None = None,
                truncate_freq: bool = True):
        """_summary_

        Args:
            Gelec (float): _description_
            state_type (Literal[&quot;adsorbate&quot;, &quot;TS&quot;]): _description_
            name (str): _description_
            freq (Sequence[float]): _description_
            ifreq (Sequence[float] | None, optional): _description_. Defaults to None.
            truncate_freq (bool, optional): _description_. Defaults to True.

        Raises:
            AttributeError: _description_

        Returns:
            _type_: _description_
        """


class State:
    def __init__(
            self, 
            state_type: str,
            Gelec: float,
            truncate_freq: bool = True,
            freq: Sequence[float] | None = None,
            i_freq: Sequence[float] | None = None,
            mass: float | None = None,
            inertia: Sequence[float] | None = None,
            sigma: float | None = None
            ):
        """Foundational class to define reaction states"""

        if isinstance(settings, dict):
            ta = TypeAdapter(StateSettings)
            settings = ta.validate_python(settings)

        # Unpack all fields from the Pydantic model directly onto the class instance
        self.__dict__.update(settings.model_dump())

        # Gfree Initializer
        self._Gfree = None


    @property
    def Gfree(self):
        if self._Gfree is None:
            raise AttributeError("Gfree has not been computed yet and it's being accessed. Use `State.get_free_energy(T,P)` first")
        return self._Gfree


    def get_free_energy(self, T:float, p:float) -> float:
        """Computes the free energy and returns it in [eV]

        Args:
            T (float): Temperature in [K]
            p (float): Pressure in [Pa]

        Returns:
            float: Gibbs free energy of the State
        """
        Gfree = self.Gelec

        # For instances with frequencies, compute the vibrational contribution
        if getattr(self, "freq", None) is not None:
            Gfree += self._calc_vibrational_contrib(T=T)

        # For gas-phase instances, compute translational and rotational modes as well
        if self.state_type == "gas":
            Gfree += self._calc_translational_contrib(T=T, p=p)
            Gfree += self._calc_rotational_contrib(T=T)

        # self._Gfree = Gfree
        return Gfree


    def _truncate(self):
        """Truncates some vibrational modes if required"""
        if self.state_type == 'gas':
            ntrunc = self.shape
        elif self.state_type == 'TS' and len(self.i_freq) == 0:
            ntrunc = 1
        else:
            ntrunc = 0
        nfreqs = self.freq.shape[0] - ntrunc
        return copy.deepcopy(self.freq[0:nfreqs])


    def _calc_zpe(self):
        """Compute the zero-point energy contribution (units consistent with `self.get_free_energy`)"""
        use_freq = self._truncate()
        return 0.5 * h * sum(use_freq) * JtoeV


    def _calc_vibrational_contrib(self, T):
        """Compute the vibrational energy contribution to the free energy (units consistent with `self.get_free_energy`)"""
        Gzpe = self._calc_zpe()
        use_freq = self._truncate()

        if sum(use_freq) != 0.0:
            return Gzpe + (kB * T * sum(np.log(1 - np.exp(-use_freq * h / (kB * T))))) * JtoeV
        else:
            return Gzpe


    def _calc_translational_contrib(self, T, p):
        """Calculates translational contribution to free energy (units consistent with `self.get_free_energy`)"""
        return (-kB * T * np.log((kB * T / p) * pow(2 * np.pi * (self.mass * amutokg) * kB * T / (h ** 2), 1.5))) * JtoeV


    def _calc_rotational_contrib(self, T):
        """Calculates rotational contribution to free energy (units consistent with `self.get_free_energy`)
        Accounts for linear and non-linear molecules"""
        I = self.inertia * amuA2tokgm2

        if self.shape == 2:
            I = np.sqrt(np.prod([I[i] for i in range(len(I)) if I[i] != 0]))
            return (-kB * T * np.log(8 * np.pi * np.pi * kB * T * I / (self.sigma * h ** 2))) * JtoeV
        else:
            return (-kB * T * np.log((np.sqrt(np.pi) / self.sigma) * pow(8 * np.pi * np.pi * kB * T / (h ** 2), 1.5) * np.sqrt(np.prod(I)))) * JtoeV






























#     state_type: str = Field(description="Identified for the state")
#     Gelec: float = Field(description="Electronic energy computed by DFT")

# vib
#     truncate_freq: bool = Field(default=True, description="Flag to remove vibrational frequencies below 12.4 meV")
#     freq: Sequence[float] = Field(description="Vibrational frequencies")

# gas
#     mass: float = Field(description="Molecular Mass of gas-phase species", default=None)
#     inertia: Sequence[float] = Field(description="Moment of inertia of gas-phase species", default=None)
#     sigma: float = Field(description="Rotational symmetry number")

# transition_state
#     i_freq: float | Sequence[float] = Field(description="Imaginary vibrational frequency")




# (bare surfaces, adsorbed species, transition states, and gas-phase molecules)


# TODO: Need to edit to receive the reaction somehow... think about it
class ScalingState(State):
    def __init__(self, settings: ScalingStateSettings | dict):
        """Computes energies based on reference reactions (e.g., adsorption energies).
        The reference reactions must be initialized first"""
        if isinstance(settings, dict):
            ta = TypeAdapter(ScalingStateSettings)
            settings = ta.validate_python(settings)

        self.__dict__.update(settings.model_dump())

        # Gelec is NOT computed here. It will be computed on demand.
        self.Gelec = None
        self._reactions_bound = False

    def bind_reactions(self, reactions: dict):
        """Resolve reaction-name strings to Reaction objects. Call once after all reactions exist."""
        if self._reactions_bound:
            return
        for r in self.scaling_reactions.values():
            if isinstance(r['reaction'], str):
                r['reaction'] = reactions[r['reaction']]
            if 'multiplicity' not in r:
                r['multiplicity'] = 1.0
        self._reactions_bound = True

    def _calc_electronic_energy(self):
        """Compute electronic energy fresh each call (reads current sC/sO Gelec)."""
        Gelec = self.scaling_coeffs['intercept']
        for idx, r in enumerate(self.scaling_reactions.values()):
            rxn = r['reaction']
            dEIS = rxn.get_reaction_energy(T=273, p=1.0e5, etype='electronic') / (eVtokJ * 1.0e3)
            if self.dereference:
                ref_EIS = sum([reac.Gelec for reac in rxn.reactants])
            else:
                ref_EIS = 0.0
            Gelec += r['multiplicity'] * (self.scaling_coeffs['gradient'][idx] * dEIS + ref_EIS)
        return Gelec

    def get_free_energy(self, T, p, verbose=False):
        if self.use_descriptor_as_reactant:
            assert self.scaling_reactions is not None
            assert self.scaling_coeffs is not None

            Gelec = self.scaling_coeffs['intercept']
            Gfree = 0.0
            for idx, r in enumerate(self.scaling_reactions.values()):
                rxn = r['reaction']
                dEIS = rxn.get_reaction_energy(T=T, p=p, etype='electronic') / (eVtokJ * 1.0e3)
                dGIS = rxn.get_reaction_energy(T=T, p=p, etype='free') / (eVtokJ * 1.0e3)
                if self.dereference:
                    ref_EIS = sum([reac.Gelec for reac in rxn.reactants])
                    ref_GIS = sum([reac.get_free_energy(T=T, p=p) for reac in rxn.reactants])
                else:
                    ref_EIS = 0.0
                    ref_GIS = 0.0
                Gelec += r['multiplicity'] * (self.scaling_coeffs['gradient'][idx] * dEIS + ref_EIS)
                Gfree += r['multiplicity'] * (-ref_EIS - dEIS + dGIS + ref_GIS)
            Gfree += Gelec
            return Gfree
        else:
            return super(ScalingState, self).get_free_energy(T=T, p=p)


class ScalingState(State):
    def __init__(self, settings: ScalingStateSettings | dict):
        """Alternative State Class that leverages linear scaling relationships to compute energies
        It leverages Pydantic Schemas to ensure input consistency and a cleaner assignment of attributes. Check the schema for info on the fields"""

        if isinstance(settings, dict):
            ta = TypeAdapter(ScalingStateSettings)
            settings = ta.validate_python(settings)

        # Unpack all fields from the Pydantic model directly onto the class instance
        self.__dict__.update(settings.model_dump())

        # Compute the electronic energy
        self.Gelec = self._calc_electronic_energy()


    def _calc_electronic_energy(self):
        """Modified version of calc_electronic_energy method.

        This patch allows the user to take multiple coefficients in list format.
        IT ASSUMES that the ordering is consistent (i.e., C, O)

        Saves value in eV.
        """
        Gelec = self.scaling_coeffs['intercept']

        for idx, r in enumerate(self.scaling_reactions.values()):
            dEIS = r['reaction'].get_reaction_energy(T=273, p=1.0e5, etype='electronic') / (eVtokJ * 1.0e3)
            if self.dereference:
                ref_EIS = sum([reac.Gelec for reac in r['reaction'].reactants])
            else:
                ref_EIS = 0.0
            if 'multiplicity' not in r:
                r['multiplicity'] = 1.0
            Gelec += r['multiplicity'] * (self.scaling_coeffs['gradient'][idx] * dEIS + ref_EIS)

        return Gelec


    def get_free_energy(self, T, p):
        """This patch allows the user to take multiple coefficients in list format.
        IT ASSUMES that the ordering is consistent (i.e., C, O)

        Saves value in eV.

        PATCHES: pycatkin.classes.state.ScalingState.calc_free_energy
        """
        if self.use_descriptor_as_reactant:

            assert(self.scaling_reactions is not None)
            assert(self.scaling_coeffs is not None)

            Gelec= self.scaling_coeffs['intercept']
            Gfree = 0.0

            for idx, r in enumerate(self.scaling_reactions.values()):
                dEIS = r['reaction'].get_reaction_energy(T=T, p=p, etype='electronic') / (eVtokJ * 1.0e3)
                dGIS = r['reaction'].get_reaction_energy(T=T, p=p, etype='free') / (eVtokJ * 1.0e3)
                if self.dereference:
                    ref_EIS = sum([reac.Gelec
                                    for reac in r['reaction'].reactants])
                    ref_GIS = sum([reac.get_free_energy(T=T, p=p)
                                    for reac in r['reaction'].reactants])
                else:
                    ref_EIS = 0.0
                    ref_GIS = 0.0
                if 'multiplicity' not in r.keys():
                    r['multiplicity'] = 1.0
                Gelec += r['multiplicity'] * (self.scaling_coeffs['gradient'][idx] * dEIS + ref_EIS)
                Gfree += r['multiplicity'] * (-ref_EIS - dEIS + dGIS + ref_GIS)
            Gfree += Gelec
            return Gfree

        else:
            return super(ScalingState, self).get_free_energy(T=T, p=p)