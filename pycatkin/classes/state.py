from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
import numpy as np
from typing import Literal, TYPE_CHECKING
from ase.thermochemistry import HarmonicThermo, IdealGasThermo

import copy

import numpy as np
from pycatkin.util.constants import h, from_J

if TYPE_CHECKING:
    from ase import Atoms
# from pycatkin.settings import ScalingStateSettings, StateSettings
# from pydantic import TypeAdapter
    

def _preprocess_frequencies(freq: Sequence[float], truncate_freq:bool = True) -> Sequence[float]:
    """Helping function to preprocess frequencies in frequency-relevant classes. 
    Frequency Cutoff set to 0.0124 eV [100 cm^-1]

    Args:
        freq (Sequence[float]): Vibrational Frequencies in [1/s]
        truncate_freq (bool, optional): Trim the frequencies below cutoff. Defaults to True.

    Returns:
        Sequence[float]: processed frequencies. NOTE: For compatibility with ASE, the frequencies are converted to eV
    """
    cutoff = 0.0124 #eV
    sorted_freq = np.array(sorted(freq, reverse=True)) * from_J["eV"](h) #eV
    if truncate_freq:
        a = sorted_freq.copy()
        sorted_freq = np.where(a >= cutoff, a, cutoff)
    return sorted_freq


class BaseState(ABC):
    def __init__(self, name: str):
        """Basic state definition (defines the bare-bones contract all State classes must follow for Reaction classes to work)

        Args:
            name (str): Unique tag for this individual state
        """
        self._name = name
        self._Uelec = None

    @property
    @abstractmethod
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def Uelec(self) -> float:
        """Electronic Energy (e.g., from DFT or Scaling Relationships)"""
        return self._Uelec

    @abstractmethod
    def is_transition_state(self) -> bool:
        """Use this to help down the road with the Reaction class (checks if a State is a transition state)"""

    @abstractmethod
    def get_Gfree(self, T: float, P: float) -> bool:
        """Computes the Gibbs Free Energy"""
    

class SurfaceState(BaseState):
    def __init__(self, name: str, Uelec: float):
        """Bare surfaces (no adsorbates, used commonly as a reference state).
        Assumes that vibrational contributions are null.

        Args:
            name (str): Unique tag for this individual state
            Uelec (float): Electronic Energy, computed from DFT in eV
        """
        super().__init__(name)
        self._Uelec = Uelec

    def is_transition_state(self) -> bool:
        """Bare surfaces are always initial or final states"""
        return False

    def get_Gfree(self, *args, **kwargs) -> bool:
        """Computes the Gibbs Free Energy. In this case, the Gibbs Free energy is null
        NOTE: Unusued *args, **kwargs are left in place to ensure compatibility with methods that might parse unnecessary args (e.g., P)"""
        return self._Uelec


class AdsorbedState(BaseState):
    def __init__(self,
                 name: str,
                 Uelec: float,
                 freq: Sequence[float],
                 ifreq: Sequence | None = None,
                 truncate_freq: bool = True
                 ):
        """Adsorbed States (adsorbed molecules and transition states)

        Args:
            name (str): Unique tag for this individual state
            Uelec (float): Electronic Energy, computed from DFT in eV
            freq (Sequence[float]): Vibrational frequency modes in [1/s]
            ifreq (Sequence | None, optional): Imaginary vibrational frequency mode [1/s] for transition states. Defaults to None.
            truncate_freq (bool, optional): Flag to trim small vibrational frequencies (values below 100 cm^-1). Defaults to True.
        """
        super().__init__(name)
        self._Uelec = Uelec
        self.freq = _preprocess_frequencies(freq=freq, truncate_freq=truncate_freq)
        self.ifreq = self._check_ifreq(ifreq=ifreq)
        self.thermo = HarmonicThermo(self.freq)
    
    def get_Gfree(self, T:float, *args, **kwargs) -> float:
        """Computes the free energy and returns it in [eV]
        NOTE: Unusued *args, **kwargs are left in place to ensure compatibility with methods that might parse unnecessary args (e.g., P)

        Args:
            T (float): Temperature in [K]

        Returns:
            float: Gibbs free energy of the State
        """
        Gfree = self._Uelec
        Gfree += self.thermo.get_helmholtz_energy(T)
        return Gfree

    def is_transition_state(self):
        """Checks if is a transition state (if imaginary frequencies are parsed, then it is a transition state)"""
        return self.ifreq is not None
    
    @staticmethod
    def _check_ifreq(ifreq: float | complex | Sequence | None):
        """Verifies the integrity of imaginary frequencies (should be one and negative or imaginary)"""
        if ifreq is not None:
            if isinstance(ifreq, Sequence):
                if len(ifreq) != 1:
                    raise ValueError(f"More than one imaginary frequency received. ifreq = {ifreq}")
                ifreq = ifreq[0]

            match ifreq:
                case complex():
                    pass
                case float():
                    if ifreq >= 0:
                        raise ValueError(f"Invalid imaginary frequency received, it should be imaginary or negative, but {ifreq} was parsed")
                case _:
                    raise ValueError(f"Invalid Imaginary frequency parsed. ifreq = {ifreq}")
        return ifreq


class GasState(BaseState):
    def __init__(self,
                name: str,
                Uelec: float,
                freq: Sequence[float],
                geometry: Literal["linear","nonlinear","monoatomic"],
                symmetrynumber: int,
                atoms: Atoms,
                vib_selection: Literal['exact', 'all', 'highest', 'abs_highest'] | None = 'highest',
                spin: float | None = None,
                truncate_freq: bool = True,
                ):
        """Gas-phase States (molecules, monoatomic and diatomic)

        Args:
            name (str): Unique tag for this individual state
            Uelec (float): Electronic Energy, computed from DFT in eV
            freq (Sequence[float]): Vibrational frequency modes in [1/s]
            geometry ('monoatomic','linear',or 'nonlinear'): geometry of the molecule
            symmetrynumber (int): symmetry number of the molecule. See, for example, Table 10.1 and Appendix B of C. Cramer “Essentials of Computational Chemistry”, 2nd Ed.
            atoms (Atoms): Ase Atoms Object
            vib_selection ('exact','all','highest','abs_highest' | None, optional): selection of input vibrational energies considered to be true vibrations (excluding translations and rotations) implied by the geometry and number of atoms. Defaults to 'highest'.
            spin (float | None, optional): _description_. Defaults to None.
            truncate_freq (bool, optional): Flag to trim small vibrational frequencies (values below 100 cm^-1). Defaults to True.
        """
        super().__init__(name)
        self._Uelec = Uelec
        self.freq = _preprocess_frequencies(freq=freq, truncate_freq=truncate_freq) 
        self.thermo = IdealGasThermo(
            vib_energies= self.freq, 
            geometry=geometry,
            atoms=atoms,
            symmetrynumber=symmetrynumber,
            spin=spin,
            vib_selection=vib_selection
            )   

    def get_Gfree(self, T:float, p:float, *args, **kwargs) -> float:
        """Computes the free energy and returns it in [eV]
        NOTE: Unusued *args, **kwargs are left in place to ensure compatibility with methods that might parse unnecessary args (e.g., P)

        Args:
            T (float): Temperature in [K]
            p (float): Pressure in [Pa]

        Returns:
            float: Gibbs free energy of the State
        """
        Gfree = self._Uelec
        Gfree += self.get_gibbs_energy(T=T, p=p)
        return Gfree

    def is_transition_state(self):
        """Checks if is a transition state (gas-phase references are never transition states)"""
        return False


#TODO: Finish this portion
class ScalingState(AdsorbedState):
    def __init__(self,
                 name: str,
                 Uelec: None,
                 freq: Sequence[float],
                 ifreq: Sequence | None = None,
                 truncate_freq: bool = True
                 ):
        """Adsorbed States (adsorbed molecules and transition states)

        Args:
            name (str): Unique tag for this individual state
            Uelec (float): Electronic Energy, computed from DFT in eV
            freq (Sequence[float]): Vibrational frequency modes in [1/s]
            ifreq (Sequence | None, optional): Imaginary vibrational frequency mode [1/s] for transition states. Defaults to None.
            truncate_freq (bool, optional): Flag to trim small vibrational frequencies (values below 100 cm^-1). Defaults to True.
        """
        super().__init__(name)
        self._Uelec = Uelec
        self.freq = _preprocess_frequencies(freq=freq, truncate_freq=truncate_freq)
        self.ifreq = self._check_ifreq(ifreq=ifreq)
        self.thermo = HarmonicThermo(self.freq)