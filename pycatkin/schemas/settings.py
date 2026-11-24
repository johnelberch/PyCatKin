"""Pydantic schemas for the main classes. This validates input files and formats the inputs
"""

from collections.abc import Sequence
from typing import Annotated, Any, Literal, Union

import numpy as np
from numpy.typing import NDArray
from pycatkin.constants.physical_constants import JtoeV, h
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


########################################
# State Class
########################################
# Base classes
class BaseStateSettings(BaseModel):
    """Foundational class state. Contains the basic features other classes should inherit"""
    state_type: str = Field(description="Identified for the state")
    Gelec: float = Field(description="Electronic energy computed by DFT")


class FrequencyMixin(BaseModel):
    """Adds frequency tracking and automatic sorting/truncation"""
    truncate_freq: bool = Field(default=True, description="Flag to remove vibrational frequencies below 12.4 meV")
    freq: Sequence[float] = Field(description="Vibrational frequencies")

    @field_validator("freq", mode="after")
    @classmethod
    def sort_frequencies(cls, v: Sequence[float]) -> NDArray[float]:
        return np.array(sorted(v, reverse=True))

    @model_validator(mode="after")
    def conditional_frequency_truncation(self) -> "FrequencyMixin":
        if self.truncate_freq:
            self.freq = [f for f in self.freq]
            cutoff = 12.4 * 1e-3 / (h * JtoeV)
            a = self.freq.copy()
            self.freq = np.where(a >= cutoff, a, cutoff)
        return self


class GasMixin(BaseModel):
    """Adds gas-phase specific properties"""
    mass: float = Field(description="Molecular Mass of gas-phase species", default=None)
    inertia: Sequence[float] = Field(description="Moment of inertia of gas-phase species", default=None)
    sigma: float = Field(description="Rotational symmetry number")

    @field_validator("inertia", mode="after")
    @classmethod
    def clean_inertia(cls, v: Sequence[float]) -> NDArray:
        inertia_cutoff = 1.0e-12
        return np.array([i if i > inertia_cutoff else 0.0 for i in v])

    @computed_field
    @property
    def shape(self) -> int:
        shape = len([i for i in self.inertia if i > 0.0])
        if shape < 2:
            raise ValueError("Too many components of the moments of inertia are zero.") 
        return shape


class TransitionStateMixin(BaseModel):
    """Adds imaginary frequency for transition states"""
    i_freq: float | Sequence[float] = Field(description="Imaginary vibrational frequency")

    @field_validator("i_freq", mode="after")
    @classmethod
    def sort_frequencies(cls, v: Sequence[float]) -> NDArray:
        return np.array(sorted(v, reverse=True))


# Different types of fully-defined states (stand-alone)
class SurfaceStateSettings(BaseStateSettings):
    """Bare surfaces"""
    state_type: Literal["surface"] = "surface"


class AdsorbedStateSettings(BaseStateSettings, FrequencyMixin):
    """Adsorbed species on surfaces"""
    state_type: Literal["adsorbate"] = "adsorbate"


class GasStateSettings(BaseStateSettings, FrequencyMixin, GasMixin):
    """Gas-phase molecules"""
    state_type: Literal["gas"] = "gas"


class TransitionStateSettings(BaseStateSettings, FrequencyMixin, TransitionStateMixin):
    """adsorbed transition states"""
    state_type: Literal["TS"] = "TS"


StateSettings = Annotated[
    SurfaceStateSettings | AdsorbedStateSettings | GasStateSettings | TransitionStateSettings,
    Field(discriminator="state_type"),
]


# States defiend with scaling relationships
class DescriptorStateSettings(AdsorbedStateSettings):
    """Adsorbed states used as energy decriptors"""
    Gelec: float | None = None


class ScalingRelationship(BaseModel):
    """Describes the linear relationship with a given reaction"""
    reaction: str = Field(description="The name of the reaction used as descriptor. Must match an enumerated reaction")
    slope: list[float] = Field(description="Model slopes for each descriptor")
    intercept: float = Field(description="Model intercept")


class ScalingMixin(BaseModel):
    """Adds scaling relationship properties and makes Gelec optional"""
    scaling_reactions: list[ScalingRelationship] 
    Gelec: float | None = None  # Overrides base Gelec to allow None


class ScalingTSStateSettings(TransitionStateSettings, ScalingMixin):
    """Transition state that uses a scaling relationship to compute its energy"""
    pass

class ScalingAdsorbedStateSettings(AdsorbedStateSettings, ScalingMixin):
    """Adsorbed state that uses a scaling relationship to compute its energy"""
    pass


ScalingStateSettings = Annotated[
    ScalingTSStateSettings | ScalingAdsorbedStateSettings,
    Field(discriminator="state_type"),
]


# Combined state
class State(BaseModel):
    name: StateSettings | ScalingStateSettings


########################################
# Reaction Class
########################################


class ReactionSettings(BaseModel):
    """Settings for :class:`pycatkin.classes.reaction.Reaction`.

    Captures the constructor arguments of ``Reaction``. Subclass-specific
    fields (e.g. ``dErxn_user`` for ``UserDefinedReaction``,
    ``base_reaction`` for ``ReactionDerivedReaction``) are accepted via
    ``extra="allow"``.
    """

    model_config = ConfigDict(extra="allow")

    name: str = "reaction"
    reac_type: Any = None
    reversible: bool = True
    reactants: Any = None
    products: Any = None
    TS: Any = None
    area: float = 1.0e-19
    scaling: float = 1.0


class ReactorSettings(BaseModel):
    """Settings for :class:`pycatkin.classes.reactor.Reactor`.

    Captures the constructor arguments of ``Reactor``. Subclass-specific
    fields are accepted via ``extra="allow"``.
    """

    model_config = ConfigDict(extra="allow")

    name: str = "reactor"
    volume: Any = None
    catalyst_area: Any = None
    residence_time: Any = None
    flow_rate: Any = None


class SystemSettings(BaseModel):
    """Settings for :class:`pycatkin.classes.system.System`.

    Captures the constructor arguments of ``System``.
    """

    model_config = ConfigDict(extra="allow")

    times: Any = None
    start_state: dict = {}
    inflow_state: dict = {}
    T: float = 293.15
    p: float = 101325.0
    use_jacobian: bool = True
    ode_solver: str = "solve_ivp"
    nsteps: float = 1e4
    rtol: float = 1e-8
    atol: float = 1e-10
    xtol: float = 1e-8
    ftol: float = 1e-8
    verbose: bool = False
    y0: Any = None
    min_tol: float = 1e-32


class SolverSettings(BaseModel):
    """Settings for :class:`pycatkin.classes.solver.SteadyStateSolver`.

    Captures the constructor arguments of ``SteadyStateSolver`` except for
    ``system``, which is the object being solved rather than a setting.
    """

    model_config = ConfigDict(extra="allow")

    ss_guess: Any = None
    verbose: bool = False
