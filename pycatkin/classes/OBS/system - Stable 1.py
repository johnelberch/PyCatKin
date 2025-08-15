"""Patched version of system.py
This version includes changes for the calculation of steady states
"""
import copy
from functools import lru_cache

import numpy as np
from scipy.integrate import ode, solve_ivp
from scipy.optimize import fsolve, least_squares, minimize, root
from typing import NamedTuple

from pycatkin.classes.energy import Energy
from pycatkin.classes.reaction import Reaction
from pycatkin.classes.reactor import InfiniteDilutionReactor, Reactor
from pycatkin.classes.state import State
from pycatkin.constants.physical_constants import *


# Auxiliary named tuple for results
class SteadyStateResults(NamedTuple):
    """Auxiliary named tuple to store coverage and success of convergence
    
    Args:
        x (np.ndarray): Steady-state coverage (includes surface and gas species, in a similar fashion to how they are
            defined inside of System.initial_system)
        success (bool): Whether the calculation turned successful or not 

    """
    x: np.ndarray
    success: bool


class System:
    """Centralized worker class. Holds information for states, reactions and reactors. 
    Has methods to solve the system compositions at steady state, compute the reaction rates,
    define species ODEs.
    """
    def __init__(
        self, 
        times: float = None, 
        start_state: dict = {}, 
        inflow_state: dict = {}, 
        T: float = 293.15, 
        p: float = 101325.0,              
        use_jacobian:bool = True, 
        ode_solver:str = 'solve_ivp', 
        nsteps: float = 1e4, 
        rtol: float = 1e-8, 
        atol: float = 1e-10,
        xtol: float = 1e-8, 
        ftol: float = 1e-8, 
        verbose: bool = False,
        y0 = None
        ):

        # Parameters for ODE
        self.ode_params = {
            "times": copy.deepcopy(times),
            "rtol": rtol,
            "atol": atol,
            "xtol": xtol,
            "ftol": ftol,
            "jacobian": use_jacobian,
            "nsteps": int(nsteps),
            "ode_solver": ode_solver
        }

        # System parameter assignments
        self.start_state = copy.deepcopy(start_state)
        self.inflow_state = copy.deepcopy(inflow_state)
        self.T = T
        self.p = p
        self.verbose = verbose

        # Blank assignments (to be populated when reading from input file)
        self.states = dict()
        self.unique_states = set()
        self.reactions = dict()
        self.reactor = None
        self.energy_landscapes = dict
        
        self.gas_indices = None
        self.dynamic_indices = None
        self.rate_constants = None
        self.rates = None
        self.times = None
        self.solution = None
        self.full_steady = None


    #----- 
    # Add functions (for System attributes)
    def add_state(self, state: "pycatkin.classes.states.State"):
        """Adds a State to the dictionary of states. Checks that state names are unique.

        Args:
            state (pycatkin.classes.states.State): instance (or subclass) of State
        
        Sets:
            self.states (dict): Dictionary of states, with (key,value) pairs corresponding to (state_name, State object)
            self.unique_states (set): Set of unique state names
        """
        # Check that the state is an appropriate instance and logging message
        assert isinstance(state, State), f"state {state} MUST be an instance of PyCatKin State"
        if self.verbose:
            print(f"Adding state {state.name}")

        # If state is repeated, raise a ValueError, else, add to the unique unique_states set
        if state.name in self.unique_states:
            raise ValueError('Found two copies of state %s. State names must be unique!' % state.name)
        else:
            self.unique_states.add(state.name)

        # Add state entry to the dictionary
        self.states[state.name] = state


    def add_reaction(self, reaction: "pycatkin.classes.reaction.Reaction"):
        """Adds a reaction to the dictionary of reactions.

        Args:
            state (pycatkin.classes.reaction.Reaction): instance (or subclass) of Reaction
        
        Sets:
            self.reactions (dict): Dictionary of reactions, with (key,value) pairs corresponding to (rxn_name, Reaction object)
        """
        # Check that the reaction is an appropriate instance and logging message
        assert isinstance(reaction, Reaction), f"reaction {reaction} MUST be an instance of PyCatKin Reaction"
        if self.verbose:
            print('Adding reaction %s.' % reaction.name)
        
        # Add the reaction to the corresponding dictionary
        self.reactions[reaction.name] = reaction


    def add_reactor(self, reactor: "pycatkin.classes.reactor.Reactor"):
        """Adds a reactor

        Args:
            state (pycatkin.classes.reactor.Reactor): instance (or subclass) of Reactor
        
        Sets:
            self.reactor (Reactor)
        """
        # Check that the reactor is an appropriate instance and logging message
        assert isinstance(reactor, Reactor), f"reaction {reactor} MUST be an instance of PyCatKin Reactor"
        if self.verbose:
            print('Adding reactor')

        self.reactor = reactor


    def add_energy_landscape(self, energy_landscape: "pycatkin.classes.energy.Energy"):
        """Adds an energy landscape to the dictionary of reactions.

        Args:
            state (pycatkin.classes.energy.Energy): instance (or subclass) of Energy
        
        Sets:
            self.energy_landscapes (dict): key,value pairs of Energy.name, Energy
        """
        # Check that the reactor is an appropriate instance and logging message
        assert isinstance(energy_landscape, Energy), f"reaction {energy_landscape} MUST be an instance of PyCatKin Reactor"
        if self.verbose:
            print(f'Adding energy landscape {energy_landscape.name}')

        self.energy_landscapes[energy_landscape.name] = energy_landscape

    def build(self):
        """Once all the states have been added, it sets the System instance attributes

        Sets:
            self.coverage_map (dict): Dictionary of surfaces and indices. Added to keep track of the coverage conservation.
                {surface_name: Set[indices]}
            self.gas_indices (set): Gas species indexes. Used to normalize fractional compositions and get partial pressures.
            self.index_map (dict): Dictionary of species names - mapping indexes for rapid indexing of composition arrays.
                {species_name: index}
            self.rate_map (dict): {reaction_name :{ "reac": [reactant indexes], "prod": [product indexes], "site_density": float, "scaling": Reaction.scaling}}
            self.reactor.set_indices()...
            self.initial_system (np.array): Array with gas fractional concentrations, and surface coverages
            self.reactions["rxn_name"].kfwd and .krev attributes
        """
        self._names_to_indices()
        self._mapping_reaction_indices()
        self._get_initial_conditions()
        self._update_rate_constants()


    #-----
    # Utility functions
    def _names_to_indices(self):
        """For speed purposes (i.e., indexing large arrays), the State objects will be mapped into a numerical
        representation (i.e., unique indices will be given for each, which will correspond to the indexes to
        use in the array objects).

        NOTE: Only indices corresponding to long-lived species (intermediates, gas-phase species, adsorbates)
        will be stored. In other words, TS won't be stored, as these are short-lived.

        There could be more efficient approaches with more extensive API changes, but it is not the aim of this patch.

        Sets:
            self.coverage_map (dict): Dictionary of surfaces and indices. Added to keep track of the coverage conservation.
                {surface_name: List[indices]}
            self.gas_indices (set): Gas species indexes. Used to normalize fractional compositions and get partial pressures.
            self.index_map (dict): Dictionary of species names - mapping indexes for rapid indexing of composition arrays.
                {species_name: index}
        """
        # Classify into three groups (relevant to keep track of coverages and conservation of sites)
        adsorbates = []
        gas = []
        surfaces = []
        for name, state in self.states.items():
            if state.state_type == "adsorbate":
                adsorbates.append(name)
            elif state.state_type == "gas":
                gas.append(name)
            elif state.state_type == "surface":
                surfaces.append(name)

        # sort gas and unique surfaces
        gas = sorted(gas)
        surfaces = sorted(surfaces)

        # Assemble a coverage id dictionary (unique surface coverages must add to 1)
        self.coverage_map = dict()
        # Assemble a gas_id set
        self.gas_indices = set()
        # Get a dictionary to map the ids
        self.index_map = dict()
        # Index variable
        count = 0

        for g in gas:
            self.index_map[g] = count # gas_name : id
            self.gas_indices.add(count)
            count +=1
        for surf in surfaces:
            self.coverage_map[surf] = {count} # Creates a set of indexes for things bound to surf (adding empty surf)
            self.index_map[surf] = count # surf_name : id
            count += 1
            for ads in adsorbates:
                if ads[0] == surf:
                    self.coverage_map[surf].add(count) # Append ads to the coverage set of surface
                    self.index_map[ads] = count # ads_name : id
                    count += 1
        
        assert sum([len(value) for value in self.coverage_map.values()]) == len(adsorbates) + len(surfaces), "There is a mismatch between adsorbates and covered sites. Check"


    def _mapping_reaction_indices(self):
        """Gets the indexes changed for each individual reaction in the network (i.e., the specific states)

        Sets:
            self.rate_map (dict): {reaction_name :{ "reac": [reactant indexes], "prod": [product indexes], "site_density": float, "scaling": Reaction.scaling}}
            self.reactor.set_indices()...
        """
        # Mapping dictionary of the indexes changed by the reaction rates
        self.rate_map = dict()
        for name, reaction in self.reactions.items():
            if str(reaction.reac_type).upper() == "GHOST":
                continue
            self.rate_map[name] = {
                "reac" : [self.index_map[n.name] for n in reaction.reactants],
                "prod" : [self.index_map[n.name] for n in reaction.products],
                'site_density': 1.0 / reaction.area if reaction.area else 0.0,
                'scaling': reaction.scaling,
            }

        # Compliance with legacy API
        is_gas = np.zeros(len(self.unique_states), dtype=int)
        is_gas[list(self.gas_indices)] = 1
        is_gas = is_gas.tolist()

        is_adsorbate = np.zeros(len(self.unique_states), dtype=int)
        for indices in self.coverage_map.values():
            is_adsorbate[list(indices)] = 1
        is_adsorbate = is_adsorbate.tolist()

        self.reactor.set_indices(is_adsorbate=is_adsorbate, is_gas=is_gas)


    def _get_initial_conditions(self):
        """Reads the initial conditions from input file and gets the coverage/gas concentration profile
        of the system at the very beginning of the simulation.

        Sets:
            self.initial_system (np.array): Array with gas fractional concentrations, and surface coverages
        """

        # Array that will store the concentration profile
        y = np.zeros(len(self.index_map.keys()))

        # Populate the profile concentrations
        for name, initial_condition in self.start_state.items():
            y[self.index_map[name]] = initial_condition
        for name, initial_condition in self.inflow_state.items():
            y[self.index_map[name]] = initial_condition

        # Normalize inputs
        y = self._normalize_y(y)

        # Assign y to a variable named initial_system
        self.initial_system = y

    def _normalize_y(self, y: np.array) -> np.array: 
        """Normalizes the concentrations of surface and gas products.
        Surface products MUST ADD to one (site conservation)
        Gas products MUST ADD to one (fractional concentration)

        Args:
            y (np.array): concentration array. Fractional concentration for gas species. Surface
                coverage for surface species. Different sites are treated separatedly (a reservoir
                can have a net coverage of 1, and the main site as well). Order matches self.index_map

        Returns:
            np.array: normalized array
        """
        # Normalize gas concentrations to get partial pressures when needed
        y[list(self.gas_indices)] /= np.sum(y[list(self.gas_indices)])

        # Normalize surface coverages (must add to 1 either way)
        for surf_indices in self.coverage_map.values():
            y[list(surf_indices)] /= np.sum(y[list(surf_indices)])

        return y

    #-----
    # Computing rates
    @lru_cache(maxsize=1)
    def _update_rate_constants(self):
        """Update rate constants for current conditions.

        Rate constants are stored in self.reactions["rxn_name"].kfwd and .krev attributes
        """
        for rxn in self.reactions.values():
            rxn.calc_rate_constants(
                T = self.T,
                p = self.p,
                verbose = self.verbose
            )

    def _calc_rates(self, y) -> np.ndarray:
        """Computes the forward and backward rates for the given reaction network

        Args:
            y (np.array): concentration array. Fractional concentration for gas species. Surface
                coverage for surface species. Different sites are treated separatedly (a reservoir
                can have a net coverage of 1, and the main site as well). Order matches self.index_map

        Sets:
            self.rates (np.array) of shape (n_reactions, 2), with first column for forward rate, second column for backward rate
                Units of 1/s
        """
        self._update_rate_constants()
        rates = []

        for name in self.rate_map.keys():
            rxn = self.reactions[name]

            # If state is gas, include partial pressure into list (y[i] is frac concentration and P is total pressure in Pa)
            # If state is surface or adsorbate, include coverge into list (THIS WILL INCLUDE THE EMPTY SURFACES)
            reactants = [y[i]*self.p if i in self.gas_indices else y[i] for i in self.rate_map[name]["reac"]] 
            products = [y[i]*self.p if i in self.gas_indices else y[i] for i in self.rate_map[name]["prod"]]

            # Multiply the concentrations by the rate constants
            rates.append(
                [
                    rxn.kfwd * np.prod(reactants),
                    rxn.krev * np.prod(products)
                ]
            )

        return np.array(rates)

    def get_dydt(self, y) -> np.ndarray:
        """Gets the right-hand side of the ODE system for each individual species
        Basically, it computes the net generation/consumption rates for each individual species.
        The rates are stored in a numpy array. The order of the array matches the state indexes defined 
        in the index_map

        Args:
            y (np.array): concentration array. Fractional concentration for gas species. Surface
                coverage for surface species. Different sites are treated separatedly (a reservoir
                can have a net coverage of 1, and the main site as well). Order matches self.index_map

        Returns:
            dydt (np.array): 1D array of size (num_tracking_elements) with net generation / consumption
                rates for all the states being tracked
        """
        # Solution array
        dydt = np.zeros(len(self.initial_system))      

        # Compute rates
        rates = self._calc_rates(y)

        # Get rate per species in network
        for idx, sub_dict in enumerate(self.rate_map.values()):
            net_rate = rates[idx,0] - rates[idx,1] #("Forward rate - Backward rate")
            for k in sub_dict["reac"]:
                dydt[k] -= net_rate
            for k in sub_dict["prod"]:
                dydt[k] += net_rate

        return dydt

    def get_forward_only(self, y) -> np.ndarray:
        """Computes the forward reaction rate only

        Args:
            y (np.array): concentration array. Fractional concentration for gas species. Surface
                coverage for surface species. Different sites are treated separatedly (a reservoir
                can have a net coverage of 1, and the main site as well). Order matches self.index_map

        returns
            np.ndarray: Forward rate only in 1/s
        """
        # Solution array
        dydt = np.zeros(len(self.initial_system))      

        # Compute rates
        rates = self._calc_rates(y)

        # Get rate per species in network
        for idx, sub_dict in enumerate(self.rate_map.values()):
            frate = rates[idx,0] #forward only
            for k in sub_dict["reac"]:
                dydt[k] -= frate
            for k in sub_dict["prod"]:
                dydt[k] += frate

        return dydt

    def _fun_ss(self, y_surf: np.array) -> np.ndarray:
        if not isinstance(self.reactor, InfiniteDilutionReactor):
            raise AttributeError("reactor must be of type pycatkin.classes.reactor.InfiniteDilutionReactor")
        
        # Invariant compositions (flow gas)
        y_gas = self.initial_system[list(self.gas_indices)]

        # Total compositions to compute rates
        y = np.concat([y_gas, y_surf])

        # Get rates for each species
        dydt = self.get_dydt(y)

        # Return the SURFACE ONLY dydt
        return dydt[len(y_gas):]
        

    def find_steady(self, max_iters: int = 30, y0: np.ndarray = None, method="lm") -> SteadyStateResults:
        """Finds the a steady-state solution.

        Args:
            max_iters (int, optional): How many times to try to refine the solution. Solution refinements are done so 
                the site conservation laws (and gas concentration profiles if required) are within their limits with 
                less than 5% error
            y0 (np.ndarray, optional): Initial guess, defaults to randomized guess from an uniform distribution

        Returns:
            SteadyStateResults: Named tuple with attributes x (1D numpy array of concentrations for all species) and
                success (boolean indicating if the calculation converged)

        NOTE: I wasn't able to find a way to FORCE the site conservation. Nonetheless, I have found that, if
        you resubmit a failed calculation, you often find the correct root!
        """
        # Randomized initial guess (take surface only)
        if y0 is None:
            y0 = self._normalize_y(np.random.uniform(size=len(self.initial_system)))
        elif len(y0) != len(self.initial_system):
            raise ValueError("Initial guess must have same length as initial guess... Include gas and surface species in here!")
        
        y0 = y0[len(self.gas_indices):]

        # Preliminars
        idx = 0 #N iterations
        factor = 1 #Tightness factor (multiplies tol)
        success = False # Will be True if calculation converges

        while idx < max_iters:
            sol = root(
                fun = self._fun_ss,
                x0 = y0,
                method = method,
                jac = False,
                tol=1e-6*factor
            )

            y0 = sol.x
            y = np.concat((self.initial_system[list(self.gas_indices)], y0))

            # Tracking the surface coverages
            surf_sum = [sum(y[list(surf_indices)]) for surf_indices in self.coverage_map.values()]
            if self.verbose:
                print(f"iter {idx:3d}:  {' , '.join(str(x)[:8] for x in surf_sum)}", end="\r")

            # Check if calculation converged
            if np.any(np.abs(np.array(surf_sum) - 1) > 0.05) or any(np.array(y0) < 0) or np.any(np.abs(self.get_dydt(y)) > 1e-6):
                y0 = self._normalize_y(np.abs(y))[len(self.gas_indices):]
                factor = factor/10**(1/4) if factor > 1e-8 else factor #Tighter tol if needed
                idx += 1
            else:
                success = True
                break

        y = np.concat((self.initial_system[list(self.gas_indices)], sol.x))
        return SteadyStateResults(y, success) 