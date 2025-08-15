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
        self.times = None


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

        # self.energy_landscapes[energy_landscape.name] = energy_landscape
        self.energy_landscapes = energy_landscape

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
            self.reaction_matrix (np.array): Reaction matrix mapping which reactions are relevant to each species (check `self._reactant_reaction_matrix()`)
        """
        self._names_to_indices()
        self._mapping_reaction_indices()
        self._get_initial_conditions()
        self._update_rate_constants(self.T, self.p)
        self._reactant_reaction_matrix()


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
    def _update_rate_constants(self, T, p):
        """Update rate constants for current conditions.

        Rate constants are stored in self.reactions["rxn_name"].kfwd and .krev attributes
        """
        for rxn in self.reactions.values():
            rxn.calc_rate_constants(
                T = T,
                p = p,
                verbose = self.verbose
            )

    def _calc_rates(self, y) -> np.ndarray:
        """Computes the forward and backward rates for the given reaction network

        Args:
            y (np.array): concentration array. Fractional concentration for gas species. Surface
                coverage for surface species. Different sites are treated separatedly (a reservoir
                can have a net coverage of 1, and the main site as well). Order matches self.index_map

        Returns:
            np.array: of shape (n_reactions, 2), with first column for forward rate, second column for backward rate
                Units of 1/s
        """
        self._update_rate_constants(self.T, self.p) 
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

    def _reactant_reaction_matrix(self):
        """Creates a Matrix mapping which reactions are relevant to which reactants so we can
        work with matrix multiplication. The matrix has the shape (species, reactions)
        in other words, each row corresponds to a species, and the column elements of that
        row to the reactions that are relevant to it. If the species is a product, the reaction
        value would be 1, if it is a reactant, -1, if the reaction is not relevant, 0

        Sets:
            reaction_matrix (np.ndarray): Array with the information needed
        """
        s_rxn_idx = np.zeros((len(self.initial_system),len(self.rate_map)))

        for idx, rxn_subdict in enumerate(self.rate_map.values()):
            s_rxn_idx[rxn_subdict["reac"],idx] = -1
            s_rxn_idx[rxn_subdict["prod"],idx] = 1

        self.reaction_matrix = s_rxn_idx

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
        # Compute rates
        rates = self._calc_rates(y)
        net_rates = rates[:,0] - rates[:,1]

        # Solution array
        return self.reaction_matrix @ net_rates

    def get_forward_only(self, y) -> np.ndarray:
        """Computes the forward reaction rate only

        Args:
            y (np.array): concentration array. Fractional concentration for gas species. Surface
                coverage for surface species. Different sites are treated separatedly (a reservoir
                can have a net coverage of 1, and the main site as well). Order matches self.index_map

        returns
            np.ndarray: Forward rate only in 1/s
        """
        # Compute rates
        f_rates = self._calc_rates(y)[:,1]

        # Solution array
        return self.reaction_matrix @ f_rates

    #-----
    # ODE jacobian
    def _jac(self, y: np.ndarray) -> np.ndarray:
        """Computes the Jacobian matrix for ALL the reactions in the network
        with regards to each reaction (drxn/dtheta). Has the following form:
        [[drxn_1/dtheta_1 , drxn_1/dtheta_2 ... drxn_1/dtheta_n]
        [...]
        [drxn_m/dtheta_1 , drxn_m/dtheta_2 ... drxn_m/dtheta_n]]

        Args:
            y (np.ndarray): Array of compositions

        Returns:
            np.ndarray: Jacobian matrix
        """
        # Get dri/dthetak matrix (shape = (n_rxns, n_elements being tracked))
        jac = []
        for rxn_name, rxn_dictio in self.rate_map.items(): # Iterate over ALL reactions
            ri_dtheta = []
            for species in self.index_map.values(): # Iterate over ALL species indices
                # Count how many times the species appears as one of the reactants or products
                reac_count = rxn_dictio["reac"].count(species)
                prod_count = rxn_dictio["prod"].count(species)

                # If the species is not part of the reaction, dr_i / dtheta_k = 0 and we continue
                if reac_count == 0 and prod_count == 0:
                    ri_dtheta.append(float(0))
                    continue
                # We have to check though we only got ONE positive number then
                elif (reac_count > 0) and (prod_count > 0):
                    raise RuntimeError(f"Species {species} appreas on both sides of reaction {rxn_name}")

                # Now we compute the jacobian matrix
                # Label and species count to use (avoiding double work)
                if reac_count > 0:
                    count_use, label_use, k_use, sign = (reac_count, "reac", self.reactions[rxn_name].kfwd, 1)
                else:
                    count_use, label_use, k_use, sign = (prod_count, "prod", self.reactions[rxn_name].krev, -1)

                # Other species involved in the reaction
                other_species = [el for el in rxn_dictio[label_use] if el != species]
    
                # Sanity check (only one adsorbing or desorbing species)
                is_gas = count_use if species in self.gas_indices else 0 #This will force only one gas species. Else, is_gas will be greater than 1 and raise AssertionError
                other_gas = sum([1 for el in other_species if el in self.gas_indices])
                assert int(is_gas + other_gas) <= 1, "Can't have multiple gas species in one reaction!"

                # Derivative and gas terms
                s_term = count_use*y[species]**(count_use-1) if int(count_use) > 1 else 1
                P_term = self.p if other_gas == 1 else 1

                # Putting everything together
                ri_dtheta.append(sign * k_use * s_term * np.prod(y[other_species]) * P_term)

            jac.append(ri_dtheta)

        return np.array(jac)

    def get_jacobian(self, y: np.ndarray) -> np.ndarray:
        """Returns the jacobian matrix for each element in the reaction network (not for each reaction).
        In other words, the jacobian with regards to d_theta_i/dt (theta_i'):
        [[dtheta_1'/theta_1, dtheta_1'/theta_2, ... dtheta_1'/theta_n]
        ...
        [dtheta_n'/theta_1, dtheta_n'/theta_2, ... dtheta_n'/theta_n]]

        Args:
            y (np.ndarray): Compositions

        Returns:
            np.ndarray: Jacobian matrix (shape equal to len(y),len(y))
        """
        # get jacobian matrix for reactions (dri/dthetak)
        _jac = self._jac(y)
        return self.reaction_matrix @ _jac

    #-----
    # Steady state solutions
    def _ss_pre(self, y_surf):
        """Common normalization (preliminar) step for steady-state functions"""
        # Invariant compositions (flow gas)
        y_gas = self.initial_system[list(self.gas_indices)]

        # Total compositions to compute rates
        y = np.concat([y_gas, y_surf])

        # # Mass balance of free sites
        # for surf_name, idx in self.coverage_map.items():
        #     surf_idx = self.index_map[surf_name]
        #     other_idx = list(idx - {surf_idx})
        #     y[surf_idx] = max(1 - sum(y[other_idx]), 0) #No negative coverage allowed
        
        return y

    def _fun_ss(self, y_surf: np.array) -> np.ndarray:
        """Modifies the dy/dt method to ONLY return the steady-state-relevant functions
        Does the internal mass balance of sites

        Args:
            y_surf (np.array): Array of surface compositions

        Returns:
            np.ndarray: Surface coverage rate changes
        """
        #ygas
        y_gas = self.initial_system[list(self.gas_indices)]
        # Get y and normalize
        y = self._ss_pre(y_surf)
        # Get rates for each species
        dydt = self.get_dydt(y)
        # Return the SURFACE ONLY dydt
        return dydt[len(y_gas):]

    def _jac_ss(self, y_surf: np.array) -> np.ndarray:
        """Modifies the main jacobian method to ONLY return the steady-state relevant 
        jacobian matrix. Does the internal mass balance of sites

        Args:
            y_surf (np.array): Array of surface compositions

        Returns:
            np.ndarray: Jacobian matrix [len(y_surf), len(y_surf)]
        """
        #ygas
        y_gas = self.initial_system[list(self.gas_indices)]
        # Get y and normalize
        y = self._ss_pre(y_surf)
        # Get rates for each species
        jac = self.get_jacobian(y)
        # Return the SURFACE ONLY dydt
        return jac[len(y_gas):,len(y_gas):]

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
        # Randomized initial guess
        if y0 is None:
            y0 = self._normalize_y(np.random.uniform(size=len(self.initial_system)))
        elif len(y0) != len(self.initial_system):
            raise ValueError("Initial guess must have same length as initial guess... Include gas and surface species in here!")
        
        # Take only the surface concentrations (gas-phase assumed invariant)
        y0 = y0[len(self.gas_indices):]

        # Preliminars
        idx = 0 #N iterations
        factor = 1 #Tightness factor (multiplies tol)
        success = False # Will be True if calculation converges

        while idx < max_iters:
            # sol = root(
            #     fun = self._fun_ss,
            #     x0 = y0,
            #     method = method,
            #     jac = None if idx == 0 else self._jac_ss,
            #     tol=1e-6*factor
            # )
            
            sol = least_squares(
                fun = self._fun_ss,
                x0 = y0,
                jac = '2-point' if idx == 0 else self._jac_ss,
                bounds = (np.zeros(len(y0)), np.ones(len(y0)))
            )

            y0 = sol.x
            y = np.concat((self.initial_system[list(self.gas_indices)], y0))

            # Tracking the surface coverages
            surf_sum = [sum(y[list(surf_indices)]) for surf_indices in self.coverage_map.values()]
            if self.verbose:
                print(f"iter {idx:3d}:  {' , '.join(str(x)[:8] for x in surf_sum)}", end="\r")

            # Check if calculation converged
            if np.any(np.abs(np.array(surf_sum) - 1) > 0.05) or any(np.round(np.array(y0),2) < 0) or np.any(np.abs(self.get_dydt(y)) > 1e-6):
                y0 = self._normalize_y(np.abs(y))[len(self.gas_indices):]
                factor = factor/10**(1/4) if factor > 1e-8 else factor #Tighter tol if needed
                idx += 1
            else:
                success = True
                break

        y = np.concat((self.initial_system[list(self.gas_indices)], sol.x))
        return SteadyStateResults(y, success) 