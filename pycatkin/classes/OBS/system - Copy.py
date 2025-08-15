"""Patched version of system.py
This version includes changes for the calculation of steady states
"""
from pycatkin.classes.reactor import *
import os
import copy
import pickle
from scipy.integrate import solve_ivp, ode
from scipy.optimize import fsolve, least_squares, minimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from pycatkin.classes.state import State
from pycatkin.classes.reaction import Reaction
from pycatkin.classes.reactor import Reactor
from pycatkin.classes.energy import Energy

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
        self.y0 = y0

        # Blank assignments (to be populated when reading from input file)
        self.states = dict()
        self.unique_states = set()
        self.reactions = dict()
        self.reactor = None
        self.energy_landscapes = dict

        
        self.adsorbate_indices = None
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
            y (np.array): concentration array

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

    def _calc_rates(self, y):
        """Computes the forward and backward rates for the given reaction network

        Sets:
            self.rates (np.array) of shape (n_reactions, 2), with first column for forward rate, second column for backward rate
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

    def _get_species_rates(self, y):
        """Gets the right-hand side of the ODE system for each individual species
        Basically, it computes the net generation/consumption rates for each individual species.
        The rates are stored in a numpy array. The order of the array matches the state indexes defined 
        in the index_map

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

    def ss_fun(self, y:np.array) -> np.array:
        """Steady-state function. The rates of all products must be zero, and site conservation must prevail
        """
        # Site conservation laws
        site_conservation = np.array(
            [np.sum(y[list(surf_indices)]) - 1 for surf_indices in self.coverage_map.values()]
        )

        # Fractional conservation must be one
        gas_conservation = np.array(
            [np.sum(y[list(self.gas_indices)]) - 1]
        )

        #Species rate
        dydt = self._get_species_rates(y)

        if isinstance(self.reactor, pycatkin.classes.reactor.InfiniteDilutionReactor):
            print("INFINITE DILUTION")
            return np.concatenate((dydt, site_conservation))
        else:
            return np.concatenate((dydt, site_conservation, gas_conservation))

    def find_steady(self):
        """Finds the a steady-state solution.

        NOTE: I wasn't able to find a way to FORCE the site conservation. Nonetheless, I have found that, if
        you resubmit a failed calculation, you often find the correct root!
        """
        # Randomized initial guess
        if self.y0 is None:
            y0 = self._normalize_y(np.random.uniform(size=len(self.initial_system)))
        else:
            y0 = self.y0

        # Stores the number of iterations
        idx = 0
        factor = 1

        while idx < 3:
            # Solve 
            sol = least_squares(
                    fun = self.ss_fun,
                    x0 = y0,
                    jac = '3-point',
                    bounds = (0,1),
                    method = "trf",
                    ftol = self.ode_params["ftol"]/factor,
                    xtol = None,
                    loss = "soft_l1",
                    max_nfev  = self.ode_params["nsteps"],
                    verbose = 2 if self.verbose else 0,
            )

            # Coverages
            gas_sum = sum(sol.x[:8])
            h_sum = sum(sol.x[8:10])
            s_sum = sum(sol.x[10:])

            # Check if calculation converged
            if np.any(0.95 > np.array([gas_sum, h_sum, s_sum])) or np.any(1.05 < np.array([gas_sum, h_sum, s_sum])):
                print("Steady-state solution not found, repeating least_squares with tighter ftol")
                y0 = self._normalize_y(sol.x)
                factor /= 10
                idx += 1
            
            else:
                self.y0 = sol.x
                break    

        return sol

    # CONTINUE HERE
    def run_and_return_tof(self, tof_terms, ss_solve=False):
        """Integrate or solve for the steady state and
        compute the TOF by summing steps in tof_terms

        Returns array of xi_i terms for each step i."""

        if ss_solve:
            full_steady = self.find_steady()
        else:
            self.solve_odes()
            full_steady = self.solution[-1, :]

        self.reaction_terms(full_steady)

        tof = 0.0
        for rind, r in enumerate(self.species_map.keys()):
            if r in tof_terms:
                tof += self.rates[rind, 0] - self.rates[rind, 1]
        return tof

    def activity(self, tof_terms, ss_solve=False):
        """Calculate the activity from the TOF

        Returns the activity."""

        self.conditions = None  # Force rate constants to be recalculated

        tof = self.run_and_return_tof(tof_terms=tof_terms, ss_solve=ss_solve)

        activity = (np.log((h * tof) / (kB * self.params['temperature'])) *
                    (R * self.params['temperature'])) * 1.0e-3 / eVtokJ

        return activity


    # def reaction_derivatives(self, y):
    #     """Constructs derivative of reactions wrt each species
    #     by multiplying rate constants by reactant coverages/pressures.

    #     Returns an (Nr x Ns) array of derivatives."""

    #     self.check_rate_constants()

    #     ny = max(y.shape)
    #     y = y.reshape((ny, 1))
    #     dr_dtheta = np.zeros((len(self.species_map), ny))

    #     def prodfun(reac, vartype, species):
    #         val = 1.0
    #         scaling = 1.0
    #         nsp = len(self.species_map[reac][vartype])
    #         for j in range(nsp):
    #             if j != species:
    #                 val *= y[self.species_map[reac][vartype][j], 0]
    #                 if vartype in ['preac', 'pprod']:
    #                     scaling = bartoPa
    #         return val * scaling

    #     for rind, r in enumerate(self.species_map.keys()):
    #         kfwd = self.rate_constants[r]['kfwd'] + self.species_map[r]['perturbation']
    #         krev = self.rate_constants[r]['krev'] * (1.0 + self.species_map[r]['perturbation'] /
    #                                                  self.rate_constants[r]['kfwd'])

    #         yfwd = prodfun(reac=r, vartype='yreac', species=None)
    #         yrev = prodfun(reac=r, vartype='yprod', species=None)
    #         pfwd = prodfun(reac=r, vartype='preac', species=None)
    #         prev = prodfun(reac=r, vartype='pprod', species=None)

    #         for ind, i in enumerate(self.species_map[r]['yreac']):
    #             dr_dtheta[rind, i] += kfwd * pfwd * prodfun(reac=r, vartype='yreac', species=ind)
    #         for ind, i in enumerate(self.species_map[r]['yprod']):
    #             dr_dtheta[rind, i] -= krev * prev * prodfun(reac=r, vartype='yprod', species=ind)
    #         for ind, i in enumerate(self.species_map[r]['preac']):
    #             dr_dtheta[rind, i] += kfwd * yfwd * prodfun(reac=r, vartype='preac', species=ind)
    #         for ind, i in enumerate(self.species_map[r]['pprod']):
    #             dr_dtheta[rind, i] -= krev * yrev * prodfun(reac=r, vartype='pprod', species=ind)
    #     return dr_dtheta

    # def species_jacobian(self, y):
    #     """Constructs derivatives of species ODEs
    #     for adsorbate coverages and pressures.

    #     Returns Jacobian with shape (Ns x Ns)."""

    #     dr_dtheta = self.reaction_derivatives(y=y)

    #     ny = max(y.shape)
    #     jac = np.zeros((ny, ny))
    #     for rind, rinfo in enumerate(self.species_map.values()):
    #         for sp1 in range(ny):
    #             for sp2 in rinfo['yreac']:  # Species consumed
    #                 jac[sp2, sp1] -= dr_dtheta[rind, sp1] * rinfo['scaling']
    #             for sp2 in rinfo['yprod']:  # Species formed
    #                 jac[sp2, sp1] += dr_dtheta[rind, sp1] * rinfo['scaling']
    #             for sp2 in rinfo['preac']:
    #                 jac[sp2, sp1] -= dr_dtheta[rind, sp1] * rinfo['scaling'] * rinfo['site_density']
    #             for sp2 in rinfo['pprod']:
    #                 jac[sp2, sp1] += dr_dtheta[rind, sp1] * rinfo['scaling'] * rinfo['site_density']
    #     return jac

    # def solve_odes(self):
    #     """Wrapper for ODE integrator.

    #     """

    #     self.conditions = None  # Force rate constants to be recalculated

    #     # Set initial coverages to zero if not specified
    #     yinit = np.zeros(len(self.snames))
    #     if self.params['start_state'] is not None:
    #         for s in self.params['start_state'].keys():
    #             yinit[self.snames.index(s)] = self.params['start_state'][s]

    #     # Set inflow mole fractions to zero if not specified
    #     yinflow = np.zeros(len(self.snames))
    #     if self.params['inflow_state'] is not None:
    #         for s in self.params['inflow_state'].keys():
    #             yinflow[self.snames.index(s)] = self.params['inflow_state'][s]

    #     if self.params['verbose']:
    #         print('=========\nInitial conditions:\n')
    #         for s, sname in enumerate(self.snames):
    #             print('%15s : %1.2e' % (sname, yinit[s]))
    #         if yinflow.any():
    #             print('=========\nInflow conditions:\n')
    #             for s, sname in enumerate(self.snames):
    #                 if s in self.gas_indices:
    #                     print('%15s : %1.2e' % (sname, yinflow[s]))

    #     solfun = lambda tval, yval: self.reactor.rhs(self.species_odes)(t=tval, y=yval, T=self.params['temperature'],
    #                                                                     inflow_state=yinflow)
    #     jacfun = lambda tval, yval: self.reactor.jacobian(self.species_jacobian)(t=tval, y=yval,
    #                                                                              T=self.params['temperature'])

    #     # Create ODE solver
    #     if self.params['ode_solver'] == 'solve_ivp':
    #         sol = solve_ivp(fun=solfun, jac=jacfun if self.params['jacobian'] else None,
    #                         t_span=(self.params['times'][0], self.params['times'][-1]),
    #                         y0=yinit, method='BDF',
    #                         rtol=self.params['rtol'], atol=self.params['atol'])
    #         if self.params['verbose']:
    #             print(sol.message)
    #         self.times = sol.t
    #         self.solution = np.transpose(sol.y)
    #     elif self.params['ode_solver'] == 'ode':
    #         sol = ode(f=solfun, jac=jacfun if self.params['jacobian'] else None)
    #         sol.set_integrator('lsoda', method='bdf', rtol=self.params['rtol'], atol=self.params['atol'])
    #         sol.set_initial_value(yinit, self.params['times'][0])
    #         self.times = np.concatenate((np.zeros(1),
    #                                      np.logspace(start=np.log10(self.params['times'][0]
    #                                                                 if self.params['times'][0] else 1.0e-8),
    #                                                  stop=np.log10(self.params['times'][-1]),
    #                                                  num=self.params['nsteps'],
    #                                                  endpoint=True)))
    #         self.solution = np.zeros((self.params['nsteps'] + 1,
    #                                   len(self.snames)))
    #         self.solution[0, :] = yinit
    #         i = 1
    #         while sol.successful() and i <= self.params['nsteps']:
    #             sol.integrate(self.times[i])
    #             self.solution[i, :] = sol.y
    #             i += 1
    #     else:
    #         raise RuntimeError('Unknown ODE solver specified. Please use solve_ivp or ode, or add a new option here.')

    #     if self.params['verbose']:
    #         print('=========\nFinal conditions:\n')
    #         for s, sname in enumerate(self.snames):
    #             print('%15s : %9.2e' % (sname, self.solution[-1][s]))


    # def degree_of_rate_control(self, tof_terms, ss_solve=False, eps=1.0e-3):
    #     """Calculate the degree of rate control xi_i

    #     Returns array of xi_i terms for each step i."""

    #     self.conditions = None  # Force rate constants to be recalculated

    #     r0 = self.run_and_return_tof(tof_terms=tof_terms, ss_solve=ss_solve)
    #     xi = dict()

    #     if self.params['verbose']:
    #         print('Checking degree of rate control...')

    #     for r in self.reactions.keys():
    #         self.species_map[r]['perturbation'] = eps * self.rate_constants[r]['kfwd']
    #         xi_r = self.run_and_return_tof(tof_terms=tof_terms, ss_solve=ss_solve)
    #         self.species_map[r]['perturbation'] = -eps * self.rate_constants[r]['kfwd']
    #         xi_r -= self.run_and_return_tof(tof_terms=tof_terms, ss_solve=ss_solve)
    #         xi_r *= (self.rate_constants[r]['kfwd']) / (2.0 * eps * self.rate_constants[r]['kfwd'] * r0)
    #         xi[r] = xi_r
    #         self.species_map[r]['perturbation'] = 0.0

    #         if self.params['verbose']:
    #             print(r + ': done.')

    #     return xi