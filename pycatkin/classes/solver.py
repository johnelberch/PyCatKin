from pycatkin.classes.system import System, SteadyStateResults
import numpy as np
from scipy.integrate import ode, solve_ivp
from scipy.optimize import fsolve, least_squares, minimize, root, Bounds
from typing import Union
from typing import NamedTuple

class SolScore(NamedTuple):
    """
    Quantifies how well a solution fits the data
    """
    y_surf: np.array
    max_rate: float
    max_jac: float
    surf_sum: list[float,...]

class SteadyStateSolver():
    def __init__(
        self, 
        system: System, 
        ss_guess: np.ndarray = None,
        verbose: bool = False,
        ):
        """Initializes a solver instance.

        Args:
            system (System): pycatkin.classes.system.System to use
            ss_guess (np.ndarray, optional): Array of steady-state surface coverages.
                Length must match number of surface sites. Defaults to None.
            verbose (bool, optional): Printing or not the progress. Defaults to False.

        Raises:
            ValueError: system provided must be an instance of pycatkin.classes.system.System
            ValueError: Initial guess must match the number of surface sites

        Sets:
            self.sys: system to use
            self.verbose: verbose
            self.ygas: invariant gas concentration array
            self.surf_map: dictionary mapping the different surface indices and sites (indices adjusted for y_surf only)
            self.ss_guess: steady-state guess
        """
        # Sanity check
        if not isinstance(system, System):
            raise ValueError("system must be Pycatkin System")

        # Assignments
        self.sys = system
        self.verbose = verbose

        # Gas-species concentrations (invariant!) and surface indices
        self.ygas = self.sys.initial_system[:len(self.sys.gas_indices)]
        surf_map = {}
        for surf_id, idx_set in self.sys.coverage_map.items():
            surf_map[surf_id] = {idx - len(self.ygas) for idx in idx_set} #basically, it adjusts the indices to exclude the gas-phase (which is always located first!)
        self.surf_map = surf_map

        # Surface coverage initial guess (randomized is None is provided, else, it is checked for consistency)
        n_surf_species = sum([len(v) for v in self.surf_map.values()])
        if ss_guess is None:
            self.ss_guess = self._norm(np.random.uniform(size=n_surf_species))
        elif len(ss_guess) != n_surf_species:
            raise ValueError(f"Initial guess must have same length as number of surface sites = {n_surf_species}")
        else:
            self.ss_guess = ss_guess

    #-----
    # AUXILIARY FUNCTIONS
    def test_convergence(
        self, 
        y_surf: np.ndarray, 
        rate_tol: float = 1e-4, 
        coverage_tol: float = 5e-2, 
        pos_jac_tol: float = 1e-2,
        log: bool = False,
        **kwargs) -> bool:
        """Checks that a calculation is well converged

        Args:
            y_surf (np.ndarray): Surface coverages
            rate_tol (float, optional): Max allowed net rate in [1/s]. Defaults to 1e-4.
            coverage_tol (float, optional): Max allowed net coverage deviation from 1. Defaults to 5e-2.
            pos_jac_tol (float, optional): Max allowed jacobian matrix eigenvalue (all should be negative). Defaults to 1e-2.
            log (bool, optional): Whether to log the tests or not. Defaults to True

        Returns:
            bool: Whether the checks passed or not. True means all tests passed
        """
        ## Checks for convergence (For all tests, True means failed)
        # Test 1: Rate of surface species is zero
        rate_residual = np.max(np.abs(self.sys._fun_ss(y_surf)))
        rate_fail = rate_residual > rate_tol

        # Test 2: Posifive surface coverages
        spos_fail = any(np.round(np.array(y_surf),2) < 0)

        # Test 3: Surface coverages add to one (need to create whole array to get indices I need)
        y_all = np.concat((self.ygas,y_surf))
        surf_sum = [sum(y_all[list(surf_indices)]) for surf_indices in self.sys.coverage_map.values()]
        ssum_fail = np.any(np.abs(np.array(surf_sum) - 1) > coverage_tol)

        # Test 4: The jacobian eigenvalues are negative numbers
        JAC = self.sys._jac_ss(y_surf)
        eigV = np.linalg.eig(JAC).eigenvalues
        jac_complex = np.iscomplex(eigV).any()
        negjac_fail = np.any(eigV.real > pos_jac_tol) if jac_complex else np.any(eigV > pos_jac_tol)

        if log:
            s = f"""    - CHECKS: rate {~rate_fail} | surf_sum {~ssum_fail} | jac_eigV {~negjac_fail}
                - surf_sum = {surf_sum}
                - rate_residual = {rate_residual}
                - jacobian_eigV_max = {max(eigV.real) if jac_complex else max(eigV)}
            """
            print(s)

        # Verifies all checks and returns passed/failed result
        if any([rate_fail, spos_fail, ssum_fail, negjac_fail]):
            return False
        else:
            return True

    def _norm(self, y_surf: np.array) -> np.array: 
        """Normalizes the concentrations of surface coverages. Surface products MUST ADD to one (site conservation).
        Readapted from normalize_y in System

        # NOTE: Caps the lowest concentration to self.min_threshold()

        Args:
            y_surf (np.array): concentration array. Surface coverage. Different sites are treated separatedly 
                (a reservoir can have a net coverage of 1, and the main site as well). Order matches self.index_map

        Returns:
            np.array: normalized array of surface coverages
        """
        y_surf = np.where(y_surf < self.sys.min_tol, self.sys.min_tol, y_surf) 

        for surf_indices in self.surf_map.values():
            y_surf[list(surf_indices)] /= np.sum(y_surf[list(surf_indices)])

        # Return caped y to min tolerance
        return y_surf

    def _score(
        self,
        y_surf
        ) -> SolScore:
        """
        Computes a "score" so we can compare how good a solution is.
        This is not an elegant approach, but I just want to keep the "closest"
        solution to convergence
        """
        # Get values to compare
        max_rate = np.max(np.abs(self.sys._fun_ss(y_surf)))
        surf_sum = [sum(y_surf[list(surf_indices)]) for surf_indices in self.surf_map.values()]
        JAC = self.sys._jac_ss(y_surf)
        eigV = np.linalg.eig(JAC).eigenvalues
        jac_complex = np.iscomplex(eigV).any()
        max_jac = np.max(eigV.real) if jac_complex else np.max(eigV)

        return SolScore(y_surf = y_surf, max_rate = max_rate, max_jac = max_jac, surf_sum = surf_sum)
        
    @staticmethod
    def compare_scores(
        s1:SolScore, 
        s2:SolScore,
        rate_tol: float = 1e-4,
        coverage_tol: float = 5e-2, 
        pos_jac_tol: float = 1e-2,
        **kwargs):
        """Compares two scores, returns best"""

        s1_rank = [
            s1.max_rate < rate_tol,
            np.all(np.abs(np.array(s1.surf_sum) - 1) < coverage_tol),
            s1.max_jac < pos_jac_tol
        ]

        s2_rank = [
            s2.max_rate < rate_tol,
            np.all(np.abs(np.array(s2.surf_sum) - 1) < coverage_tol),
            s2.max_jac < pos_jac_tol
        ]

        # Deviations from perfect coverages
        _d1 = np.abs(np.linalg.norm(s1.surf_sum)-1)
        _d2 = np.abs(np.linalg.norm(s2.surf_sum)-1)

        # If both rates work
        if s1_rank[0] and s2_rank[0]:
            # If both surfaces work
            if s1_rank[1] and s2_rank[1]:
                # Return lowest jacobian
                return s1 if s1.max_jac < s2.max_jac else s2

            # One surface doesn't work. Return the one working
            elif s1_rank[1] ^ s2_rank[1]:
                return s1 if s1_rank[1] else s2

            # No surface works
            else:
                # Both jacobian work
                if s1_rank[2] and s2_rank[2]:
                    #return the surfaces closest to 1 (min score)
                    return s1 if  _d1 < _d2 else s2
                # One jacobian works (return the working case)
                elif s1_rank[2] ^ s2_rank[2]:
                    return s1 if s1_rank[2] else s2
                # No jacobian works. Return best-performing surface
                else:
                    return s1 if  _d1 < _d2 else s2
            
        # One rate works (return the working case)
        elif s1_rank[0] ^ s2_rank[0]:
            return s1 if s1_rank[0] else s2

        # No rate works
        else:
            # Return the best performing rate
            return s1 if  s1.max_rate < s2.max_rate else s2

    #-----
    # MAIN SOLVER FUNCTIONS (DIFFERENT APPROACHES)
    def solve_root(
        self, 
        max_iters: int = 30, 
        method: str ="hybr", 
        use_jac: bool = True, 
        tol: float =1e-8,
        test_convergence_kwargs: dict = None,
        log_every: int = 5,
        ) -> SteadyStateResults:
        """Finds the a steady-state solution using scipy.optimize.root.

        Args:
            max_iters (int, optional): How many times to try to refine the solution. Solution refinements are done so 
                the site conservation laws (and gas concentration profiles if required) are within their limits with 
                less than 5% error
            method (str, optional): Method to be used by SciPy root function. Defaults to hybr
            use_jac (bool, optional): Use the analytical jacobian or not. Defaults to True
            tol (float, optional): Initial tolerance. Defaults to 1e-8
            test_convergence_kwargs (dict, optional): Kwargs for test_convergence method
            log_every (bool, optional): How frequent to log the progress.

        Returns:
            SteadyStateResults: Named tuple with attributes x (1D numpy array of concentrations for all species) and
                success (boolean indicating if the calculation converged)
        """
        # Preliminars
        iter_n = 0 #iterations
        factor = 1 #Tightness factor (gradually decrease the solver tolerance)
        success = False #Exit flag for iteration loop

        # Function, jacobian, residual (if needed)
        fun = self.sys._fun_ss
        jac = self.sys._jac_ss if use_jac else None
        test_convergence_kwargs = {} if test_convergence_kwargs is None else test_convergence_kwargs
        x0 = self.ss_guess

        # Score the initial guess
        s_keep = self._score(x0)

        #Iterate until the solution is found
        while iter_n < max_iters and not success:
            # Normalize x0 
            x0 = self._norm(x0)

            # Get the solution
            sol = root(
                fun = fun,
                x0 = x0,
                method=method,
                jac = jac,
                tol = tol*factor
            )

            # Check for convergence
            log = True if (self.verbose and iter_n%log_every == 0) else False
            test_convergence_kwargs["log"] = log
            success = self.test_convergence(sol.x, **test_convergence_kwargs)

            # Update iteration number, surface coverage, and factor
            x0 = sol.x
            iter_n += 1
            factor = factor/10**(1/4) if tol*factor < 1e-16 else factor #Tighter tol if needed until low cap is reached

            # Compare the scores
            s_new = self._score(x0)
            s_keep = self.compare_scores(s_keep, s_new, **test_convergence_kwargs)

        # Return the optimized coverages / convergences and the success flag
        return SteadyStateResults(sol.x, success) if success else SteadyStateResults(s_keep.y_surf, False)

    def solve_minimize(
        self, 
        max_iters: int = 30, 
        method: str = None, 
        use_jac: Union[bool,str] = True, 
        tol: float =1e-8,
        test_convergence_kwargs: dict = None,
        log_every: int = 5,
        use_bounds = True,
        ) -> SteadyStateResults:
        """Finds the a steady-state solution using scipy.optimize.minimize.

        Args:
            max_iters (int, optional): How many times to try to refine the solution. Solution refinements are done so 
                the site conservation laws (and gas concentration profiles if required) are within their limits with 
                less than 5% error
            method (str, optional): Method to be used by SciPy minimize function. Defaults to hybr
            use_jac (bool, optional): Use the analytical jacobian or not. Defaults to True. Could be a string representing a particular jacobian
            tol (float, optional): Initial tolerance. Defaults to 1e-8
            test_convergence_kwargs (dict, optional): Kwargs for test_convergence method
            log_every (bool, optional): How frequent to log the progress.
            use_bounds (bool, optional): Whether to set the system bounds or not.

        Returns:
            SteadyStateResults: Named tuple with attributes x (1D numpy array of concentrations for all species) and
                success (boolean indicating if the calculation converged)
        """
        # Preliminars
        iter_n = 0 #iterations
        factor = 1 #Tightness factor (gradually decrease the solver tolerance)
        success = False #Exit flag for iteration loop

        # Function, jacobian, residual (if needed)
        if isinstance(use_jac,str):
            jac = use_jac
        elif use_jac:
            def jac(y_surf):
                row_id = np.argmax(np.abs(self.sys._fun_ss(y_surf)))
                return self.sys._jac_ss(y_surf)[row_id,:]
        else:
            jac = None
        test_convergence_kwargs = {} if test_convergence_kwargs is None else test_convergence_kwargs
        x0 = self.ss_guess
        def fun(y_surf): return np.max(np.abs(self.sys._fun_ss(y_surf)))
        bounds = Bounds(lb=0,ub=1) if use_bounds else None

        # Score the initial guess
        s_keep = self._score(x0)

        #Iterate until the solution is found
        while iter_n < max_iters and not success:
            # Normalize x0 
            x0 = self._norm(x0)

            # Get the solution
            sol = minimize(
                fun = fun,
                x0 = x0,
                method=method,
                jac = jac,
                bounds = bounds,
                tol = tol*factor
            )

            # Check for convergence
            log = True if (self.verbose and iter_n%log_every == 0) else False
            test_convergence_kwargs["log"] = log
            success = self.test_convergence(sol.x, **test_convergence_kwargs)

            # Update iteration number, surface coverage, and factor
            x0 = sol.x
            iter_n += 1
            factor = factor/10**(1/4) if tol*factor < 1e-16 else factor #Tighter tol if needed until low cap is reached

            # Compare the scores
            s_new = self._score(x0)
            s_keep = self.compare_scores(s_keep, s_new, **test_convergence_kwargs)

        # Return the optimized coverages / convergences and the success flag
        return SteadyStateResults(sol.x, success) if success else SteadyStateResults(s_keep.y_surf, False)

    def solve_ode(
        self, 
        method: str = "RK45", 
        use_jac: bool = True, 
        rtol: float = 1e-10,
        atol: float = 1e-12,
        tmax: float = 1e4,
        test_convergence_kwargs: dict = None,
        ) -> SteadyStateResults:
        """Finds the a steady-state solution using scipy.integrate.solve_ivp.

        Args:
            method (str, optional): Method to be used by scipy.integrate.solve_ivp function. Defaults to None
            use_jac (bool, optional): Use the analytical jacobian or not. Defaults to True. 
            rtol (float, optional): in line with scipy
            atol (float, optional): in line with scipy
            test_convergence_kwargs (dict, optional): Kwargs for test_convergence method

        Returns:
            SteadyStateResults: Named tuple with attributes x (1D numpy array of concentrations for all species) and
                success (boolean indicating if the calculation converged)
        """
        def fun(t, ysurf): return self.sys._fun_ss(ysurf)
        def jac_fun(t, ysurf) : return self.sys._jac_ss(ysurf)
        test_convergence_kwargs = {} if test_convergence_kwargs is None else test_convergence_kwargs
        y0 = self.sys.initial_system[len(self.sys.gas_indices):]

        sol = solve_ivp(
            fun = fun,
            t_span = (0, tmax),
            y0 = y0,
            method = method,
            rtol = 1e-10,
            atol = 1e-12,
            jac = jac_fun if use_jac else None,
        )

        y_new = sol.y[:,-1]

        # Check for convergence
        test_convergence_kwargs["log"] = True if self.verbose else False
        success = self.test_convergence(y_new, **test_convergence_kwargs)

        # Return the optimized coverages / convergences and the success flag
        return SteadyStateResults(y_new, success)