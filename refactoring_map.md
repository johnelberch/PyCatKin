# Output from opencode
- State (also base of ScalingState)
  |- name: crucial  (dict key in System.states; used by Reaction printouts, Energy labels)
  |- state_type: crucial  (System classifies adsorbate/gas/surface; Reaction branches on gas for adsorption/desorption; Energy flags TS)
  |- Gelec: crucial  (Reaction sums electronic energies of reactants/products/TS; parallel_mkm.py writes states["sC"].Gelec = EC directly)
  |- mass: crucial  (Reaction.calc_rate_constants passes gas_state.mass to kads/kdes)
  |- sigma: crucial  (passed to kdes)
  |- inertia: crucial  (passed to kdes)
  |- __init__: crucial
  |- get_free_energy(T, p): crucial  (Reaction.calc_reaction_energy calls it for every reactant/product/TS)



  |- Gfree: internal  (computed cache; external readers must use get_free_energy())
  |- Gzpe, Gvibr, Gtran, Grota: internal  (contribution caches computed by calc_* methods)
  |- path, vibs_path: internal  (file handles used only by get_atoms/get_vibrations/calc_electronic_energy)
  |- gasdata: internal  (only iterated inside State's translational/rotational methods)
  |- add_to_energy, read_from_alternate, truncate_freq, energy_source, freq_source: internal
  |- free_source, tran_source, rota_source, vibr_source: internal  (source-tracking flags)
  |- atoms, freq, i_freq, shape: internal  (lazy-loaded caches)
  |- calc_free_energy, calc_electronic_energy, calc_zpe, calc_vibrational_contrib, calc_translational_contrib, calc_rotational_contrib: internal
  |- get_atoms, get_vibrations: internal
  |- get_potential_energy, set_energy_modifier: internal (set_energy_modifier is only used by deprecated Uncertainty)
  |- save_pickle, save_pdb, view_atoms, save_vibrations, save_energy: internal utilities


- ScalingState (extends State)
  |- scaling_reactions: crucial  (holds dicts whose 'reaction' key is a Reaction object resolved by load_input; read by calc_electronic_energy/calc_free_energy)
  |- scaling_coeffs: internal  (only read inside ScalingState)
  |- dereference, use_descriptor_as_reactant: internal
  |- calc_electronic_energy, calc_free_energy: internal  (overrides of State methods)
  |- save_pickle, save_pdb, view_atoms: internal

SubagentsShellTerminalsescshow inactive ctrl+atabs ←/→… Switch agent… One-shot answer from the session's context without adding to the conversation… Change working directory… Clear session… Compact session… Connect an integration… Switch session… Copy last assistant message… View debug info… Open diff viewer… Open editor… Switch model variant… Exit the app… Export session transcript… Fork session… keep working until an objective is met [pause|resume|clear]… Help… Show release highlights… guided AGENTS.md setup… MCP servers… Switch model… New session… Open session or project… Pair device… Plugins… Open session or project… Open session or project… Exit the app… Exit the app… Reload configuration… Rename session… Restart service… Switch session… review changes [commit|branch|pr], defaults to uncommitted… Switch session… Open settings… Skills… Usage statistics… View status… New terminal… Switch theme… Switch model variant… Jump to message… Undo previous message… Update OpenCode… View usage… Switch model variant… Pair device… Manage workspaces
- Reaction (plus UserDefinedReaction, ReactionDerivedReaction)
  |- name: crucial  (System.reactions dict key; parallel_mkm.py does reactions["C_ads"])
Plan  |- reac_type: crucial  (System skips GHOST; Reaction.calc_rate_constants branches on it)·Nemotron 3.5 Super VL Free
  |- reactants, products, TS: crucial  (load_input wires State objects into them; System.rate_map maps their indexes)
~/Documents/02_UNIVERSITY/saa_paper/[01] Calculations/[04] MKM/PyCatKin  |- area: crucial  (System computes site_density = 1/area; Reaction uses it for kads/kdes)
  |- scaling: crucial  (stored in System.rate_map)
  |- kfwd, krev: crucial  (System._calc_rates multiplies by them)
  |- reversible: internal  (read only inside Reaction methods)
  |- dErxn_user (UserDefinedReaction): crucial  (parallel_mkm.py writes reactions["C_ads"].dErxn_user = EC)
  |- base_reaction (ReactionDerivedReaction): crucial  (cross-reference to another Reaction)
  |- Keq, dGrxn, dGa_fwd/rev, dErxn, dEa_fwd/rev: internal  (computed caches; outsiders use the get_* accessors)
  |- __init__: crucial
  |- calc_rate_constants(T, p): crucial  (System._update_rate_constants calls it for every reaction)
  |- get_reaction_energy(T, p, etype), get_reaction_barriers(T, p, etype): crucial  (ScalingState and functions/presets.py call them)
  |- calc_reaction_energy(T, p): internal  (called only by the above within Reaction)
  |- save_pickle: internal
- Reactor (plus InfiniteDilutionReactor, CSTReactor)
  |- is_adsorbate, is_gas: crucial  (written by System._mapping_reaction_indices via set_indices; read in rhs/jacobian)
  |- name, volume, catalyst_area, residence_time, flow_rate: internal  (constructor config consumed only inside Reactor)
  |- scaling, dynamic_indices: internal  (computed within Reactor)
  |- set_indices(is_adsorbate, is_gas): crucial  (called by System.build() → _mapping_reaction_indices)
  |- rhs, jacobian, set_scaling, get_dynamic_indices: internal  (currently not called by the current System; the ODE RHS/Jacobian live inside System itself — a simplification seam)
- System
  |- states: crucial  (load_input, parallel_mkm.py (states["sC"]), presets.py all index into it)
  |- reactions: crucial  (parallel_mkm.py writes reactions["C_ads"].dErxn_user; presets.py iterates it)
  |- reactor: crucial  (System.build() calls reactor.set_indices(...))
  |- T, p: crucial  (passed into state free-energy and reaction-rate computations)
  |- verbose: crucial  (parallel_mkm.py sets sis_use.verbose = False)
  |- coverage_map: crucial  (read by Solver to build surf_map and check convergence)
  |- gas_indices: crucial  (read by Solver to split gas/surface parts of initial_system)
  |- initial_system: crucial  (read by Solver for ygas and the ODE initial guess)
  |- min_tol: crucial  (read by Solver._norm)
  |- index_map, rate_map, reaction_matrix: internal  (computed by System for its own rate/jacobian bookkeeping)
  |- unique_states, start_state, inflow_state, ode_params, times, energy_landscapes: internal
  |- __init__, add_state, add_reaction, add_reactor, add_energy_landscape, build(): crucial  (build() is called by parallel_mkm.py; the adders by load_input)
  |- _update_rate_constants(T, p): internal  (only called by build() and _calc_rates, both inside System)
  |- _fun_ss(y_surf), _jac_ss(y_surf): crucial  (despite underscore names, these are the System→Solver seam; Solver.test_convergence/solve_* call them directly)
  |- _calc_rates, get_dydt, _jac, get_jacobian, find_steady(...): find_steady crucial (public solve API, also referenced by presets.py); the rest internal
  |- _names_to_indices, _mapping_reaction_indices, _get_initial_conditions, _normalize_y, _reactant_reaction_matrix, _ss_pre: internal  (steps orchestrated by build())
- Solver (SteadyStateSolver)
  |- sys: crucial  (holds the System; used everywhere)
  |- ss_guess: crucial  (parallel_mkm.py writes solver.ss_guess = ... and passes it to the constructor)
  |- ygas, surf_map: internal  (derived from sys.initial_system / sys.coverage_map)
  |- verbose: internal
  |- __init__(system, ss_guess, verbose): crucial
  |- solve_ode(...), solve_minimize(...), solve_root(...): crucial  (all three called by parallel_mkm.py)
  |- test_convergence(y_surf, ...): crucial  (called by parallel_mkm.py)
  |- _score(y_surf), compare_scores(s1, s2): crucial  (called by parallel_mkm.py despite underscore prefixes)
  |- _norm(y_surf): internal
- SteadyStateResults (NamedTuple)
  |- x: crucial
  |- success: crucial
- Energy (auxiliary; not in your overview, but wired via System.add_energy_landscape)
  |- name, minima, labels: crucial  (labels read by functions/presets.py)
  |- energy_landscape: crucial  (computed cache read by presets.py)
  |- construct_energy_landscape(T, p), draw_energy_landscape(...), evaluate_energy_span_model(...): crucial  (public API used by presets.py)
  |- save_pickle: internal

# Observations
1. System ↔ Solver: the actual interface is not just the public solve_* methods. Solver reaches into System._fun_ss, System._jac_ss, System.coverage_map, System.gas_indices, System.initial_system, and System.min_tol, and parallel_mkm.py additionally calls Solver._score / Solver.compare_scores and sets Solver.ss_guess. These "private" members are effectively public cross-class seams.
2. Reaction ↔ State: Reaction only needs state.Gelec, state.get_free_energy(T, p), and (for adsorption/desorption) state.mass / state.sigma / state.inertia.
3. System ↔ Reactor: in the current code System only calls reactor.set_indices(is_adsorbate, is_gas); reactor.rhs / reactor.jacobian are unused by System (the ODE RHS/Jacobian are computed inside System.get_dydt / get_jacobian). That's a prime simplification target.
4. State.path/vibs_path and the lazy-loading get_atoms/get_vibrations machinery matter only if states are built from VASP files; if states are always constructed with Gelec given directly (as parallel_mkm.py does), much of that can be dropped.

# REFACTORING
Evaluation of your refactoring plan
Overall: feasible and well-directed, but the parts are not equally easy, and the order you tackle them in matters. Your three ideas address real pain points (long constructor signatures, pickle security, implicit interfaces), but the classes are tightly coupled, so changing one class without adjusting its neighbors will break things.
Below is a part-by-part assessment, then the critical coupling issues you should resolve first, then a suggested refined plan.
1. Abstract base classes for the main classes
Feasibility: moderate. This is doable, but an ABC may be the wrong tool for this codebase.
What works well:
- The classes have clear conceptual roles (State = species free energy, Reaction = rates/energies, System = network assembly, Solver = root finding, Reactor = boundary conditions).
- Defining abstract contracts (e.g., State.get_free_energy(T, p), Reaction.calc_rate_constants(T, p), System.build()) would document the cross-class seams I mapped earlier.
Risks / cautions:
- isinstance checks. System.add_state, add_reaction, add_reactor, add_energy_landscape all do isinstance(x, SomeClass). If you introduce ABCs, you must keep those checks working (register subclasses or keep the concrete classes in the hierarchy).
- ScalingState breaks the simple State contract. Its calc_electronic_energy / calc_free_energy call back into Reaction.get_reaction_energy, which calls State.get_free_energy. That's a circular dependency any base-class design must accommodate.
- Pythonic alternative: Consider typing.Protocol (structural subtyping) instead of abc.ABC. The code already leans on duck typing (**kwargs from JSON). Protocols let you define "has a get_free_energy method" without forcing an inheritance chain, which fits a simplification effort better than heavy ABCs.
2. Dataclasses / pydantic BaseModels for settings
Feasibility: high. This is the strongest part of your plan.
load_input.py already constructs everything via **kwargs unpacked from JSON:
states[s] = State(name=s, **pck_system['states'][s])
sim_system = System(**sys_params)
reactions[r] = Reaction(name=r, **pck_system['reactions'][r])
So the JSON keys are the settings. Wrapping them in pydantic models would give you validation, defaults, and clearer documentation of required fields.
Where it gets tricky:
- Subclass proliferation. State ← ScalingState; Reaction ← UserDefinedReaction ← ReactionDerivedReaction; Reactor ← InfiniteDilutionReactor ← CSTReactor. Each subclass adds fields (scaling_coeffs, dErxn_user, base_reaction, flow_rate...). Pydantic handles inheritance fine, but you'll need a model per concrete class, not one per "main class."
- Mixed types. System takes start_state and inflow_state as dicts, T, p, verbose, plus ODE tolerances. Reactor needs volume/catalyst_area/residence_time/flow_rate with validation logic (CST computes residence time if missing). These validation rules currently live inside __init__ and would migrate into pydantic validators — good cleanup.
- Don't make pydantic the internal storage. Keep the pydantic models at the boundary (i.e., inside load_input.py or a new settings.py), then instantiate the runtime classes from them. If you try to make the runtime classes themselves pydantic models, you'll fight pydantic's immutability/validation semantics everywhere the code mutates attributes (e.g., sis_use.reactions["C_ads"].dErxn_user = EC in parallel_mkm.py).
3. Remove pickle usage
Feasibility: high, and a safe quick win.
My grep confirms path_to_pickle and save_pickle are defined in every class but never called anywhere in the package. load_input.py builds everything from JSON, not pickles. So removing:
- the path_to_pickle branch from each __init__
- every save_pickle method
…will not break any internal code.
One caveat: Parallel_run_CRC/parallel_mkm.py (your driver script, outside the package) uses pickle.dump for results.pkl and per-worker logs. That's outside pycatkin proper, but if your refactor covers the whole workflow you'll want to replace that too (e.g., json for small dicts, numpy.savez / parquet for arrays, or dill/cloudpickle only if you truly need arbitrary-object serialization).
Critical coupling issues to resolve first
Before introducing ABCs or settings models, fix these leaky seams. They are the parts most likely to break during refactoring:
Seam	Why it matters	What leaks
Solver ↔ System	Solver reaches into System._fun_ss, System._jac_ss, System.coverage_map, System.gas_indices, System.initial_system, System.min_tol	Private System methods are effectively public API
parallel_mkm.py ↔ Solver	Writes solver.ss_guess = ..., calls solver._score(...), solver.compare_scores(...)	Private Solver members used externally
Reaction ↔ ScalingState	ScalingState.calc_electronic_energy calls reaction.get_reaction_energy(...), which calls state.get_free_energy(...)	Circular state↔reaction dependency
System.build() monolith	Does 5 things in one call (_names_to_indices, _mapping_reaction_indices, _get_initial_conditions, _update_rate_constants, _reactant_reaction_matrix)	Hard to test or override individual steps
Reactor vs System	System only calls reactor.set_indices(...); reactor.rhs / reactor.jacobian are unused by System (ODE math lives inside System.get_dydt / get_jacobian)	Duplicate/parallel RHS logic
Suggested improvements / refined plan
Phase 0 — Safety first (no API change):
1. Remove path_to_pickle from all __init__ methods and delete all save_pickle methods.
2. Add thin pydantic settings models in a new module (e.g., pycatkin/settings.py) that mirror the JSON schema, and have load_input.py validate through them before unpacking **kwargs. This preserves input-file compatibility while giving you validation.
Phase 1 — Interfaces, not inheritance:
2. Define Protocols for StateLike, ReactionLike, SystemLike, SolverLike, ReactorLike capturing the crucial methods/attributes from my earlier map (e.g., StateLike.get_free_energy(T, p), ReactionLike.calc_rate_constants(T, p)). Use these for type hints; keep concrete classes as-is for now.
Phase 2 — Untangle the seams:
3. Make Solver stop reaching into System._fun_ss / _jac_ss. Either expose them as public methods on System, or move the steady-state residual/Jacobian into Solver itself.
4. Stop parallel_mkm.py from touching solver._score, solver.compare_scores, solver.ss_guess. Expose them publicly or move that orchestration into a Solver.run() method.
5. Break System.build() into separately callable, testable steps.
6. Decide what to do with Reactor.rhs/jacobian — either wire them into System (removing the duplicate ODE math in System) or mark them legacy.
Phase 3 — Optional ABC layer:
7. Only after the above, introduce ABCs if you genuinely need isinstance polymorphism. Otherwise the Protocols from Phase 1 are sufficient and lighter.
Compatibility guardrail: Whatever you do, keep load_input.py's ClassName(name=..., **json_dict) call pattern working. Your existing CH4_input.json and any other input files must continue to load unchanged.


# Notes on circular import in State
No — as written, my revised ScalingState would not fully cope with iterative changes to sC and sO.
Here's why:
1. Gelec is computed once and cached. In bind_reactions(), I call self.Gelec = self._calc_electronic_energy() and store the result. If the worker later does states["sC"].Gelec = EC and states["sO"].Gelec = EO, any ScalingState that depends on C_ads/O_ads (through get_reaction_energy) still holds the stale Gelec value from when bind_reactions() ran.
2. The original code was more dynamic. In your original ScalingState, __init__ did not compute Gelec. It was computed lazily inside calc_electronic_energy() / calc_free_energy() each time they were called, so it always read the current sC.Gelec / sO.Gelec. My version breaks that by caching.
How to fix it
Don't cache Gelec in bind_reactions(). Use bind_reactions() only to resolve the reaction-name strings to actual Reaction objects. Then recompute Gelec dynamically inside _calc_electronic_energy() / get_free_energy() on every call — matching the original behavior:
class ScalingState(State):
    def __init__(self, settings: ScalingStateSettings | dict):
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
What this means for your heatmap loop
- sC and sO (regular State instances): parallel_mkm.py sets states["sC"].Gelec = EC directly. State.get_free_energy() reads self.Gelec, so it picks up the new value each iteration. ✅
- Scaling states that depend on C_ads/O_ads: With the dynamic version above, get_free_energy() calls rxn.get_reaction_energy() which reads the current sC.Gelec / sO.Gelec, so they pick up the new descriptor values each iteration. ✅
The key is to not compute Gelec once in bind_reactions() — only resolve the string references there, and compute the energy on demand.