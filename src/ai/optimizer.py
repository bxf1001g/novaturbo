"""
NovaTurbo — Multi-Objective Optimizer (NSGA-II)

Optimizes engine design for multiple competing objectives:
- Maximize thrust
- Minimize fuel consumption (TSFC)
- Minimize weight
- Maximize thrust-to-weight ratio

Uses NSGA-II genetic algorithm to find the Pareto-optimal front
of non-dominated designs.

Reference: Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Dict
import time

from ..physics.brayton import FlightConditions, EngineInputs, solve_brayton_cycle
from .dataset import DesignSpaceBounds, evaluate_design

try:
    import pandas as pd
except ImportError:
    pd = None


@dataclass
class OptimizationConfig:
    """Configuration for NSGA-II optimizer."""
    # GA parameters
    population_size: int = 100
    n_generations: int = 200
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    mutation_scale: float = 0.1        # Gaussian mutation std (fraction of range)

    # Tournament selection
    tournament_size: int = 3

    # Objectives (minimize all — negate for maximization)
    objectives: List[str] = field(default_factory=lambda: [
        'thrust_N',            # Maximize (negated)
        'tsfc_kg_N_s',         # Minimize
        'total_mass_kg',       # Minimize
        'thrust_to_weight',    # Maximize (negated)
    ])
    maximize: List[bool] = field(default_factory=lambda: [
        True,   # thrust
        False,  # TSFC
        False,  # mass
        True,   # T/W
    ])

    # Constraints
    min_thrust_N: float = 30.0
    max_tsfc: float = 0.00005
    max_mass_kg: float = 3.0

    # Use surrogate model for fast evaluation?
    use_surrogate: bool = False
    surrogate_model_path: Optional[str] = None


@dataclass
class Individual:
    """A single design in the population."""
    genes: np.ndarray = field(default_factory=lambda: np.array([]))
    objectives: np.ndarray = field(default_factory=lambda: np.array([]))
    rank: int = 0
    crowding_distance: float = 0.0
    is_feasible: bool = True
    performance: dict = field(default_factory=dict)


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Check if solution a dominates solution b (all objectives minimized)."""
    return np.all(a <= b) and np.any(a < b)


def _fast_non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """NSGA-II fast non-dominated sorting."""
    n = len(objectives)
    domination_count = np.zeros(n, dtype=int)
    dominated_set = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(objectives[i], objectives[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif _dominates(objectives[j], objectives[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1

        if domination_count[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]  # Remove empty fronts


def _crowding_distance(objectives: np.ndarray, front: List[int]) -> np.ndarray:
    """Calculate crowding distance for a Pareto front."""
    n = len(front)
    distances = np.zeros(n)

    if n <= 2:
        distances[:] = float('inf')
        return distances

    n_obj = objectives.shape[1]
    for m in range(n_obj):
        # Sort by objective m
        sorted_idx = np.argsort(objectives[front, m])
        distances[sorted_idx[0]] = float('inf')
        distances[sorted_idx[-1]] = float('inf')

        obj_range = objectives[front[sorted_idx[-1]], m] - objectives[front[sorted_idx[0]], m]
        if obj_range < 1e-10:
            continue

        for i in range(1, n - 1):
            distances[sorted_idx[i]] += (
                objectives[front[sorted_idx[i + 1]], m] -
                objectives[front[sorted_idx[i - 1]], m]
            ) / obj_range

    return distances


def _tournament_select(population: List[Individual], k: int = 3) -> Individual:
    """Tournament selection based on rank and crowding distance."""
    candidates = np.random.choice(len(population), size=min(k, len(population)), replace=False)
    best = candidates[0]
    for c in candidates[1:]:
        if (population[c].rank < population[best].rank or
            (population[c].rank == population[best].rank and
             population[c].crowding_distance > population[best].crowding_distance)):
            best = c
    return population[best]


def _sbx_crossover(parent1: np.ndarray, parent2: np.ndarray,
                   bounds_lo: np.ndarray, bounds_hi: np.ndarray,
                   eta: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover (SBX)."""
    n = len(parent1)
    child1, child2 = parent1.copy(), parent2.copy()

    for i in range(n):
        if np.random.random() < 0.5:
            if abs(parent1[i] - parent2[i]) > 1e-10:
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

                child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])

                child1[i] = np.clip(child1[i], bounds_lo[i], bounds_hi[i])
                child2[i] = np.clip(child2[i], bounds_lo[i], bounds_hi[i])

    return child1, child2


def _polynomial_mutation(genes: np.ndarray, bounds_lo: np.ndarray,
                         bounds_hi: np.ndarray, rate: float,
                         eta: float = 20.0) -> np.ndarray:
    """Polynomial mutation."""
    mutated = genes.copy()
    for i in range(len(genes)):
        if np.random.random() < rate:
            delta = bounds_hi[i] - bounds_lo[i]
            if delta < 1e-10:
                continue
            u = np.random.random()
            if u < 0.5:
                delta_q = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            mutated[i] = genes[i] + delta_q * delta
            mutated[i] = np.clip(mutated[i], bounds_lo[i], bounds_hi[i])
    return mutated


def optimize(config: Optional[OptimizationConfig] = None,
             bounds: Optional[DesignSpaceBounds] = None,
             verbose: bool = True) -> Tuple[List[Individual], dict]:
    """
    Run NSGA-II multi-objective optimization.

    Returns:
        (pareto_front, optimization_history)
    """
    if config is None:
        config = OptimizationConfig()
    if bounds is None:
        bounds = DesignSpaceBounds()

    # Extract bounds as arrays
    param_names = list(bounds.__dataclass_fields__.keys())
    bounds_lo = np.array([getattr(bounds, p)[0] for p in param_names])
    bounds_hi = np.array([getattr(bounds, p)[1] for p in param_names])
    n_params = len(param_names)

    if verbose:
        print(f"NovaTurbo NSGA-II Optimizer")
        print(f"  Population: {config.population_size}")
        print(f"  Generations: {config.n_generations}")
        print(f"  Parameters: {n_params}")
        print(f"  Objectives: {config.objectives}")

    # Initialize population
    population = []
    for _ in range(config.population_size):
        genes = bounds_lo + np.random.random(n_params) * (bounds_hi - bounds_lo)
        ind = Individual(genes=genes)
        population.append(ind)

    # Evaluate initial population
    _evaluate_population(population, param_names, config)

    history = {'best_objectives': [], 'pareto_size': [], 'feasible_count': []}

    t0 = time.time()
    for gen in range(config.n_generations):
        # Create offspring
        offspring = []
        while len(offspring) < config.population_size:
            p1 = _tournament_select(population, config.tournament_size)
            p2 = _tournament_select(population, config.tournament_size)

            if np.random.random() < config.crossover_rate:
                c1_genes, c2_genes = _sbx_crossover(
                    p1.genes, p2.genes, bounds_lo, bounds_hi)
            else:
                c1_genes, c2_genes = p1.genes.copy(), p2.genes.copy()

            c1_genes = _polynomial_mutation(c1_genes, bounds_lo, bounds_hi, config.mutation_rate)
            c2_genes = _polynomial_mutation(c2_genes, bounds_lo, bounds_hi, config.mutation_rate)

            offspring.append(Individual(genes=c1_genes))
            offspring.append(Individual(genes=c2_genes))

        offspring = offspring[:config.population_size]

        # Evaluate offspring
        _evaluate_population(offspring, param_names, config)

        # Combine parent + offspring
        combined = population + offspring

        # Extract objectives
        all_objectives = np.array([ind.objectives for ind in combined])

        # Non-dominated sorting
        fronts = _fast_non_dominated_sort(all_objectives)

        # Assign ranks
        for rank, front in enumerate(fronts):
            for idx in front:
                combined[idx].rank = rank

        # Select next generation
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= config.population_size:
                # Add entire front
                for idx in front:
                    new_population.append(combined[idx])
            else:
                # Partial front — use crowding distance
                cd = _crowding_distance(all_objectives, front)
                sorted_by_cd = np.argsort(-cd)  # Descending
                remaining = config.population_size - len(new_population)
                for i in range(remaining):
                    idx = front[sorted_by_cd[i]]
                    combined[idx].crowding_distance = cd[sorted_by_cd[i]]
                    new_population.append(combined[idx])
                break

        population = new_population

        # Track history
        pareto_front = [ind for ind in population if ind.rank == 0]
        feasible = [ind for ind in population if ind.is_feasible]
        history['pareto_size'].append(len(pareto_front))
        history['feasible_count'].append(len(feasible))

        if pareto_front:
            best_obj = np.array([ind.objectives for ind in pareto_front])
            history['best_objectives'].append(best_obj.mean(axis=0).tolist())

        if verbose and (gen + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  Gen {gen+1:4d}/{config.n_generations}  "
                  f"Pareto: {len(pareto_front):3d}  "
                  f"Feasible: {len(feasible):3d}  "
                  f"Time: {elapsed:.1f}s")

    # Final Pareto front
    pareto_front = sorted(
        [ind for ind in population if ind.rank == 0],
        key=lambda x: x.objectives[0]  # Sort by first objective
    )

    if verbose:
        print(f"\n  Optimization complete!")
        print(f"  Pareto front size: {len(pareto_front)}")
        print(f"  Total time: {time.time()-t0:.1f}s")

        if pareto_front:
            print(f"\n  Pareto Front Summary:")
            for i, obj_name in enumerate(config.objectives):
                vals = [ind.objectives[i] for ind in pareto_front]
                sign = -1 if config.maximize[i] else 1
                print(f"    {obj_name:25s}: {min(v*sign for v in vals):.4f} — "
                      f"{max(v*sign for v in vals):.4f}")

    return pareto_front, history


def _evaluate_population(population: List[Individual],
                         param_names: List[str],
                         config: OptimizationConfig):
    """Evaluate all individuals in the population."""
    for ind in population:
        # Convert genes to parameter dict
        params = {}
        for i, name in enumerate(param_names):
            params[name] = ind.genes[i]

        # Create a pandas-like object for evaluate_design
        if pd is not None:
            row = pd.Series(params)
        else:
            row = type('Row', (), params)()

        result = evaluate_design(row)
        ind.performance = result
        ind.is_feasible = result.get('is_valid', False)

        # Extract objectives (negate for maximization)
        objectives = []
        for i, obj_name in enumerate(config.objectives):
            val = result.get(obj_name, 0)
            if config.maximize[i]:
                val = -val  # Negate for minimization framework
            objectives.append(val)

        # Penalty for infeasible designs
        if not ind.is_feasible:
            objectives = [1e10] * len(objectives)

        ind.objectives = np.array(objectives)


def get_best_design(pareto_front: List[Individual],
                    priority: str = "thrust_to_weight") -> Optional[Individual]:
    """
    Select the best design from the Pareto front based on a priority metric.
    """
    if not pareto_front:
        return None

    feasible = [ind for ind in pareto_front if ind.is_feasible]
    if not feasible:
        return pareto_front[0]

    if priority == "thrust_to_weight":
        return max(feasible, key=lambda x: x.performance.get('thrust_to_weight', 0))
    elif priority == "thrust":
        return max(feasible, key=lambda x: x.performance.get('thrust_N', 0))
    elif priority == "efficiency":
        return min(feasible, key=lambda x: x.performance.get('tsfc_kg_N_s', float('inf')))
    elif priority == "lightweight":
        return min(feasible, key=lambda x: x.performance.get('total_mass_kg', float('inf')))
    else:
        return feasible[0]


if __name__ == "__main__":
    # Quick test with small population
    config = OptimizationConfig(population_size=20, n_generations=10)
    pareto, history = optimize(config, verbose=True)

    best = get_best_design(pareto, priority="thrust_to_weight")
    if best:
        print(f"\nBest design (T/W priority):")
        for k, v in best.performance.items():
            print(f"  {k}: {v}")
