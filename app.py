"""
NovaTurbo AI — Micro Turbojet Engine Design System
Main entry point for the complete pipeline.

Usage:
    python app.py                          # Run full demo
    python app.py --generate 1000          # Generate 1000 design variants
    python app.py --train data/generated/dataset_1000.csv  # Train AI model
    python app.py --optimize               # Run optimization
    python app.py --design --thrust 100    # Design engine for 100N thrust
"""

import argparse
import sys
import os

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_geometry_demo():
    """Demonstrate the parametric geometry system."""
    print("\n" + "="*60)
    print("  PHASE 1: Parametric Engine Geometry")
    print("="*60)

    from src.geometry.assembly import EngineAssemblyParams, assemble_engine, print_engine_summary
    params = EngineAssemblyParams()
    asm = assemble_engine(params)
    print_engine_summary(asm, params)
    return asm


def run_brayton_demo():
    """Demonstrate the Brayton cycle solver."""
    print("\n" + "="*60)
    print("  PHASE 2: Brayton Cycle Analysis")
    print("="*60)

    from src.physics.brayton import FlightConditions, EngineInputs, solve_brayton_cycle, print_cycle_results
    flight = FlightConditions()
    engine = EngineInputs()
    results = solve_brayton_cycle(flight, engine)
    print_cycle_results(results)
    return results


def run_materials_demo():
    """Demonstrate material selection."""
    from src.physics.materials import load_materials, check_thermal_limits, recommend_material

    materials = load_materials()
    print("\n  Material Thermal Validation:")
    components = {
        'compressor': ('ti6al4v', 450),
        'combustor': ('inconel_718', 1100),
        'turbine': ('inconel_718', 1100),
        'nozzle': ('inconel_718', 800),
    }
    for comp, (mat, temp) in components.items():
        check = check_thermal_limits(mat, temp, materials)
        status_icon = '✓' if check['valid'] else '✗'
        print(f"    {status_icon} {comp:12s} | {materials[mat].name:20s} | "
              f"{temp}K → margin: {check['margin_continuous_K']:+.0f}K ({check['status']})")


def run_dataset_generation(n_samples: int = 1000):
    """Generate training dataset."""
    print("\n" + "="*60)
    print(f"  PHASE 3a: Dataset Generation ({n_samples} samples)")
    print("="*60)

    from src.ai.dataset import generate_dataset
    dataset = generate_dataset(n_samples=n_samples, verbose=True)
    return dataset


def run_training(csv_path: str):
    """Train the surrogate neural network."""
    print("\n" + "="*60)
    print("  PHASE 3b: Neural Network Training")
    print("="*60)

    from src.ai.surrogate import train_surrogate, SurrogateConfig
    config = SurrogateConfig(epochs=100, patience=15)
    model, history = train_surrogate(csv_path, config, verbose=True)
    return model, history


def run_optimization(n_pop: int = 50, n_gen: int = 50):
    """Run multi-objective optimization."""
    print("\n" + "="*60)
    print("  PHASE 3c: Multi-Objective Optimization (NSGA-II)")
    print("="*60)

    from src.ai.optimizer import optimize, OptimizationConfig, get_best_design
    config = OptimizationConfig(population_size=n_pop, n_generations=n_gen)
    pareto, history = optimize(config, verbose=True)

    # Show best designs
    for priority in ['thrust', 'efficiency', 'lightweight', 'thrust_to_weight']:
        best = get_best_design(pareto, priority=priority)
        if best and best.is_feasible:
            perf = best.performance
            print(f"\n  Best ({priority}):")
            print(f"    Thrust: {perf.get('thrust_N', 0):.1f} N "
                  f"({perf.get('thrust_N', 0)/9.81:.2f} kgf)")
            print(f"    TSFC: {perf.get('tsfc_kg_N_s', 0):.7f} kg/(N·s)")
            print(f"    Mass: {perf.get('total_mass_kg', 0)*1000:.0f} g")
            print(f"    T/W: {perf.get('thrust_to_weight', 0):.1f}")

    return pareto, history


def run_full_demo():
    """Run the complete NovaTurbo pipeline."""
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║   ███╗   ██╗ ██████╗ ██╗   ██╗ █████╗                ║
    ║   ████╗  ██║██╔═══██╗██║   ██║██╔══██╗               ║
    ║   ██╔██╗ ██║██║   ██║██║   ██║███████║               ║
    ║   ██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║               ║
    ║   ██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║               ║
    ║   ╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝               ║
    ║                                                       ║
    ║   T U R B O   A I                                     ║
    ║   Micro Turbojet Engine Design System                 ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)

    # Phase 1: Geometry
    asm = run_geometry_demo()

    # Phase 2: Physics
    results = run_brayton_demo()
    run_materials_demo()

    # Phase 3: AI Pipeline
    print("\n" + "="*60)
    print("  PHASE 3: AI Pipeline")
    print("="*60)

    # Generate small dataset for demo
    dataset = run_dataset_generation(n_samples=500)

    # Train surrogate model
    csv_path = "data/generated/dataset_500.csv"
    if os.path.exists(csv_path):
        model, history = run_training(csv_path)

    # Run optimization (small for demo)
    pareto, opt_history = run_optimization(n_pop=30, n_gen=30)

    print("\n" + "="*60)
    print("  Pipeline Complete!")
    print("="*60)
    print("""
    Next steps:
    1. Generate larger dataset:  python app.py --generate 10000
    2. Train better model:       python app.py --train data/generated/dataset_10000.csv
    3. Run full optimization:    python app.py --optimize --pop 100 --gen 200
    4. Export STL for printing:  (Phase 5 — coming soon)
    """)


def main():
    parser = argparse.ArgumentParser(description='NovaTurbo AI — Micro Turbojet Engine Design')
    parser.add_argument('--demo', action='store_true', help='Run full demo pipeline')
    parser.add_argument('--geometry', action='store_true', help='Show engine geometry')
    parser.add_argument('--brayton', action='store_true', help='Run Brayton cycle analysis')
    parser.add_argument('--generate', type=int, metavar='N', help='Generate N design variants')
    parser.add_argument('--train', type=str, metavar='CSV', help='Train surrogate on dataset CSV')
    parser.add_argument('--optimize', action='store_true', help='Run NSGA-II optimization')
    parser.add_argument('--pop', type=int, default=50, help='Optimizer population size')
    parser.add_argument('--gen', type=int, default=50, help='Optimizer generations')

    args = parser.parse_args()

    if args.geometry:
        run_geometry_demo()
    elif args.brayton:
        run_brayton_demo()
        run_materials_demo()
    elif args.generate:
        run_dataset_generation(args.generate)
    elif args.train:
        run_training(args.train)
    elif args.optimize:
        run_optimization(args.pop, args.gen)
    elif args.demo:
        run_full_demo()
    else:
        # Default: show geometry + physics
        run_geometry_demo()
        run_brayton_demo()
        run_materials_demo()
        print("\n  Use --demo for full pipeline, --help for all options.")


if __name__ == "__main__":
    main()
