from datetime import datetime
import json
import os
import time
import traceback

import numpy as np

# CONSTANTS
N_POINTS = 600
DIMENSION = 11
ALL_CONFIGS_FILE = (
    f"solutions/spherical_code_baseline/all_configs_{N_POINTS}_{DIMENSION}.jsonl"
)
BEST_CONFIG_FILE = (
    f"solutions/spherical_code_baseline/best_config_{N_POINTS}_{DIMENSION}.json"
)


def validate_spherical_code(points):
    """
    Validates the spherical code configuration.

    CRITICAL:
    1. Checks array shapes.
    2. CHECKS SPHERICAL CONSTRAINT (||x|| = 1).
    3. Computes fitness (Negative Maximum Pairwise Cosine Similarity).

    Args:
        points: np.ndarray of shape (N_POINTS, DIMENSION)

    Returns:
        dict: Metrics including 'fitness', 'max_cosine', and 'is_valid'.

    Raises:
        ValueError: If shapes are wrong or constraint is violated.
    """
    TOLERANCE = 1e-12

    # --- 1. Shape & Type Checks ---
    try:
        points = np.asarray(points, dtype=float)
    except Exception:
        raise ValueError("Points could not be converted to numpy array.")

    if points.ndim != 2:
        raise ValueError(
            f"Invalid shape: expected ({N_POINTS}, {DIMENSION}), got {points.shape}"
        )

    if points.shape[0] != N_POINTS or points.shape[1] != DIMENSION:
        raise ValueError(
            f"Shape mismatch: expected ({N_POINTS}, {DIMENSION}), got {points.shape}"
        )

    if not np.all(np.isfinite(points)):
        raise ValueError("Some coordinates are NaN or infinite.")

    # --- 2. Spherical Constraint Check (CRITICAL) ---
    # Every point must lie strictly on the unit sphere
    norms = np.linalg.norm(points, axis=1)
    max_deviation = np.max(np.abs(norms - 1.0))

    if max_deviation > TOLERANCE:
        raise ValueError(
            f"CONSTRAINT VIOLATION: Points not on sphere.\n"
            f"  Max deviation from unit norm: {max_deviation:.2e} (Tol: {TOLERANCE})"
        )

    # --- 3. Objective: Maximize Negative Max Cosine ---
    # Compute all pairwise inner products
    dot_matrix = np.dot(points, points.T)

    # Fill the diagonal with -infinity so we don't measure a point against itself
    np.fill_diagonal(dot_matrix, -np.inf)

    max_cosine = np.max(dot_matrix)

    # Fitness is negative max_cosine.
    # Example: max_cosine = 0.5 -> fitness = -0.5
    # If max_cosine improves to 0.4 -> fitness = -0.4 (which is GREATER, so we maximize)
    fitness = -max_cosine

    return {
        "fitness": float(fitness),
        "max_cosine": float(max_cosine),
        "max_constraint_violation": float(max_deviation),
        "is_valid": True,
    }


def save_best_config(points, metrics, output_file: str):
    """Saves the configuration if it beats the existing record (higher fitness is better)."""
    if not output_file:
        return

    fitness = metrics.get("fitness")

    existing_fitness = -float("inf")
    if os.path.exists(output_file):
        try:
            with open(output_file) as f:
                data = json.load(f)
                existing_fitness = data.get("fitness", -float("inf"))
        except:
            pass

    # MAXIMIZATION logic: new fitness must be higher than existing
    if fitness > existing_fitness:
        output_data = {
            "fitness": float(fitness),
            "max_cosine": float(-fitness),  # Retrieve max cosine back
            "points": points.tolist(),  # Save the actual points for the best config
            "n_points": N_POINTS,
            "dimension": DIMENSION,
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(
            f"  >>> NEW RECORD SAVED! Max Cosine: {-fitness:.6f} (Fit: {fitness:.6f})"
        )


def log_all_config(step_label, points, metrics=None, error=None, delta_fitness=None):
    """Logs every configuration status to the all_configs file."""
    os.makedirs(os.path.dirname(ALL_CONFIGS_FILE), exist_ok=True)

    # Get current date and time in readable format
    datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build base entry (Excluding 'points' coordinate list to save disk space)
    base_entry = {"step": step_label, "datetime": datetime_str, "valid": error is None}

    # Extract fitness first if available
    fitness_value = None
    other_metrics = {}
    if metrics:
        clean_metrics = {
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in metrics.items()
        }
        fitness_value = clean_metrics.pop("fitness", None)
        other_metrics = clean_metrics

    # Build entry
    entry = {}
    if fitness_value is not None:
        entry["fitness"] = fitness_value

    if delta_fitness is not None:
        entry["delta_fitness"] = float(delta_fitness)

    entry.update(base_entry)
    entry.update(other_metrics)

    if error:
        entry["error"] = str(error)

    with open(ALL_CONFIGS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_pipeline(entrypoint_ans):
    """
    Executes the validation pipeline.
    Calls entrypoint and expects the numpy array directly.
    STRICT MODE: Any ValueError (constraint violation) raises immediately.
    """
    print(f"--- Running Optimization Function for N={N_POINTS}, D={DIMENSION} ---")

    output_file = BEST_CONFIG_FILE
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    best_global_config = None
    best_global_fitness = -float("inf")
    previous_fitness = None

    # ==========================================
    # Evaluation Loop (1 Run)
    # ==========================================
    print("\n=== Evaluating 1 Run ===")

    candidates = []

    for i in range(1):
        print(f"\n[Run {i + 1}] Validating entrypoint result...")

        # 1. Capture Entrypoint Answer
        t0 = time.time()
        try:
            result = entrypoint_ans
            points = np.asarray(result, dtype=float)
        except Exception as e:
            raise ValueError(f"Entrypoint execution or conversion failed: {e}")

        dt = time.time() - t0

        # 2. Validate & Log (Strict)
        try:
            metrics = validate_spherical_code(points)
            metrics["duration"] = dt

            fitness = metrics["fitness"]
            delta_fitness = (
                fitness - previous_fitness if previous_fitness is not None else None
            )

            # Log success
            log_all_config(
                f"Run.{i + 1}", points, metrics=metrics, delta_fitness=delta_fitness
            )

            print(
                f"  Result Run.{i + 1}: Max Cosine {-fitness:.6f} (Valid) | Time: {dt:.2f}s"
            )
            candidates.append((fitness, points))

            previous_fitness = fitness

            if fitness > best_global_fitness:
                best_global_fitness = fitness
                best_global_config = points
                save_best_config(points, metrics, output_file)

        except ValueError as e:
            # Log failure and RAISE (Do not forgive)
            log_all_config(f"Run.{i + 1}", points, error=str(e), delta_fitness=None)
            print(f"  Result Run.{i + 1}: INVALID ({e}) - STRICT MODE: ABORTING")
            raise e

    if not candidates:
        print("No valid configurations produced.")
        return -float("inf")

    # Summary
    candidates.sort(key=lambda x: x[0], reverse=True)
    winner_fitness, _ = candidates[0]
    print(f"\n>>> Best Result: Max Cosine {-winner_fitness:.6f}")

    return best_global_fitness


def validate(entrypoint_ans):
    """
    Main entry point called by the testing framework.
    Args:
        entrypoint_ans: The directly returned result of `entrypoint(n, d, seed)`.
    """
    try:
        final_fitness = run_pipeline(entrypoint_ans)

        return {
            "fitness": final_fitness,
            "is_valid": True if final_fitness > -float("inf") else False,
        }
    except Exception as e:
        # Catch errors here to return a structured failure
        traceback.print_exc()
        return {
            "fitness": -1000.0,
            "is_valid": False,
        }


if __name__ == "__main__":
    from initial_programs.gemini5 import entrypoint

    results = validate(entrypoint(n=N_POINTS, d=DIMENSION, seed=0))
    print(f"\nFinal Validation Results: {results}")
