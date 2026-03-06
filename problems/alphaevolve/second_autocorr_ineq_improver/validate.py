import numpy as np
import json
import os
import time
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
PROBLEM_NAME = "second_autocorr"
# Log file for metrics of every step (lightweight)
ALL_CONFIGS_FILE = f"solutions/{PROBLEM_NAME}/all_configs.jsonl"
# Storage for the actual best function array (heavy)
BEST_FILE = f"solutions/{PROBLEM_NAME}/best_f.json"

# ==========================================
# VALIDATION LOGIC
# ==========================================
def validate_f(f_values):
    """
    Validates function f and computes the autocorrelation constant C.
    
    Mathematical Definition:
        C(f) = ||f * f||_2^2 / ( ||f * f||_1 * ||f * f||_inf )
    
    Constraints:
    1. f(x) >= 0 (Non-negative)
    2. f is not trivial (not all zeros)
    3. N >= 1024 (Target resolution)
    """
    # --- 1. Basic Type & Shape Checks ---
    try:
        f_values = np.asarray(f_values, dtype=float)
    except Exception:
        raise ValueError("Input could not be converted to a numpy array of floats.")

    if f_values.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {f_values.shape}")

    if f_values.size == 0:
        raise ValueError("Array cannot be empty")
    
    # Strict Resolution Check defined in constraints
    if f_values.size < 100:
        raise ValueError(f"Resolution critically low (N={f_values.size} < 100).")

    if not np.all(np.isfinite(f_values)):
        raise ValueError("Some values are NaN or infinite")

    # --- 2. Constraint Checks ---
    # Non-negativity with tolerance
    if np.any(f_values < -1e-6):
        raise ValueError(
            f"Function must be non-negative. Minimum value: {np.min(f_values):.2e}"
        )

    # Non-triviality
    if np.all(np.abs(f_values) < 1e-12):
        raise ValueError("Function is identically zero (trivial solution)")

    # Ensure strictly non-negative for convolution
    f_nonneg = np.maximum(f_values, 0.0)
    
    # --- 3. Compute Autocorrelation (g = f * f) ---
    convolution = np.convolve(f_nonneg, f_nonneg, mode="full")

    # --- 4. Compute Metrics (L2, L1, L_inf) ---
    num_conv_points = len(convolution)
    # Domain [-0.5, 0.5] mapped to indices
    x_points = np.linspace(-0.5, 0.5, num_conv_points + 2)
    x_intervals = np.diff(x_points)
    # Pad convolution with 0s at ends
    y_points = np.concatenate(([0], convolution, [0]))

    # Vectorized L2 Norm Squared calculation (Trapezoidal-ish on convolution)
    y1 = y_points[:-1]
    y2 = y_points[1:]
    # Interval contribution: (h/3) * (y1^2 + y1*y2 + y2^2)
    interval_l2_squared = (x_intervals / 3.0) * (y1**2 + y1 * y2 + y2**2)
    l2_norm_squared = np.sum(interval_l2_squared)

    # L1 Norm (approximation)
    norm_1 = np.sum(np.abs(convolution)) / (len(convolution) + 1)
    
    # L_inf Norm
    norm_inf = np.max(np.abs(convolution))
    
    # Avoid division by zero
    if norm_1 == 0 or norm_inf == 0:
        raise ValueError("Convolution norms are zero.")

    # Objective C
    c2 = l2_norm_squared / (norm_1 * norm_inf)

    if not np.isfinite(c2) or c2 <= 0:
        raise ValueError(f"Invalid C value: {c2}")

    return {
        "fitness": c2,
        "l2_sq": l2_norm_squared,
        "l1": norm_1,
        "linf": norm_inf,
        "resolution": int(f_values.size),
        "is_valid": True,
    }

# ==========================================
# LOGGING & SAVING
# ==========================================
def update_global_record_if_better(f_values, metrics, output_file: str):
    """
    Checks if the current f_values beat the GLOBAL record on disk.
    If so, overwrites the file.
    Does NOT affect the return value of the current validation run.
    """
    current_fitness = metrics.get("fitness", -1.0)
    
    # Read existing record
    existing_fitness = -1.0
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                data = json.load(f)
                existing_fitness = data.get("fitness", -1.0)
        except (json.JSONDecodeError, IOError):
            pass
    
    # Update if better
    if current_fitness > existing_fitness:
        output_data = {
            "fitness": float(current_fitness),
            "f": f_values.tolist() if isinstance(f_values, np.ndarray) else list(f_values),
            "resolution": metrics.get("resolution"),
            "metrics": {k: v for k, v in metrics.items() if k != "fitness"}
        }
        
        # Atomic write
        temp_file = output_file + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(output_data, f)
        os.replace(temp_file, output_file)
        
        diff = current_fitness - existing_fitness

def log_metrics(step_label, metrics=None, error=None, delta_fitness=None, intensity=None):
    """
    Logs metadata to the JSONL file. 
    """
    os.makedirs(os.path.dirname(ALL_CONFIGS_FILE), exist_ok=True)
    
    datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    entry = {
        "step": step_label,
        "datetime": datetime_str,
        "valid": error is None
    }
    
    if metrics:
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)):
                entry[k] = v
    
    if delta_fitness is not None:
        entry["delta_fitness"] = float(delta_fitness)
        
    if intensity is not None:
        entry["intensity"] = float(intensity)
        
    if error:
        entry["error"] = str(error)
        
    with open(ALL_CONFIGS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ==========================================
# PIPELINE
# ==========================================
def run_pipeline(improver_class):
    """
    Executes the optimization pipeline.
    
    RETURNS:
        local_best_fitness: The best fitness found strictly by THIS run.
    """
    
    try:
        improver = improver_class(seed=42)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Improver class: {e}")

    os.makedirs(os.path.dirname(BEST_FILE), exist_ok=True)

    # We track the best result of THIS run specifically
    local_best_fitness = -1000
    local_best_f = None
    
    # Previous fitness in the chain (for hill climbing decisions)
    chain_prev_fitness = None 

    # ==========================================
    # STAGE A: Exploration
    # ==========================================
    stage_a_candidates = []

    for i in range(3):
        try:
            t0 = time.time()
            # STRICT: Always start fresh
            draft_f = improver.generate_config(initial_resolution=1024)
            improved_f = improver.improve(draft_f)
            dt = time.time() - t0

            # Validate
            metrics = validate_f(improved_f)
            metrics['duration'] = dt
            
            fitness = metrics['fitness']
            delta_fitness = fitness - chain_prev_fitness if chain_prev_fitness is not None else None
            
            # Log
            log_metrics(f"A.{i+1}", metrics=metrics, delta_fitness=delta_fitness)
            stage_a_candidates.append((fitness, improved_f))
            
            chain_prev_fitness = fitness 

            # Update Local Best
            if fitness > local_best_fitness:
                local_best_fitness = fitness
                local_best_f = improved_f
            
            # Update Global Record (Side Effect Only)
            update_global_record_if_better(improved_f, metrics, BEST_FILE)

        except ValueError as e:
            log_metrics(f"A.{i+1}", error=str(e))
        except Exception as e:
            log_metrics(f"A.{i+1}", error=str(e))

    # If Stage A failed completely, we cannot proceed
    if not stage_a_candidates:
        return -1.0
    
    # Pick Winner of this run to continue
    stage_a_candidates.sort(key=lambda x: x[0], reverse=True)
    winner_fitness, winner_f = stage_a_candidates[0]

    current_f = winner_f
    chain_prev_fitness = winner_fitness

    # ==========================================
    # STAGE B: Refinement
    # ==========================================
    
    intensities = [100.0, 10.0, 1.0, 0.1, 0.01, 0.001]
    
    for loop_idx in range(5):
        for step, intensity in enumerate(intensities):
            step_label = f"B.{loop_idx+1}.{step}"
            try:
                # 1. Perturb & Improve (Copy to ensure safety)
                perturbed_f = improver.perturb(current_f.copy(), intensity=intensity)
                refined_f = improver.improve(perturbed_f)
                
                # 2. Validate
                metrics = validate_f(refined_f)
                new_fitness = metrics['fitness']
                
                delta = new_fitness - chain_prev_fitness
                
                # 3. Log
                log_metrics(step_label, metrics=metrics, delta_fitness=delta, intensity=intensity)
                
                # --- Acceptance Logic (Greedy/Hill-Climb) ---
                if new_fitness >= chain_prev_fitness:
                    current_f = refined_f
                    chain_prev_fitness = new_fitness
                # Update Local Best
                if new_fitness > local_best_fitness:
                    local_best_fitness = new_fitness
                    local_best_f = refined_f
                
                # Update Global Record (Side Effect Only)
                update_global_record_if_better(refined_f, metrics, BEST_FILE)
                
            except ValueError as e:
                log_metrics(step_label, error=str(e), intensity=intensity)
            except Exception as e:
                import traceback
                traceback.print_exc()
                log_metrics(step_label, error=str(e), intensity=intensity)

    return local_best_fitness

def validate(entrypoint_class):
    """
    Main entry point.
    """
    try:
        final_fitness = run_pipeline(entrypoint_class)
        return {
            "fitness": final_fitness,
            "is_valid": True if final_fitness > 0 else False
        }
    except Exception as e:
        return {
            "fitness": -1000,
            "is_valid": False
        }