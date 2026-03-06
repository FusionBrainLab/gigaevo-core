import numpy as np
import json
import os
import time
from datetime import datetime

# Helper imports must exist in the environment
from helper import check_hexagon_overlap_two, compute_outer_hex_side_length

HEX_NUM = 11
ALL_CONFIGS_FILE = f"solutions/hexn_perturb/all_configs_{HEX_NUM}.jsonl"

def validate_hexagons(data):
    """
    Validates the hexagon configuration.
    
    CRITICAL: 
    1. Checks array shapes.
    2. CHECKS OVERLAPS. If ANY overlap is found, Raises ValueError IMMEDIATELY.
    3. Computes enclosing boundary size (fitness).

    Args:
        data: Tuple (centers, angles)

    Returns:
        dict: Metrics including 'fitness' and 'is_valid'.
    
    Raises:
        ValueError: If shapes are wrong or ANY overlap is detected.
    """
    centers, angles = data
    unit_side = 1.0

    # --- 1. Shape & Type Checks ---
    try:
        centers = np.asarray(centers, dtype=float)
        angles = np.asarray(angles, dtype=float)
    except Exception:
        raise ValueError("Centers or Angles could not be converted to numpy arrays.")

    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"Invalid shape for centers: expected ({HEX_NUM}, 2), got {centers.shape}")
    
    if angles.ndim != 1:
        raise ValueError(f"Angles must be a 1D array, got shape {angles.shape}")
    
    if centers.shape[0] != angles.shape[0]:
        raise ValueError(f"Mismatch: {centers.shape[0]} centers vs {angles.shape[0]} angles")
    
    if centers.shape[0] != HEX_NUM:
        raise ValueError(f"Expected {HEX_NUM} hexagons, got {centers.shape[0]}")
    
    if not np.all(np.isfinite(centers)):
        raise ValueError("Some center coordinates are NaN or infinite.")
    
    if not np.all(np.isfinite(angles)):
        raise ValueError("Some rotation angles are NaN or infinite.")

    # --- 2. Pairwise Overlap Check (CRITICAL) ---
    n = centers.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if check_hexagon_overlap_two(centers[i], angles[i], centers[j], angles[j]):
                # STOP IMMEDIATELY
                raise ValueError(
                    f"OVERLAP DETECTED: Hexagon {i} and {j} overlap.\n"
                    f"  Dist: {np.linalg.norm(centers[i]-centers[j]):.3f}"
                )

    # --- 3. Fitness: Minimal Enclosing Hexagon Size ---
    computed_outer = compute_outer_hex_side_length(centers, angles, unit_side)

    return {
        "fitness": -computed_outer,  # Maximize negative size
        "size": computed_outer,
        "is_valid": True,
    }

def save_best_config(configuration, metrics, output_file: str):
    """Saves the configuration if it beats the existing record."""
    if not output_file:
        return

    centers, angles = configuration
    fitness = metrics.get("fitness")
    
    existing_fitness = -float('inf')
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                data = json.load(f)
                existing_fitness = data.get("fitness", -float('inf'))
        except:
            pass
    
    if fitness > existing_fitness:
        output_data = {
            "fitness": float(fitness),
            "enclosing_side": float(-fitness),
            "centers": centers.tolist(),
            "angles": angles.tolist(),
            "hex_num": HEX_NUM
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

def log_all_config(step_label, configuration, metrics=None, error=None, 
                   delta_fitness=None, intensity=None):
    """Logs every configuration to the all_configs file.
    
    Order: fitness, delta_fitness, intensity, then other fields.
    """
    os.makedirs(os.path.dirname(ALL_CONFIGS_FILE), exist_ok=True)
    centers, angles = configuration
    
    # Get current date and time in readable format
    datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build base entry
    base_entry = {
        "step": step_label,
        "datetime": datetime_str,
        "centers": centers.tolist(),
        "angles": angles.tolist(),
        "valid": error is None
    }
    
    # Extract fitness first if available
    fitness_value = None
    other_metrics = {}
    if metrics:
        # Convert numpy types to python native for JSON
        clean_metrics = {k: (float(v) if isinstance(v, (np.floating, float)) else v) 
                         for k, v in metrics.items()}
        fitness_value = clean_metrics.pop("fitness", None)
        other_metrics = clean_metrics
    
    # Build entry with correct order: fitness, delta_fitness, intensity, then rest
    entry = {}
    if fitness_value is not None:
        entry["fitness"] = fitness_value
    
    if delta_fitness is not None:
        entry["delta_fitness"] = float(delta_fitness)
    
    if intensity is not None:
        entry["intensity"] = float(intensity)
    
    # Add other fields
    entry.update(base_entry)
    entry.update(other_metrics)
    
    if error:
        entry["error"] = str(error)
        
    with open(ALL_CONFIGS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def run_pipeline(improver_class):
    """
    Executes the Stage A (Explore) -> Stage B (Refine) pipeline.
    STRICT MODE: Any ValueError (overlap) raises immediately.
    """
    try:
        improver = improver_class(hex_num=HEX_NUM, seed=42)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Improver class: {e}")

    output_file = f"solutions/hexn_perturb/best_config_{HEX_NUM}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    best_global_config = None
    best_global_fitness = -float('inf')
    previous_fitness = None  # Track previous fitness for delta calculation

    # ==========================================
    # STAGE A: Exploration & Selection
    # ==========================================
    
    stage_a_candidates = []

    for i in range(10):
        
        # 1. Generate & Improve
        t0 = time.time()
        draft_config = improver.generate_config(seed=i)
        improved_config = improver.improve(draft_config, seed=i)
        dt = time.time() - t0

        # 2. Validate & Log (Strict)
        try:
            metrics = validate_hexagons(improved_config)
            metrics['duration'] = dt
            
            fitness = metrics['fitness']
            # Calculate delta_fitness (None for first config, otherwise difference)
            delta_fitness = fitness - previous_fitness if previous_fitness is not None else None
            
            # Log success (Stage A has no perturbation, so intensity is None)
            log_all_config(f"A.{i+1}", improved_config, metrics=metrics, 
                          delta_fitness=delta_fitness, intensity=None)
            
            stage_a_candidates.append((fitness, improved_config))
            
            # Update previous_fitness for next iteration
            previous_fitness = fitness

            if fitness > best_global_fitness:
                best_global_fitness = fitness
                best_global_config = improved_config
                save_best_config(improved_config, metrics, output_file)

        except ValueError as e:
            # Log failure and RAISE (Do not forgive)
            log_all_config(f"A.{i+1}", improved_config, error=str(e),
                          delta_fitness=None, intensity=None)
            raise e

    if not stage_a_candidates:
        # Should not be reached if we raise on error, but for safety
        return -float('inf')

    # Pick Winner
    stage_a_candidates.sort(key=lambda x: x[0], reverse=True)
    winner_fitness, winner_config = stage_a_candidates[0]
    
    # Set previous_fitness to winner's fitness for Stage B delta calculations
    previous_fitness = winner_fitness

    # ==========================================
    # STAGE B: Refinement
    # ==========================================
    
    current_config = winner_config
    intensities = [100.0, 50, 10.0, 5.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    for loop_idx in range(15):
        for step, intensity in enumerate(intensities):
            step_label = f"B.{loop_idx+1}.{step}"
            
            # 1. Perturb & Improve
            perturbed_config = improver.perturb(current_config, intensity=intensity, seed=loop_idx*100+step)
            refined_config = improver.improve(perturbed_config)
            
            # 2. Validate & Log (Strict)
            try:
                metrics = validate_hexagons(refined_config)
                new_fitness = metrics['fitness']
                
                # Calculate delta_fitness compared to previous configuration
                delta_fitness = new_fitness - previous_fitness if previous_fitness is not None else None
                
                # Log success (intensity is available since perturbation was applied)
                log_all_config(step_label, refined_config, metrics=metrics,
                              delta_fitness=delta_fitness, intensity=intensity)

                # --- CONDITIONAL UPDATE LOGIC ---
                if new_fitness >= previous_fitness:
                    current_config = refined_config
                    previous_fitness = new_fitness
                
                # Check Global Record
                if new_fitness > best_global_fitness:
                    diff = new_fitness - best_global_fitness
                    best_global_fitness = new_fitness
                    best_global_config = refined_config
                    save_best_config(refined_config, metrics, output_file)

            except ValueError as e:
                # Log failure and RAISE
                log_all_config(step_label, refined_config, error=str(e),
                              delta_fitness=None, intensity=intensity)
                raise e

    return best_global_fitness

def validate(Improver_class):
    """
    Main entry point called by the testing framework.
    """
    try:
        final_fitness = run_pipeline(Improver_class)
        
        return {
            "fitness": final_fitness,
            "is_valid": True if final_fitness > -float('inf') else False
        }
    except Exception as e:
        # We catch the raised errors here to return a structured failure
        # But the pipeline itself was unforgiving.
        import traceback
        traceback.print_exc()
        return {
            "fitness": -1000,
            "is_valid": False,
        }