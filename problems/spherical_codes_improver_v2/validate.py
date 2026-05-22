from datetime import datetime
import json
import os
import time

import numpy as np

# CONSTANTS
N_POINTS = 600
DIMENSION = 11
ALL_CONFIGS_FILE = f"solutions/spherical_code/all_configs_{N_POINTS}_{DIMENSION}.jsonl"
BEST_CONFIG_FILE = f"solutions/spherical_code/best_config_{N_POINTS}_{DIMENSION}.json"


def validate_spherical_code(points):
    """
    Validates the spherical code configuration.

    CRITICAL:
    1. Checks array shapes.
    2. CHECKS SPHERICAL CONSTRAINT (||x|| = 1).
    3. Computes fitness (Negative Maximum Pairwise Cosine Similarity).

    Args:
        points: np.ndarray of shape (N, d)

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
    # Using np.dot is highly optimized via BLAS for matrix multiplication
    dot_matrix = np.dot(points, points.T)

    # Fill the diagonal with -infinity so we don't measure a point against itself
    np.fill_diagonal(dot_matrix, -np.inf)

    max_cosine = np.max(dot_matrix)

    # Check for collocation (points are identical or dangerously close)
    if max_cosine > 1.0 - 1e-9:
        # While strictly valid geometrically, this is a terrible configuration.
        # We don't raise ValueError unless it breaks the pipeline,
        # but it will naturally be penalized as a horrible fitness score.
        pass

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
    """Saves the configuration if it beats the existing record (higher fitness is better).

    Concurrency-safe: all I/O is wrapped — parallel workers sharing this path may
    race-write/clobber the file but must NEVER propagate an OSError/JSONDecodeError
    out (would kill the parent evaluation since the outer try/except was removed).
    """
    if not output_file:
        return

    try:
        fitness = metrics.get("fitness")

        existing_fitness = -float("inf")
        if os.path.exists(output_file):
            try:
                with open(output_file) as f:
                    data = json.load(f)
                    existing_fitness = data.get("fitness", -float("inf"))
            except Exception:
                pass

        if fitness <= existing_fitness:
            return

        output_data = {
            "fitness": float(fitness),
            "max_cosine": float(-fitness),
            "points": points.tolist(),
            "n_points": N_POINTS,
            "dimension": DIMENSION,
        }
        tmp_file = f"{output_file}.tmp.{os.getpid()}"
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(tmp_file, "w") as f:
            json.dump(output_data, f, indent=2)
        os.replace(tmp_file, output_file)
        print(
            f"  >>> NEW RECORD SAVED! Max Cosine: {-fitness:.6f} (Fit: {fitness:.6f})"
        )
    except Exception as e:
        print(f"  >>> save_best_config swallowed I/O error: {type(e).__name__}: {e}")


def log_all_config(
    step_label, metrics=None, error=None, delta_fitness=None, intensity=None
):
    """Append a per-step log entry to ALL_CONFIGS_FILE. Concurrency-safe by swallow."""
    try:
        os.makedirs(os.path.dirname(ALL_CONFIGS_FILE), exist_ok=True)
        datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        base_entry = {
            "step": step_label,
            "datetime": datetime_str,
            "valid": error is None,
        }

        fitness_value = None
        other_metrics = {}
        if metrics:
            clean_metrics = {
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in metrics.items()
            }
            fitness_value = clean_metrics.pop("fitness", None)
            other_metrics = clean_metrics

        entry: dict = {}
        if fitness_value is not None:
            entry["fitness"] = fitness_value
        if delta_fitness is not None:
            entry["delta_fitness"] = float(delta_fitness)
        if intensity is not None:
            entry["intensity"] = float(intensity)
        entry.update(base_entry)
        entry.update(other_metrics)
        if error:
            entry["error"] = str(error)

        # A single <4KB append on POSIX is atomic; safe under multi-worker.
        with open(ALL_CONFIGS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        # I/O race or fs error: skip; don't kill the evaluation.
        pass


# def _compact_step_log(step: str, status: str, metrics=None, note: str | None = None):
#     """Build a token-lean log record suitable for mutation context artifacts."""
#     def _r(x: float) -> float:
#         return round(float(x), 5)

#     record: dict[str, object] = {"step": step, "status": status}
#     if metrics:
#         if "max_cosine" in metrics:
#             record["max_cosine"] = _r(metrics["max_cosine"])
#     if note:
#         record["note"] = note
#     return record


def _build_feedback_preview(data: dict) -> str:
    """Build compact feedback text without depending on gigaevo modules."""
    if "error" in data:
        return f"Validation crashed: {data.get('error', 'unknown error')}"

    summary = data.get("summary", {})
    final_best = data.get("final_best", {})
    baseline = data.get("baseline_max_cosine")
    success_steps = data.get("successful_step_indices", [])
    success_moments = data.get("success_moments", [])
    stage_a_candidates = data.get("stage_a_candidates", [])
    stage_b_schedule = data.get("stage_b_schedule", [])

    accepted = int(summary.get("stage_b_accepted", 0))
    rejected = int(summary.get("stage_b_rejected", 0))
    stage_a_total = int(summary.get("stage_a_total", 0))
    stage_a_valid = int(summary.get("stage_a_valid", 0))
    stage_a_exc = int(summary.get("stage_a_exceptions", 0))
    stage_b_exc = int(summary.get("stage_b_exceptions", 0))
    stage_b_regress = max(rejected - stage_b_exc, 0)
    total_stage_b = accepted + rejected
    acceptance_rate = (100.0 * accepted / total_stage_b) if total_stage_b else 0.0
    stage_b_last_error = data.get("stage_b_last_error")

    final_best_cos = final_best.get("max_cosine")
    stage_b_delta = None
    if isinstance(baseline, (int, float)) and isinstance(final_best_cos, (int, float)):
        stage_b_delta = float(final_best_cos) - float(baseline)

    lines = [
        f"### Validation Results: N={data.get('n_points', '?')}, D={data.get('dimension', '?')}"
    ]
    lines.append("**Status:** Valid (No constraint violations)")
    if isinstance(final_best_cos, (int, float)):
        lines.append(f"**Final Max Cosine:** {float(final_best_cos):.5f}")
    lines.append("")
    lines.append("#### Execution Analysis")
    if isinstance(baseline, (int, float)):
        lines.append(
            f"* **Stage A (improve only):** Achieved baseline max cosine of `{float(baseline):.5f}`."
        )
    if stage_a_total:
        lines.append(
            f"* **Stage A health:** {stage_a_valid}/{stage_a_total} candidates valid, {stage_a_exc} exceptions."
        )
    if stage_b_delta is not None and stage_b_delta < 0:
        lines.append(
            f"* **Stage B (perturb + improve):** Successfully reduced max cosine further by `{stage_b_delta:.5f}`."
        )
    elif stage_b_delta is not None:
        lines.append(
            "* **Stage B (perturb + improve):** Did not improve beyond the Stage A baseline."
        )
    lines.append(
        f"* **Perturbation Destructiveness:** `{acceptance_rate:.0f}%` acceptance rate ({accepted} accepted, {rejected} rejected = {stage_b_regress} regressions + {stage_b_exc} exceptions)."
    )
    if stage_b_exc > 0 and stage_b_last_error:
        lines.append(
            f"* **Stage B last exception:** `{stage_b_last_error}` — your perturb/improve raised on at least one input; if exception count is high, treat this as a code bug, not 'perturbation too destructive'."
        )
    if success_steps:
        step_str = ", ".join(str(int(x)) for x in success_steps)
        lines.append(
            f"* **Successful Intensity Scales:** Improvements were found at steps `{step_str}` of the refinement loop."
        )
    else:
        lines.append(
            "* **Successful Intensity Scales:** No strict improvements in Stage B."
        )

    if success_moments:
        lines.append("")
        lines.append("#### Success Moments (latest)")
        for item in success_moments[-5:]:
            step = item.get("step", "?")
            intensity = item.get("intensity", "?")
            cos = item.get("max_cosine")
            delta = item.get("delta_cosine")
            if isinstance(cos, (int, float)) and isinstance(delta, (int, float)):
                lines.append(
                    f"* `{step}` (intensity `{intensity}`): max cosine `{float(cos):.5f}` (delta `{float(delta):.5f}`)"
                )

    if stage_a_candidates:
        lines.append("")
        lines.append("#### Stage A Candidate Cosines")
        candidate_rows: list[str] = []
        for item in stage_a_candidates:
            step = item.get("step", "?")
            cos = item.get("max_cosine")
            if isinstance(cos, (int, float)):
                candidate_rows.append(f"{step}:{float(cos):.5f}")
        if candidate_rows:
            lines.append("* " + ", ".join(candidate_rows))

    if stage_b_schedule:
        lines.append("")
        lines.append("#### Stage B Cosine History (before -> intensity -> after)")
        for item in stage_b_schedule:
            step = item.get("step", "?")
            intensity = item.get("intensity", "?")
            cos_before = item.get("cosine_before")
            cos_after = item.get("cosine_after")
            if isinstance(cos_before, (int, float)) and isinstance(
                cos_after, (int, float)
            ):
                lines.append(
                    f"* `{step}`: {float(cos_before):.5f} -> {intensity} -> {float(cos_after):.5f}"
                )

    return "\n".join(lines)


def run_pipeline(improver_class):
    """
    Executes the pipeline: Generate -> Improve -> Perturb -> Improve.
    STRICT MODE: Any ValueError (constraint violation) raises immediately.
    """
    print(f"--- Initializing Improver for N={N_POINTS}, D={DIMENSION} ---")

    try:
        improver = improver_class(n=N_POINTS, d=DIMENSION, seed=42)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Improver class: {e}")

    best_global_config = None
    best_global_fitness = -float("inf")  # Standard Maximization Initialization
    previous_fitness = None
    artifact = {
        "task": "spherical_codes",
        "n_points": N_POINTS,
        "dimension": DIMENSION,
        "baseline_max_cosine": None,
        "stage_a_candidates": [],
        "stage_b_schedule": [],
        "summary": {
            "stage_a_total": 0,
            "stage_a_valid": 0,
            "stage_a_exceptions": 0,
            "stage_b_total": 0,
            "stage_b_accepted": 0,
            "stage_b_rejected": 0,
            "stage_b_exceptions": 0,
        },
        "stage_b_last_error": None,
        "successful_step_indices": [],
        "success_moments": [],
        "timing": {},
    }

    def append_success(record, limit=10):
        artifact["success_moments"].append(record)
        if len(artifact["success_moments"]) > limit:
            artifact["success_moments"] = artifact["success_moments"][-limit:]

    # ==========================================
    # STAGE A: Exploration & Selection
    # ==========================================
    print("\n=== STAGE A: Exploration (10 Candidates) ===")

    stage_a_candidates = []
    last_stage_a_error: str | None = None

    for i in range(10):
        artifact["summary"]["stage_a_total"] += 1
        print(f"\n[A.{i + 1}] Generating & Improving...")

        t0 = time.time()
        try:
            draft_config = improver.generate_config(seed=i)
            improved_config = improver.improve(draft_config, seed=i)
            metrics = validate_spherical_code(improved_config)
        except Exception as e:
            dt = time.time() - t0
            last_stage_a_error = f"{type(e).__name__}: {e}"
            artifact["summary"]["stage_a_exceptions"] += 1
            log_all_config(
                f"A.{i + 1}", error=str(e), delta_fitness=None, intensity=None
            )
            print(
                f"  Result A.{i + 1}: INVALID ({last_stage_a_error}) - SKIPPED ({dt:.2f}s)"
            )
            continue

        dt = time.time() - t0
        metrics["duration"] = dt
        artifact["summary"]["stage_a_valid"] += 1

        fitness = metrics["fitness"]
        max_cosine = metrics["max_cosine"]
        delta_fitness = (
            fitness - previous_fitness if previous_fitness is not None else None
        )

        log_all_config(
            f"A.{i + 1}",
            metrics=metrics,
            delta_fitness=delta_fitness,
            intensity=None,
        )

        print(
            f"  Result A.{i + 1}: Max Cos {max_cosine:.6f} | Fit: {fitness:.6f} (Valid) | Time: {dt:.2f}s"
        )
        stage_a_candidates.append((fitness, improved_config))
        artifact["stage_a_candidates"].append(
            {
                "step": f"A.{i + 1}",
                "max_cosine": round(float(max_cosine), 5),
            }
        )

        previous_fitness = fitness

        if fitness > best_global_fitness:
            best_global_fitness = fitness
            best_global_config = improved_config.copy()
            save_best_config(improved_config, metrics, BEST_CONFIG_FILE)

    if not stage_a_candidates:
        raise ValueError(
            "All 10 Stage A candidates raised exceptions before producing a valid (n, d) "
            f"unit-norm array. Last error from A.10: {last_stage_a_error}. "
            "Every cold-start call to `improve(generate_config(seed=i), seed=i)` failed; "
            "check shape `(n, d)`, the 1e-12 norm tolerance, and NaN/Inf in your returned array."
        )

    # Pick Winner (Maximizing fitness, so sort descending)
    stage_a_candidates.sort(key=lambda x: x[0], reverse=True)
    winner_fitness, winner_config = stage_a_candidates[0]
    artifact["baseline_max_cosine"] = round(float(-winner_fitness), 5)
    print(
        f"\n>>> Stage A Winner: Max Cos {-winner_fitness:.6f} (Fit: {winner_fitness:.6f})"
    )

    previous_fitness = winner_fitness

    # ==========================================
    # STAGE B: Refinement
    # ==========================================
    print("\n=== STAGE B: Refinement (Perturbation Loops) ===")

    current_config = winner_config.copy()
    # Intensities from high to low for simulated annealing-like structure
    intensities = np.geomspace(1.0, 0.0001, num=10)

    for loop_idx in range(5):
        print(f"\n--- Refinement Loop {loop_idx + 1}/5 ---")

        for step, intensity in enumerate(intensities):
            step_label = f"B.{loop_idx + 1}.{step}"
            artifact["summary"]["stage_b_total"] += 1
            print(
                f"  [{step_label}] Perturb (int={intensity:.1e}) -> Improve...",
                end="",
                flush=True,
            )

            try:
                # Pass a fresh copy so an LLM that mutates `points` in-place can't
                # corrupt the next iteration's `current_config` baseline.
                perturbed_config = improver.perturb(
                    current_config.copy(),
                    intensity=intensity,
                    seed=loop_idx * 100 + step,
                )
                refined_config = improver.improve(
                    perturbed_config, seed=10000 + loop_idx * 100 + step
                )
                metrics = validate_spherical_code(refined_config)
            except Exception as e:
                err_str = f"{type(e).__name__}: {e}"
                log_all_config(
                    step_label, error=str(e), delta_fitness=None, intensity=intensity
                )
                artifact["summary"]["stage_b_rejected"] += 1
                artifact["summary"]["stage_b_exceptions"] += 1
                artifact["stage_b_last_error"] = f"{step_label}: {err_str}"
                print(f" REJECTED ({err_str})")
                continue

            prior_fitness = previous_fitness
            new_fitness = metrics["fitness"]
            max_cosine = metrics["max_cosine"]
            prev_cosine = -float(prior_fitness)

            delta_fitness = (
                new_fitness - previous_fitness if previous_fitness is not None else None
            )

            log_all_config(
                step_label,
                metrics=metrics,
                delta_fitness=delta_fitness,
                intensity=intensity,
            )

            print(f" Max Cos: {max_cosine:.6f} (Fit: {new_fitness:.6f})", end="")
            artifact["stage_b_schedule"].append(
                {
                    "step": step_label,
                    "intensity": f"{float(intensity):.1e}",
                    "cosine_before": round(float(prev_cosine), 5),
                    "cosine_after": round(float(max_cosine), 5),
                }
            )

            # --- CONDITIONAL UPDATE LOGIC ---
            # Greedily accept improvements (Higher fitness is better)
            if new_fitness >= previous_fitness:
                current_config = refined_config.copy()
                previous_fitness = new_fitness
                artifact["summary"]["stage_b_accepted"] += 1
                print(" [ACCEPTED]", end="")
                if new_fitness > prior_fitness:
                    step_idx = int(step)
                    if step_idx not in artifact["successful_step_indices"]:
                        artifact["successful_step_indices"].append(step_idx)
                    prev_cos = -float(prior_fitness)
                    append_success(
                        {
                            "step": step_label,
                            "intensity": f"{float(intensity):.1e}",
                            "max_cosine": round(float(max_cosine), 5),
                            "delta_cosine": round(float(max_cosine - prev_cos), 5),
                        }
                    )
            else:
                artifact["summary"]["stage_b_rejected"] += 1
                print(" [REJECTED]", end="")

            # Check Global Record
            if new_fitness > best_global_fitness:
                diff = new_fitness - best_global_fitness
                print(f" [NEW RECORD +{diff:.6f}]")
                best_global_fitness = new_fitness
                best_global_config = refined_config.copy()
                save_best_config(refined_config, metrics, BEST_CONFIG_FILE)
            else:
                print("")

    artifact["successful_step_indices"].sort()
    artifact["timing"]["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    artifact["final_best"] = {
        "max_cosine": round(float(-best_global_fitness), 5),
    }
    artifact["feedback_preview"] = _build_feedback_preview(artifact)
    print(artifact["feedback_preview"])
    return best_global_fitness, artifact


def validate(Improver_class):
    """
    Main entry point called by the testing framework.

    Exceptions raised by `run_pipeline` are NOT caught here — the framework's
    validator stage formats them as crash feedback the LLM can read.
    Returning fitness=-1000 would make the framework silently drop the program
    from the archive without surfacing the error to the next iteration.
    """
    final_fitness, artifact = run_pipeline(Improver_class)
    return (
        {
            "fitness": final_fitness,
            "is_valid": True,
        },
        artifact,
    )
