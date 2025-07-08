import numpy as np
from helper import (
    compute_outer_hex_side_length,
    check_hexagon_overlap_two
)

def validate(data):
    """
    Validate that:
    - Centers and angles are valid arrays.
    - All hexagons (with rotation) are non-overlapping (touching is allowed).
    - The claimed layout includes exactly 11 hexagons.
    - The true enclosing hexagon side length is recomputed and returned as fitness.

    Raises:
        ValueError with detailed message on failure.

    Returns:
        Dict of validation metrics.
    """
    centers, angles = data
    unit_side = 1.0

    # --- Input checks ---
    centers = np.asarray(centers, dtype=float)
    angles = np.asarray(angles, dtype=float)

    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"Invalid shape for centers: expected (n, 2), got {centers.shape}")
    if angles.ndim != 1:
        raise ValueError(f"Angles must be a 1D array, got shape {angles.shape}")
    if centers.shape[0] != angles.shape[0]:
        raise ValueError(f"Mismatch: {centers.shape[0]} centers vs {angles.shape[0]} angles")
    if centers.shape[0] != 11:
        raise ValueError(f"Expected 11 hexagons, got {centers.shape[0]}")
    if not np.all(np.isfinite(centers)):
        raise ValueError("Some center coordinates are NaN or infinite.")
    if not np.all(np.isfinite(angles)):
        raise ValueError("Some rotation angles are NaN or infinite.")

    # --- Pairwise overlap check ---
    n = centers.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if check_hexagon_overlap_two(centers[i], angles[i], centers[j], angles[j]):
                raise ValueError(
                    f"Hexagon {i} and {j} overlap.\n"
                    f"  Centers: ({centers[i][0]:.3f}, {centers[i][1]:.3f}) vs ({centers[j][0]:.3f}, {centers[j][1]:.3f})\n"
                    f"  Angles: {angles[i]:.4f} vs {angles[j]:.4f}"
                )

    # --- Fitness: minimal enclosing hexagon size ---
    computed_outer = compute_outer_hex_side_length(centers, angles, unit_side)

    return {
        "fitness": -computed_outer,  # Negative = minimizing outer hexagon side
        "is_valid": 1,
    }