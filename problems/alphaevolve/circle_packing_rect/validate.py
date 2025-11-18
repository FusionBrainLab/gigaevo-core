"""
Validation function for: Circle Packing in Rectangle Problem - optimal arrangement of 21 circles inside a rectangle
"""

from helper import *
import numpy as np

BENCHMARK = 2.3658321334167627
NUM_CIRCLES = 21
TOL = 1e-6

def validate(circles):
    """
    Validate the solution and compute fitness metrics.

    Args:
        circles: (21, 3) array from entrypoint()

    Returns:
        dict with metrics:
        - radii_sum: Sum of the radii of all 21 circles (PRIMARY OBJECTIVE - maximize)
        - combined_score: Progress toward benchmark: radii_sum / 2.3658321334167627
        - is_valid: Whether the program is valid (1 valid, 0 invalid)
    """

    try:
        if not isinstance(circles, np.ndarray):
            circles = np.array(circles)

        if circles.shape != (NUM_CIRCLES, 3):
            raise ValueError(
                f"Invalid shapes: circles = {circles.shape}, expected {(21,3)}"
            )

        validate_packing_radii(circles[:, -1])
        validate_packing_overlap_wtol(circles, TOL)
        validate_packing_inside_rect_wtol(circles, TOL)

        radii_sum = np.sum(circles[:, -1])

        return {
            "radii_sum": float(radii_sum),
            "combined_score": float(radii_sum / BENCHMARK),
            "is_valid": 1
        }
    except Exception as e:
        return {"radii_sum": 0.0, "combined_score": 0.0, "is_valid": 0}