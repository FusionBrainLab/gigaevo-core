"""
Validation function for: Heilbronn Triangle Problem - maximize minimum triangle area
"""

from helper import *


def validate(coordinates):
    """
    Validate the solution and compute fitness metrics.

    Args:
        coordinates: (11, 2) array from entrypoint()

    Returns:
        dict with metrics:
        - fitness: Main objective [area of the smallest triangle]
        - is_valid: Whether the program is valid (1 valid, 0 invalid)
    """
    # TODO: Unpack/process input data

    # TODO: Validate constraints from task_description.txt

    # TODO: Compute metrics
    fitness = 0.0  # Main objective [area of the smallest triangle]
    is_valid = 1   # Set to 0 if any constraint violated

    return {
        "fitness": fitness,
        "is_valid": is_valid,
    }