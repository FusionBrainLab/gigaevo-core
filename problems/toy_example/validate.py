import numpy as np


def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (
            x - r < -1e-6
            or x + r > 1 + 1e-6
            or y - r < -1e-6
            or y + r > 1 + 1e-6
        ):
            raise ValueError(
                f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square"
            )

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if (
                dist < radii[i] + radii[j] - 1e-6
            ):  # Allow for tiny numerical errors
                raise ValueError(
                    f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}"
                )

    return True


def validate(data):
    """
    Validate that the sum of radii is correct

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    centers, radii = data
    if not isinstance(centers, np.ndarray):
        centers = np.array(centers)
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii)

    # Validate solution
    valid = validate_packing(centers, radii)
    shape_valid = centers.shape == (9, 2) and radii.shape == (9,)
    if not shape_valid:
        raise ValueError(
            f"Shape is invalid: centers.shape={centers.shape}, radii.shape={radii.shape}"
        )
    is_not_nan_and_finite = np.all(np.isfinite(radii)) and np.all(
        np.isfinite(centers)
    )
    if not is_not_nan_and_finite:
        raise ValueError(
            f"Radii or centers have NaN or infinite values: radii={radii}, centers={centers}"
        )

    valid = shape_valid and is_not_nan_and_finite and valid
    if not valid:
        raise ValueError("Solution is invalid: valid={valid}")
    metrics = {}
    metrics["is_valid"] = float(valid)
    metrics["fitness"] = np.sum(radii)

    return metrics
