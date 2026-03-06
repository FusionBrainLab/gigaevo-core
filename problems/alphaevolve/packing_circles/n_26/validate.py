import numpy as np


def validate_packing(centers, radii):
    n = centers.shape[0]
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            raise ValueError(
                f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square"
            )

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-6:
                raise ValueError(
                    f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i] + radii[j]}"
                )

    return True


def validate(data):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.shape != (26, 3):
        raise ValueError(f"Invalid shape: expected (26, 3), got {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError("Data has NaN or infinite values")
    
    centers = data[:, :2]  
    radii = data[:, 2]     
    
    if np.any(radii <= 0):
        raise ValueError("All radii must be positive")
    try:
        validate_packing(centers, radii)
        is_valid = 1
    except ValueError as e:
        raise ValueError(f"Invalid packing: {e}")

    fitness = float(np.sum(radii))

    return {
        "fitness": fitness,
        "is_valid": is_valid,
    }

