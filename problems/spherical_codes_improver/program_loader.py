import re

import numpy as np


def process_spherical_code(input_path, output_path=None, d=8, values_per_line=32):
    """
    1. Loads raw data
    2. Reshapes to (n, d)
    3. Performs SOTA analysis
    4. Saves in specific 32-value-per-line format
    """

    print(f"--- Loading Data from {input_path} ---")
    with open(input_path) as f:
        raw_text = f.read()

    # Extract all numbers using regex (handles scientific notation and negatives)
    raw_values = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\b[-+]?\d+\b", raw_text)
    data = np.array([float(x) for x in raw_values])

    total_floats = len(data)
    if total_floats % d != 0:
        print(f"Warning: {total_floats} values not divisible by d={d}. Truncating.")
        data = data[: (total_floats // d) * d]
        total_floats = len(data)

    # SHAPE: (n, d) - This is the standard mathematical format
    n = total_floats // d
    points = data.reshape(n, d)
    print(f"Parsed configuration: n={n}, d={d}")

    # --- ANALYSIS SECTION ---
    # 1. Check Norms
    norms = np.linalg.norm(points, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-7):
        print("Note: Points not on unit sphere. Projecting to S^(d-1)...")
        points = points / norms[:, np.newaxis]

    # 2. Find Minimum Separation (SOTA Metric)
    # Dot product of every point with every other point
    inner_products = np.dot(points, points.T)
    # Mask self-comparison
    np.fill_diagonal(inner_products, -1.0)

    max_cos = np.max(inner_products)
    min_angle = np.degrees(np.arccos(np.clip(max_cos, -1.0, 1.0)))

    print(f"\n--- SOTA Analysis (d={d}) ---")
    print(f"Total Points (N):      {n}")
    print(f"Max Inner Product:     {max_cos:.12f}")
    print(f"Min Separation Angle:  {min_angle:.6f}°")

    # --- EXPORT SECTION ---
    # We flatten the points back to a 1D list to format them into lines of 32
    flat_data = points.flatten()

    # print(f"\n--- Exporting to {output_path} ---")
    # with open(output_path, 'w') as f:
    #     for i in range(0, len(flat_data), values_per_line):
    #         chunk = flat_data[i : i + values_per_line]
    #         # Convert floats to string with high precision
    #         line_str = ",".join([f"{x:.18f}" for x in chunk])
    #         f.write(line_str + "\n")

    print(f"Done. Successfully wrote {len(flat_data) // values_per_line} lines.")
    return points


# --- EXECUTION ---
# This assumes your file is named 'input.txt'
if __name__ == "__main__":
    # Create a dummy file for testing if you don't have one
    # with open('input.txt', 'w') as f: f.write("0.1, 0.2, ...")

    try:
        # This returns the (n, d) array you requested for further use
        points_nd = process_spherical_code("input.txt", "output_formatted.txt", d=8)

        # Example of how you can now use 'points_nd' (shape: 1576, 8)
        print(f"\nArray shape in memory: {points_nd.shape}")

    except FileNotFoundError:
        print(
            "Error: 'input.txt' not found. Please place your raw data in 'input.txt'."
        )
