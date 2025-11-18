import itertools
from itertools import combinations

from helper import get_unit_triangle
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

def compute_layout_metrics(points: np.ndarray) -> dict:
    """
    Compute geometric and distribution metrics for a set of 11 2D points.
    Includes triangle quality, spacing, spread, and convex hull area.
    """
    assert points.shape == (11, 2), "Expected exactly 11 points in 2D."

    def triangle_area(p1, p2, p3):
        return 0.5 * abs(
            (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
        )

    def min_triangle_angle_deg(p1, p2, p3):
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)
        if a == 0 or b == 0 or c == 0:
            return 0.0
        angles = np.arccos(
            np.clip(
                [
                    (b**2 + c**2 - a**2) / (2 * b * c),
                    (a**2 + c**2 - b**2) / (2 * a * c),
                    (a**2 + b**2 - c**2) / (2 * a * b),
                ],
                -1,
                1,
            )
        )
        return np.degrees(np.min(angles))

    areas = []
    min_angles = []
    degenerate_count = 0
    degenerate_threshold = 1e-4

    for i, j, k in combinations(range(11), 3):
        p1, p2, p3 = points[i], points[j], points[k]
        area = triangle_area(p1, p2, p3)
        min_angle = min_triangle_angle_deg(p1, p2, p3)
        areas.append(area)
        min_angles.append(min_angle)
        if area < degenerate_threshold:
            degenerate_count += 1

    areas = np.array(areas)
    min_angles = np.array(min_angles)

    # Pairwise distances
    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    upper_dists = dists[np.triu_indices(11, k=1)]

    # Centroid and spread
    centroid = np.mean(points, axis=0)
    point_var = np.var(points, axis=0)

    # Convex hull area (in 2D, volume is area)
    hull = ConvexHull(points)
    convex_hull_area = hull.volume

    min_area, max_area = np.min(areas), np.max(areas)
    log_bins = np.geomspace(max(min_area, 1e-8), max_area, num=16)  # 8 bins
    hist, bin_edges = np.histogram(areas, bins=log_bins)

    low_bin_frac = np.sum(hist[:3]) / np.sum(hist)
    probs = hist / np.sum(hist)
    entropy = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
    mean_bin = np.average(np.arange(len(hist)), weights=hist)

    return {
        "mean_triangle_area": float(np.mean(areas)),
        "std_triangle_area": float(np.std(areas)),
        "min_triangle_angle_deg": float(np.min(min_angles)),
        "mean_min_triangle_angle_deg": float(np.mean(min_angles)),
        "degenerate_triangle_count": int(degenerate_count),
        "pairwise_distance_min": float(np.min(upper_dists)),
        "pairwise_distance_max": float(np.max(upper_dists)),
        "pairwise_distance_std": float(np.std(upper_dists)),
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "spread_x_var": float(point_var[0]),
        "spread_y_var": float(point_var[1]),
        "convex_hull_area": float(convex_hull_area),
        "triangle_area_low_bin_frac": float(low_bin_frac),
        "triangle_area_hist_entropy": float(entropy),
        "triangle_area_mean_bin": float(mean_bin),
    }


def validate(coordinates):
    coordinates = np.asarray(coordinates, dtype=float)

    # --- Input shape checks ---
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            f"Invalid shape for coordinates: expected (n, 2), got {coordinates.shape}"
        )
    if coordinates.shape[0] != 11:
        raise ValueError(f"Expected 11 points, got {coordinates.shape[0]}")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Some coordinates are NaN or infinite.")

    # --- Construct unit-area equilateral triangle ---
    A, B, C = get_unit_triangle()

    # --- Barycentric coordinate check (vectorized) ---
    def is_inside_triangle(points, a, b, c):
        # Compute barycentric coordinates
        v0 = c - a
        v1 = b - a
        v2 = points - a

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)

        d20 = np.einsum("ij,j->i", v2, v0)
        d21 = np.einsum("ij,j->i", v2, v1)

        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        return np.all((u >= -1e-12) & (v >= -1e-12) & (w >= -1e-12))

    if not is_inside_triangle(coordinates, A, B, C):
        raise ValueError("Some coordinates are outside the triangle.")

    dists = pdist(coordinates)
    min_dist = np.min(dists)

    if min_dist < 1e-6:
        raise ValueError("Some points are too close or overlapping.")

    # --- Vectorized triangle area computation for all (n choose 3) triplets ---
    n = coordinates.shape[0]
    idx = np.array(list(itertools.combinations(range(n), 3)))
    pts = coordinates[idx]  # shape: (N, 3, 2)

    a = pts[:, 0, :]
    b = pts[:, 1, :]
    c = pts[:, 2, :]

    areas = 0.5 * np.abs(
        (
            a[:, 0] * (b[:, 1] - c[:, 1])
            + b[:, 0] * (c[:, 1] - a[:, 1])
            + c[:, 0] * (a[:, 1] - b[:, 1])
        )
    )

    min_area = np.min(areas)
    metrics = compute_layout_metrics(coordinates)
    return {
        "fitness": float(min_area),
        "is_valid": 1,
        **metrics,
    }
