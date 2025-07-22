import numpy as np


def get_hexagon_vertices(
    center: np.ndarray, angle: float, side: float = 1.0
) -> np.ndarray:
    """Compute the 6 vertices of a rotated regular hexagon centered at `center`."""
    cx, cy = np.asarray(center).reshape(2)
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + angle
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.stack([x, y], axis=-1)


def compute_outer_hex_side_length(
    centers: np.ndarray, angles: np.ndarray, unit_side: float = 1.0
) -> float:
    """Compute the enclosing side length for a regular hexagon enclosing all inner unit hexagons."""
    all_vertices = np.concatenate(
        [get_hexagon_vertices(c, a, unit_side) for c, a in zip(centers, angles)]
    )
    projection_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    directions = np.stack(
        [np.cos(projection_angles), np.sin(projection_angles)], axis=-1
    )
    extents = np.max(np.abs(all_vertices @ directions.T), axis=0)
    return 2 * np.max(extents) / np.sqrt(3)


def _segments_intersect(p1, p2, q1, q2) -> bool:
    """Check if two line segments (p1-p2 and q1-q2) intersect — excludes touching endpoints."""

    def _orient(a, b, c):
        return np.sign(
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        )

    def _on_segment(a, b, c):
        return (
            min(a[0], b[0]) - 1e-8 <= c[0] <= max(a[0], b[0]) + 1e-8
            and min(a[1], b[1]) - 1e-8 <= c[1] <= max(a[1], b[1]) + 1e-8
        )

    o1, o2 = _orient(p1, p2, q1), _orient(p1, p2, q2)
    o3, o4 = _orient(q1, q2, p1), _orient(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True

    # Allow edge-touching: ignore intersection if it happens exactly at endpoint
    def is_touching_endpoint(p, q):
        return np.any(np.allclose(p, q, atol=1e-8))

    if (
        o1 == 0
        and _on_segment(p1, p2, q1)
        and not is_touching_endpoint(q1, p1)
        and not is_touching_endpoint(q1, p2)
    ):
        return True
    if (
        o2 == 0
        and _on_segment(p1, p2, q2)
        and not is_touching_endpoint(q2, p1)
        and not is_touching_endpoint(q2, p2)
    ):
        return True
    if (
        o3 == 0
        and _on_segment(q1, q2, p1)
        and not is_touching_endpoint(p1, q1)
        and not is_touching_endpoint(p1, q2)
    ):
        return True
    if (
        o4 == 0
        and _on_segment(q1, q2, p2)
        and not is_touching_endpoint(p2, q1)
        and not is_touching_endpoint(p2, q2)
    ):
        return True

    return False


def _polygons_intersect(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    """Check if two convex polygons intersect using edge-edge checks (no containment test)."""
    for i in range(6):
        a1, a2 = poly_a[i], poly_a[(i + 1) % 6]
        for j in range(6):
            b1, b2 = poly_b[j], poly_b[(j + 1) % 6]
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


def _aabb_overlap(v1: np.ndarray, v2: np.ndarray) -> bool:
    """Axis-aligned bounding box overlap check."""
    min1, max1 = np.min(v1, axis=0), np.max(v1, axis=0)
    min2, max2 = np.min(v2, axis=0), np.max(v2, axis=0)
    return np.all(max1 >= min2) and np.all(max2 >= min1)


def check_hexagon_overlap_two(
    center: np.ndarray,
    angle: float,
    other_center: np.ndarray,
    other_angle: float,
) -> bool:
    """Check if two hexagons overlap (true overlap — touching is allowed)."""
    v1 = get_hexagon_vertices(center, angle)
    v2 = get_hexagon_vertices(other_center, other_angle)

    if not _aabb_overlap(v1, v2):
        return False

    return _polygons_intersect(v1, v2)


def check_hexagon_overlap_many(centers: np.ndarray, angles: np.ndarray) -> bool:
    """Check if any pair of hexagons overlaps."""
    n = len(centers)
    hex_vertices = [
        get_hexagon_vertices(centers[i], angles[i]) for i in range(n)
    ]

    for i in range(n):
        vi = hex_vertices[i]
        for j in range(i + 1, n):
            vj = hex_vertices[j]
            if not _aabb_overlap(vi, vj):
                continue
            if _polygons_intersect(vi, vj):
                return True
    return False
