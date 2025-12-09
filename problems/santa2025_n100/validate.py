"""
Validator for Santa 2025 N=100 Tree Packing problem.
"""

import numpy as np
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.geometry import Polygon
from scipy.spatial.distance import pdist

getcontext().prec = 25
SCALE_FACTOR = Decimal('1e15')


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0'):
        self.center_x = Decimal(str(center_x))
        self.center_y = Decimal(str(center_y))
        self.angle = Decimal(str(angle))

        trunk_w, trunk_h = Decimal('0.15'), Decimal('0.2')
        base_w, mid_w, top_w = Decimal('0.7'), Decimal('0.4'), Decimal('0.25')
        tip_y, tier_1_y, tier_2_y, base_y = Decimal('0.8'), Decimal('0.5'), Decimal('0.25'), Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon([
            (Decimal('0.0') * SCALE_FACTOR, tip_y * SCALE_FACTOR),
            (top_w / 2 * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
            (top_w / 4 * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
            (mid_w / 2 * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
            (mid_w / 4 * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
            (base_w / 2 * SCALE_FACTOR, base_y * SCALE_FACTOR),
            (trunk_w / 2 * SCALE_FACTOR, base_y * SCALE_FACTOR),
            (trunk_w / 2 * SCALE_FACTOR, trunk_bottom_y * SCALE_FACTOR),
            (-trunk_w / 2 * SCALE_FACTOR, trunk_bottom_y * SCALE_FACTOR),
            (-trunk_w / 2 * SCALE_FACTOR, base_y * SCALE_FACTOR),
            (-base_w / 2 * SCALE_FACTOR, base_y * SCALE_FACTOR),
            (-mid_w / 4 * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
            (-mid_w / 2 * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
            (-top_w / 4 * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
            (-top_w / 2 * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
        ])
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * SCALE_FACTOR),
            yoff=float(self.center_y * SCALE_FACTOR)
        )


def validate(solution: np.ndarray) -> dict[str, float]:
    solution = np.asarray(solution, dtype=float)

    if solution.ndim != 2 or solution.shape != (100, 3):
        raise ValueError(f"Expected shape (100, 3), got {solution.shape}")

    if not np.all(np.isfinite(solution)):
        raise ValueError("Solution contains NaN or infinite values")

    x_coords, y_coords, angles = solution[:, 0], solution[:, 1], solution[:, 2]

    if x_coords.min() < -100 or x_coords.max() > 100:
        raise ValueError(f"x out of bounds: [{x_coords.min():.3f}, {x_coords.max():.3f}]")
    if y_coords.min() < -100 or y_coords.max() > 100:
        raise ValueError(f"y out of bounds: [{y_coords.min():.3f}, {y_coords.max():.3f}]")

    trees = [ChristmasTree(x_coords[i], y_coords[i], angles[i]) for i in range(100)]
    polygons = [t.polygon for t in trees]

    tree_index = STRtree(polygons)
    for i, poly in enumerate(polygons):
        for j in tree_index.query(poly):
            if i < j and poly.intersects(polygons[j]) and not poly.touches(polygons[j]):
                raise ValueError(
                    f"Trees {i} and {j} overlap: "
                    f"({x_coords[i]:.3f}, {y_coords[i]:.3f}, {angles[i]:.1f}) vs "
                    f"({x_coords[j]:.3f}, {y_coords[j]:.3f}, {angles[j]:.1f})"
                )

    bounds = unary_union(polygons).bounds
    side_length = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / float(SCALE_FACTOR)
    fitness = (side_length ** 2) / 100.0

    single_tree_area = 0.284
    packing_density = (single_tree_area * 100) / (side_length ** 2) if side_length > 0 else 0.0
    min_tree_spacing = float(pdist(solution[:, :2]).min()) if len(solution) > 1 else float('inf')

    return {
        "fitness": float(fitness),
        "is_valid": 1,
        "packing_density": packing_density,
        "min_tree_spacing": min_tree_spacing,
    }
