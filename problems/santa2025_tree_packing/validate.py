"""
Validator for Christmas Tree Packing problem.

Validates submission format and computes fitness score with collision detection.
"""

import numpy as np
import pandas as pd
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.geometry import Polygon


getcontext().prec = 25
scale_factor = Decimal('1e18')


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    # NON-EVOLVE-BLOCK-START
    def __init__(self, center_x='0', center_y='0', angle='0'):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal('0.15')
        trunk_h = Decimal('0.2')
        base_w = Decimal('0.7')
        mid_w = Decimal('0.4')
        top_w = Decimal('0.25')
        tip_y = Decimal('0.8')
        tier_1_y = Decimal('0.5')
        tier_2_y = Decimal('0.25')
        base_y = Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                # Start at Tip
                (Decimal('0.0') * scale_factor, tip_y * scale_factor),
                # Right side - Top Tier
                (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
                # Right side - Middle Tier
                (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
                # Right side - Bottom Tier
                (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
                # Right Trunk
                (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
                # Left Trunk
                (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Bottom Tier
                (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Middle Tier
                (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
                # Left side - Top Tier
                (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated,
                                          xoff=float(self.center_x * scale_factor),
                                          yoff=float(self.center_y * scale_factor))


def validate(submission: pd.DataFrame) -> dict[str, float]:
    """
    Validate submission and compute fitness score.

    Args:
        submission: DataFrame with columns ['n', 'tree_idx', 'x', 'y', 'deg']

    Returns:
        Dictionary with metrics:
        - fitness: Sum of normalized square areas (lower is better)
        - is_valid: 1 if valid, 0 if invalid
        - sum_score_small: Sum of s_n^2/n for n=1 to 20
        - sum_score_medium: Sum of s_n^2/n for n=21 to 100
        - sum_score_large: Sum of s_n^2/n for n=101 to 200
        - avg_rotation_variance: Average rotation variance
        - avg_centroid_distance: Average distance from origin
    """
    if not isinstance(submission, pd.DataFrame):
        raise ValueError(
            f"Program result must be a pandas DataFrame, got {type(submission).__name__}"
        )

    required_cols = ['n', 'tree_idx', 'x', 'y', 'deg']
    missing_cols = [col for col in required_cols if col not in submission.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Expected columns: {required_cols}, found: {list(submission.columns)}"
        )

    submission = submission.copy()

    # Check total row count
    expected_total = sum(range(1, 201))  # 20,100
    if len(submission) != expected_total:
        raise ValueError(
            f"Expected {expected_total} total rows (sum of 1..200 trees), "
            f"got {len(submission)} rows"
        )

    # Check position bounds
    limit = 100
    x_vals = submission['x'].astype(float)
    y_vals = submission['y'].astype(float)

    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    bad_x = (x_min < -limit) or (x_max > limit)
    bad_y = (y_min < -limit) or (y_max > limit)

    if bad_x or bad_y:
        violations = []
        if x_min < -limit:
            violations.append(f"x_min={x_min:.3f} < -100")
        if x_max > limit:
            violations.append(f"x_max={x_max:.3f} > 100")
        if y_min < -limit:
            violations.append(f"y_min={y_min:.3f} < -100")
        if y_max > limit:
            violations.append(f"y_max={y_max:.3f} > 100")
        raise ValueError(
            f"Position bounds violated (must be in [-100, 100]): {', '.join(violations)}"
        )

    total_score = Decimal('0.0')
    sum_score_small = Decimal('0.0')
    sum_score_medium = Decimal('0.0')
    sum_score_large = Decimal('0.0')
    all_rotations = []
    all_centroids = []

    for n_trees, df_group in submission.groupby('n'):
        n_trees = int(n_trees)

        if len(df_group) != n_trees:
            raise ValueError(
                f"Configuration n={n_trees} has {len(df_group)} trees, expected {n_trees}. "
                f"Each configuration must have exactly n trees."
            )

        expected_indices = set(range(n_trees))
        actual_indices = set(df_group['tree_idx'].astype(int).values)
        if expected_indices != actual_indices:
            missing = expected_indices - actual_indices
            extra = actual_indices - expected_indices
            error_parts = []
            if missing:
                error_parts.append(f"missing indices {sorted(missing)}")
            if extra:
                error_parts.append(f"extra indices {sorted(extra)}")
            raise ValueError(
                f"Configuration n={n_trees} has incorrect tree_idx values: "
                f"{', '.join(error_parts)}. Expected indices: 0 to {n_trees-1}"
            )

        placed_trees = []
        for _, row in df_group.iterrows():
            placed_trees.append(ChristmasTree(row['x'], row['y'], row['deg']))

        all_polygons = [p.polygon for p in placed_trees]
        r_tree = STRtree(all_polygons)

        for i, poly in enumerate(all_polygons):
            indices = r_tree.query(poly)
            for index in indices:
                if index == i:
                    continue
                if poly.intersects(all_polygons[index]) and not poly.touches(all_polygons[index]):
                    tree_i = df_group.iloc[i]
                    tree_j = df_group.iloc[index]
                    raise ValueError(
                        f"Overlapping trees in configuration n={n_trees}: "
                        f"tree_idx {int(tree_i['tree_idx'])} at ({tree_i['x']:.3f}, {tree_i['y']:.3f}, deg={tree_i['deg']:.1f}) "
                        f"overlaps with tree_idx {int(tree_j['tree_idx'])} at ({tree_j['x']:.3f}, {tree_j['y']:.3f}, deg={tree_j['deg']:.1f})"
                    )

        bounds = unary_union(all_polygons).bounds
        side_length_scaled = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        group_score = (Decimal(side_length_scaled) ** 2) / (scale_factor**2) / Decimal(n_trees)
        total_score += group_score

        # Accumulate sums for behavior dimensions
        if n_trees <= 20:
            sum_score_small += group_score
        elif n_trees <= 100:
            sum_score_medium += group_score
        else:
            sum_score_large += group_score

        rotations = df_group['deg'].astype(float).values
        all_rotations.extend(rotations)

        x_vals = df_group['x'].astype(float).values
        y_vals = df_group['y'].astype(float).values
        centroids = np.sqrt(x_vals**2 + y_vals**2)
        all_centroids.extend(centroids)

    rotation_variances = []
    for n_trees, df_group in submission.groupby('n'):
        rotations = df_group['deg'].astype(float).values
        rotation_variances.append(np.var(rotations))
    avg_rotation_variance = float(np.mean(rotation_variances))

    avg_centroid_distance = float(np.mean(all_centroids))

    return {
        "fitness": float(total_score),
        "is_valid": 1,
        "sum_score_small": float(sum_score_small),
        "sum_score_medium": float(sum_score_medium),
        "sum_score_large": float(sum_score_large),
        "avg_rotation_variance": avg_rotation_variance,
        "avg_centroid_distance": avg_centroid_distance,
    }
