import numpy as np
import pandas as pd
import scipy
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon

np.random.seed(42)

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
    # NON-EVOLVE-BLOCK-END


def entrypoint() -> pd.DataFrame:
    rows = []

    for n in range(1, 201):
        best_score, best_trees = float("inf"), []
        for n_even in range(1, n + 1):
            for n_odd in [n_even, n_even - 1]:
                all_trees = []
                rest = n
                r = 0
                while rest > 0:
                    m = min(rest, n_even if r % 2 == 0 else n_odd)
                    rest -= m

                    angle = str(0 if r % 2 == 0 else 180)
                    x_offset = Decimal("0") if r % 2 == 0 else Decimal("0.7") / 2
                    y = r // 2 * Decimal("1.0") if r % 2 == 0 else (Decimal("0.8") + (r - 1) // 2 * Decimal("1.0"))
                    row_trees = [ChristmasTree(center_x=str(Decimal("0.7") * i + x_offset), center_y=str(y), angle=angle) for i in range(m)]
                    all_trees.extend(row_trees)

                    r += 1
                xys = np.concatenate([np.asarray(t.polygon.exterior.xy).T / 1e15 for t in all_trees])

                min_x, min_y = xys.min(axis=0)
                max_x, max_y = xys.max(axis=0)

                score = max(max_x - min_x, max_y - min_y) ** 2
                if score < best_score:
                    best_score = score
                    best_trees = all_trees

        for i, tree in enumerate(best_trees):
            rows.append({
                'n': n,
                'tree_idx': i,
                'x': float(tree.center_x),
                'y': float(tree.center_y),
                'deg': float(tree.angle)
            })

    return pd.DataFrame(rows)
