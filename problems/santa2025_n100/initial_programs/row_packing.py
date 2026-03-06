import numpy as np

np.random.seed(42)


# NON-EVOLVE-BLOCK-START
class ChristmasTree:
    """Christmas tree geometry."""

    trunk_w, trunk_h = 0.15, 0.2
    base_w, mid_w, top_w = 0.7, 0.4, 0.25
    tip_y, tier_1_y, tier_2_y, base_y = 0.8, 0.5, 0.25, 0.0
    trunk_bottom_y = -trunk_h

    BASE_VERTICES = np.array([
        [0.0, tip_y],
        [top_w / 2, tier_1_y], [top_w / 4, tier_1_y],
        [mid_w / 2, tier_2_y], [mid_w / 4, tier_2_y],
        [base_w / 2, base_y],
        [trunk_w / 2, base_y], [trunk_w / 2, trunk_bottom_y],
        [-trunk_w / 2, trunk_bottom_y], [-trunk_w / 2, base_y],
        [-base_w / 2, base_y],
        [-mid_w / 4, tier_2_y], [-mid_w / 2, tier_2_y],
        [-top_w / 4, tier_1_y], [-top_w / 2, tier_1_y],
    ], dtype=np.float64)

    def __init__(self, center_x=0.0, center_y=0.0, angle=0.0):
        self.x, self.y, self.angle = float(center_x), float(center_y), float(angle)
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        self.vertices = np.round((self.BASE_VERTICES @ rot.T) + np.array([center_x, center_y]), 12)

    def get_bounds(self):
        return self.vertices.min(axis=0), self.vertices.max(axis=0)
# NON-EVOLVE-BLOCK-END


def entrypoint() -> np.ndarray:
    n = 100
    best_score, best_positions = float("inf"), []

    for n_even in range(1, n + 1):
        for n_odd in [n_even, n_even - 1]:
            all_trees = []
            rest = n
            r = 0

            while rest > 0:
                m = min(rest, n_even if r % 2 == 0 else n_odd)
                rest -= m

                angle = 0 if r % 2 == 0 else 180
                x_offset = 0.0 if r % 2 == 0 else 0.35
                y = round(r // 2 * 1.0, 12) if r % 2 == 0 else round((0.8 + (r - 1) // 2 * 1.0), 12)

                for i in range(m):
                    x = round(0.7 * i + x_offset, 12)
                    all_trees.append([x, y, float(angle)])

                r += 1

            trees = [ChristmasTree(t[0], t[1], t[2]) for t in all_trees]
            all_verts = np.vstack([t.vertices for t in trees])
            min_xy = all_verts.min(axis=0)
            max_xy = all_verts.max(axis=0)

            score = max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]) ** 2
            if score < best_score:
                best_score = score
                best_positions = all_trees

    return np.array(best_positions, dtype=np.float64)
