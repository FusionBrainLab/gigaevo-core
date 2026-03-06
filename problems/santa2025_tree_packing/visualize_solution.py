"""
Visualization script for Christmas Tree Packing solutions.

Usage:
    python visualize_solution.py --solution path/to/solution.csv --output output_dir/ --configs 1,5,10,20,50,100,200

This script reads a solution DataFrame and generates PNG visualizations showing
tree placements and bounding squares for specified configurations.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from pathlib import Path

getcontext().prec = 25
scale_factor = Decimal('1e18')


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

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


def plot_configuration(df_config, num_trees, output_path=None):
    """
    Plots a single configuration of trees and saves to PNG.

    Args:
        df_config: DataFrame with columns ['tree_idx', 'x', 'y', 'deg'] for one n value
        num_trees: Number of trees in this configuration
        output_path: Path to save the PNG file (if None, displays instead)
    """
    # Create tree objects from DataFrame
    placed_trees = []
    for _, row in df_config.iterrows():
        tree = ChristmasTree(
            center_x=str(row['x']),
            center_y=str(row['y']),
            angle=str(row['deg'])
        )
        placed_trees.append(tree)

    # Compute bounding box
    all_coords = []
    for tree in placed_trees:
        coords = np.asarray(tree.polygon.exterior.xy).T
        all_coords.append(coords)

    all_coords = np.concatenate(all_coords)
    min_x, min_y = all_coords.min(axis=0)
    max_x, max_y = all_coords.max(axis=0)

    width = max_x - min_x
    height = max_y - min_y
    side_length = max(width, height) / 1e18

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.viridis([i / max(num_trees, 1) for i in range(num_trees)])

    # Get bounds for square positioning
    all_polygons = [t.polygon for t in placed_trees]
    bounds = unary_union(all_polygons).bounds

    # Plot each tree
    for i, tree in enumerate(placed_trees):
        # Rescale for plotting
        x_scaled, y_scaled = tree.polygon.exterior.xy
        x = [Decimal(str(val)) / scale_factor for val in x_scaled]
        y = [Decimal(str(val)) / scale_factor for val in y_scaled]
        ax.plot([float(xi) for xi in x], [float(yi) for yi in y], color=colors[i], linewidth=1.5)
        ax.fill([float(xi) for xi in x], [float(yi) for yi in y], alpha=0.5, color=colors[i])

    # Compute square position (centered on bounding box)
    minx = Decimal(str(bounds[0])) / scale_factor
    miny = Decimal(str(bounds[1])) / scale_factor
    maxx = Decimal(str(bounds[2])) / scale_factor
    maxy = Decimal(str(bounds[3])) / scale_factor

    width_dec = maxx - minx
    height_dec = maxy - miny

    square_x = minx if width_dec >= height_dec else minx - (Decimal(str(side_length)) - width_dec) / 2
    square_y = miny if height_dec >= width_dec else miny - (Decimal(str(side_length)) - height_dec) / 2

    # Draw bounding square
    bounding_square = Rectangle(
        (float(square_x), float(square_y)),
        float(side_length),
        float(side_length),
        fill=False,
        edgecolor='red',
        linewidth=2,
        linestyle='--',
    )
    ax.add_patch(bounding_square)

    # Set plot limits with padding
    padding = Decimal('0.5')
    ax.set_xlim(
        float(square_x - padding),
        float(square_x + Decimal(str(side_length)) + padding)
    )
    ax.set_ylim(
        float(square_y - padding),
        float(square_y + Decimal(str(side_length)) + padding)
    )
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    # Compute score for title
    score = Decimal(str(side_length)) ** 2 / Decimal(str(num_trees))
    plt.title(f'n={num_trees} trees | side={side_length:.6f} | score={float(score):.6f}', fontsize=14)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_solution(solution_df, output_dir, configs=None):
    """
    Generate visualizations for specified configurations.

    Args:
        solution_df: DataFrame with columns ['n', 'tree_idx', 'x', 'y', 'deg']
        output_dir: Directory to save PNG files
        configs: List of n values to visualize (if None, uses default set)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if configs is None:
        configs = [100]

    for n in configs:
        df_config = solution_df[solution_df['n'] == n]
        if df_config.empty:
            print(f"Warning: No data for n={n}, skipping...")
            continue

        output_path = output_dir / f"config_n{n:03d}.png"
        plot_configuration(df_config, n, output_path=output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Christmas Tree Packing solutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize default configurations
  python visualize_solution.py --solution solution.csv --output viz/

  # Visualize specific configurations
  python visualize_solution.py --solution solution.csv --output viz/ --configs 1,10,50,200

  # Load from initial program
  python -c "from initial_programs.row_packing import entrypoint; df = entrypoint(); df.to_csv('solution.csv', index=False)"
  python visualize_solution.py --solution solution.csv --output viz/
        """
    )
    parser.add_argument(
        '--solution',
        type=str,
        required=True,
        help='Path to solution CSV file with columns [n, tree_idx, x, y, deg]'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for PNG files'
    )
    parser.add_argument(
        '--configs',
        type=str,
        default=None,
        help='Comma-separated list of n values to visualize (e.g., "1,5,10,20,50,100,200")'
    )

    args = parser.parse_args()

    # Load solution
    solution_df = pd.read_csv(args.solution)
    print(f"Loaded solution with {len(solution_df)} rows")

    # Parse configs
    configs = None
    if args.configs:
        configs = [int(x.strip()) for x in args.configs.split(',')]
        print(f"Visualizing configurations: {configs}")

    # Generate visualizations
    visualize_solution(solution_df, args.output, configs=configs)
    print(f"\nVisualization complete! Files saved to: {args.output}")


if __name__ == '__main__':
    main()
