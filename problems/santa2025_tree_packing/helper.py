"""
Helper functions for Santa 2025 Christmas Tree Packing problem.

Provides collision detection utility for validating tree placements.
"""

from shapely.strtree import STRtree


def check_trees_overlap(tree_polygons: list) -> bool:
    """
    Check if any trees in the list overlap with each other.

    Args:
        tree_polygons: List of shapely Polygon objects representing trees

    Returns:
        True if any trees overlap, False if all trees are non-overlapping
    """
    if len(tree_polygons) <= 1:
        return False

    tree_index = STRtree(tree_polygons)

    for i, poly in enumerate(tree_polygons):
        possible_collisions = tree_index.query(poly)
        for j in possible_collisions:
            if i < j and poly.intersects(tree_polygons[j]):
                return True

    return False
