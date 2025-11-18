from helper import *
import numpy as np


def entrypoint():
    """
    Grid-based systematic circle placement

    Returns:
        (21, 3) NumPy array of (x, y, radius) for 21 circles
    """
    num_circles = 21
    # Use a 7x3 grid (7 columns, 3 rows)
    cols = 7
    rows = 3
    
    # Target rectangle: 0.8 x 1.2 (perimeter = 4, width + height = 2.0)
    rect_width = 0.8
    rect_height = 1.2
    
    # Calculate spacing to avoid overlaps
    # Leave some margin from edges
    margin = 0.05
    usable_width = rect_width - 2 * margin
    usable_height = rect_height - 2 * margin
    
    # Spacing between circle centers
    x_spacing = usable_width / (cols - 1) if cols > 1 else usable_width
    y_spacing = usable_height / (rows - 1) if rows > 1 else usable_height
    
    # Calculate radius to ensure no overlaps
    # Minimum distance between centers should be at least 2*radius
    min_center_dist = min(x_spacing, y_spacing) if cols > 1 and rows > 1 else min(usable_width, usable_height) / max(cols, rows)
    radius = min_center_dist * 0.4  # Conservative radius to avoid overlaps
    
    circles = []
    for row in range(rows):
        for col in range(cols):
            if len(circles) >= num_circles:
                break
            x = margin + col * x_spacing
            y = margin + row * y_spacing
            circles.append([x, y, radius])
        if len(circles) >= num_circles:
            break
    
    return np.array(circles)