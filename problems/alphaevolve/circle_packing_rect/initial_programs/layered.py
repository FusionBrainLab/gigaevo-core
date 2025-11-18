from helper import *
import numpy as np


def entrypoint():
    """
    Layered circle packing strategy

    Returns:
        (21, 3) NumPy array of (x, y, radius) for 21 circles
    """
    num_circles = 21
    
    # Target rectangle: 0.9 x 1.1 (perimeter = 4, width + height = 2.0)
    rect_width = 0.9
    rect_height = 1.1
    
    # Place circles in horizontal layers
    # Distribute 21 circles across layers (e.g., 7, 7, 7)
    layers = [7, 7, 7]
    
    margin = 0.05
    usable_width = rect_width - 2 * margin
    usable_height = rect_height - 2 * margin
    
    # Calculate layer spacing
    num_layers = len(layers)
    y_spacing = usable_height / (num_layers - 1) if num_layers > 1 else usable_height / 2
    
    circles = []
    for layer_idx, num_in_layer in enumerate(layers):
        y = margin + layer_idx * y_spacing
        
        # Calculate spacing and radius for this layer
        if num_in_layer > 1:
            x_spacing = usable_width / (num_in_layer - 1)
        else:
            x_spacing = usable_width
        
        # Radius should ensure no overlaps within layer and between layers
        min_dist = min(x_spacing, y_spacing) if layer_idx > 0 else x_spacing
        radius = min_dist * 0.4
        
        for i in range(num_in_layer):
            if len(circles) >= num_circles:
                break
            x = margin + i * x_spacing
            circles.append([x, y, radius])
        if len(circles) >= num_circles:
            break
    
    return np.array(circles)