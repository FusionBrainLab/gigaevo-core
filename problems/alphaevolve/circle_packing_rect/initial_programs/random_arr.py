from helper import *
import numpy as np


def entrypoint():
    """
    Random circle arrangement with constraints

    Returns:
        (21, 3) NumPy array of (x, y, radius) for 21 circles
    """
    num_circles = 21
    
    # Target rectangle: 0.85 x 1.15 (perimeter = 4, width + height = 2.0)
    rect_width = 0.85
    rect_height = 1.15
    
    margin = 0.05
    usable_width = rect_width - 2 * margin
    usable_height = rect_height - 2 * margin
    
    # Base radius - will be adjusted per circle
    base_radius = 0.03
    
    circles = []
    np.random.seed(42)  # For reproducibility
    
    def is_valid(new_circle, existing_circles):
        """Check if a new circle is valid (no overlaps, within bounds)"""
        x, y, r = new_circle
        
        # Check bounds
        if x - r < margin or x + r > rect_width - margin:
            return False
        if y - r < margin or y + r > rect_height - margin:
            return False
        
        # Check overlaps with existing circles
        for cx, cy, cr in existing_circles:
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < r + cr - 1e-6:
                return False
        
        return True
    
    max_attempts = 1000
    for _ in range(num_circles):
        placed = False
        attempts = 0
        
        while not placed and attempts < max_attempts:
            # Random position
            x = margin + np.random.rand() * usable_width
            y = margin + np.random.rand() * usable_height
            
            # Try different radii
            for radius_mult in [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1]:
                radius = base_radius * radius_mult
                candidate = [x, y, radius]
                
                if is_valid(candidate, circles):
                    circles.append(candidate)
                    placed = True
                    break
            
            attempts += 1
        
        # If we couldn't place, use a very small circle
        if not placed:
            x = margin + np.random.rand() * usable_width
            y = margin + np.random.rand() * usable_height
            circles.append([x, y, base_radius * 0.05])
    
    return np.array(circles)