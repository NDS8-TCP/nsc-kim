"""
Mandelbrot Set Generator
Author : [ Kim Nielsen ]
Course : Numerical Scientific Computing 2026
"""

# ----------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import statistics
import time

# ----------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------
width          = 1024
height         = 1024
x_min          = -2.0
x_max          = 1.0
y_min          = -1.5
y_max          = 1.5
max_iteration  = 100 # We set a max iteration as number not part of Mandelbrot set will result in diverging and will keep going forever.
mandelbrot_set = [ [0 for _ in range(width)] for _ in range(height) ] # reset mandelbrot set

# ----------------------------------------------------------------------------------------------------
# Methods
# ----------------------------------------------------------------------------------------------------
def create_grid(width, height):
    '''
    Create 1024x1024 grid of complex numbers that correspond to pixels in output image.
    '''
    x = np.linspace(x_min, x_max, width)  # 1024 horizontal values
    y = np.linspace(y_min, y_max, height) # 1024 vertical values
    X, Y = np.meshgrid(x,y)               # 2D grids
    C = X + 1j * Y                        # Form complex grid

    print(f'Shape: {C.shape}')            # shape=(1024, 1024)
    print(f'Type: {C.dtype}')             # dtype=complex128
    return C

# ----------------------------------------------------------------------------------------------------
# Main region
# ----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    C_grid = create_grid(width, height)
