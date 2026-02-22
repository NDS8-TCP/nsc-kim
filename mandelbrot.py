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
def mandelbrot(Z, C, M, mask):
    '''
    This function computes Mandelbrot algorithm: z_{n+1} = z_n^2 + c
    '''
    # compute: z_{n+1} = z_n^2 + c for all C grid values
    for k in range(max_iteration):
        Z[mask]           = Z[mask]**2 + C[mask]    # We use boolean mask or array to only update points that have not yet diverged -> means all true values in boolean mask
        escaped           = np.abs(Z) > 2           # Check if magnitude of new z point exceeds 2 -> means point diverges. All points that exceed 2, is stored in escaped (also boolean array)
        M[escaped & mask] = k                       # By ANDing values in escaped and current values in mask, we store number of iterations for points that have diverged the first time.
        mask             &= ~escaped                # We negate the boolean values in escaped to remove them from the resulting mask, so we only get the non-diverging points which is assumed be in the mandelbrot set.
    return M

def compute_mandelbrot(C_grid):
    '''
    Initialize numpy arrays before Mandelbrot computations. 
    '''
    Z              = np.zeros_like(C_grid)              # Initialize Z as same shape as C grid
    M              = np.zeros(C_grid.shape, dtype=int)  # Initialize M as same shape as C grid
    mask           = np.ones(C_grid.shape, dtype=bool)  # Boolean mask with same shape as C grid, used to check z>2
    return mandelbrot(Z, C_grid, M, mask)

def create_grid(width, height):
    '''
    Create 1024x1024 grid of complex numbers that correspond to pixels in output image.
    '''
    x    = np.linspace(x_min, x_max, width)  # 1024 horizontal values
    y    = np.linspace(y_min, y_max, height) # 1024 vertical values
    X, Y = np.meshgrid(x,y)                  # 2D grids
    C    = X + 1j * Y                        # Form complex grid

    print(f'Shape: {C.shape}')               # shape=(1024, 1024)
    print(f'Type: {C.dtype}')                # dtype=complex128
    return C

def benchmark ( func , * args , n_runs =3) -> tuple:
    """ Time func , return median of n_runs . """
    times = []
    for _ in range ( n_runs ):
        t0 = time . perf_counter ()
        result = func (* args )
        times . append ( time . perf_counter () - t0 )
    
    median_t = statistics . median ( times )
    
    print (f" Median : { median_t :.4f}s "
        f"( min ={ min( times ):.4f}, max ={ max( times ):.4f})")
    
    return median_t, result

# ----------------------------------------------------------------------------------------------------
# Main region
# ----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    C_grid = create_grid(width, height)
    t, M = benchmark(compute_mandelbrot, C_grid)
