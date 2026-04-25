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
from typing import Callable, Any

# ----------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------
N              = 1024  # Same width & height
x_min          = -2.0  # min point on real axis
x_max          = 1.0   # max point on real axis
y_min          = -1.5  # min point on imaginary axis
y_max          = 1.5   # max point on imaginary axis
max_iter       = 100   # We set a max iteration as number not part of Mandelbrot set will result in diverging and will keep going forever.
mandelbrot_set = [ [0 for _ in range(N)] for _ in range(N) ] # reset mandelbrot set

# ----------------------------------------------------------------------------------------------------
# Methods
# ----------------------------------------------------------------------------------------------------
def numpy_mandelbrot(Z:np.ndarray, C:np.ndarray, M:np.ndarray, mask:np.ndarray) -> np.ndarray:
    """This function computes Mandelbrot algorithm: z_{n+1} = z_n^2 + c
    
    Parameters
    ----------
    Z : np.ndarray
        Numpy array of z values (initialized to zero)
    C : np.ndarray
        Numpy array representing the complex grid with all points.
    M : np.ndarray
        Numpy array containing escape counts for each pixel.
    mask : np.ndarray
        Boolean array, where each element indicates if points are still active (Have not diverged yet).

    Returns
    ----------
    np.ndarray
        A 2D array of the same shape as C.
        Each array element represents either the number of escape counts before divergence or max_iter, depending on if the associated pixel has diverged.
    
    """
    # compute: z_{n+1} = z_n^2 + c for all C grid values
    for k in range(max_iter):
        Z[mask]           = Z[mask]**2 + C[mask]    # We use boolean mask or array to only update points that have not yet diverged -> means all true values in boolean mask
        escaped           = np.abs(Z) > 2           # Check if magnitude of new z point exceeds 2 -> means point diverges. All points that exceed 2, is stored in escaped (also boolean array)
        escaped          &= mask                    # Skip points that diverged earlier
        
        M[escaped] = k                              # We store number of iterations k for points that have diverged the first time.
        mask      &= ~escaped                       # We negate the boolean values in escaped to remove them from the resulting mask, so we only get the non-diverging points which is assumed be in the mandelbrot set.
        
        # Break out of for loop if all booleans in mask are false.
        if not mask.any():
            break
    
    # Points that did not diverge must be in mandelbrot set
    M[mask] = max_iter
    return M

def numpy_compute_mandelbrot(C_grid:np.ndarray) -> np.ndarray:
    """Initialize numpy arrays before Mandelbrot computations.
    
    Parameters
    ----------
    C_grid : np.ndarray
        Numpy array representing the complex grid with all points.
    
    Returns
    ----------
    np.ndarray
        A 2D array of the same shape as C.
        Each array element represents either the number of escape counts before divergence or max_iter, depending on if the associated pixel has diverged.
    """
    Z              = np.zeros_like(C_grid)              # Initialize Z as same shape as C grid
    M              = np.zeros(C_grid.shape, dtype=int)  # Initialize M as same shape as C grid
    mask           = np.ones(C_grid.shape, dtype=bool)  # Boolean mask with same shape as C grid, used to check z>2
    return numpy_mandelbrot(Z, C_grid, M, mask)

def create_grid(N:int) -> np.ndarray:
    """Create NxN grid of complex numbers that correspond to pixels in output image.
    
    Parameters
    ----------
    N : int
        Size of the axes of the complex grid.
        Real and imaginary axis will both be N big.
    
    Returns
    ----------
    np.ndarray
        Numpy array representing the complex grid with all points.
    """
    x    = np.linspace(x_min, x_max, N) # N horizontal values
    y    = np.linspace(y_min, y_max, N) # N vertical values
    X, Y = np.meshgrid(x,y)                  # 2D grids
    C    = X + 1j * Y                        # Form complex grid

    print(f'Shape: {C.shape}')               # shape=(N, N)
    print(f'Type: {C.dtype}')                # dtype=complex128
    return C

def benchmark ( func:Callable[..., Any], *args , n_runs:int=3) -> tuple[float, Any]:
    """Time func , return median of n_runs.
    
    Parameters
    ----------
    func : Callable[..., Any]
        The function with arbitrary arguments to benchmark.
    args :
        Arguments for function to benchmark.
    n_runs : int
        Number of times to run function. Default is 3.
    
    Returns
    ----------
    tuple[float, Any]
        tuple with time median of n_runs and result of benchmarked function.
    """
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
# Example usage
# ----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    numpy_C_grid = create_grid(N)
    t, M = benchmark(numpy_compute_mandelbrot, numpy_C_grid)

    # Plot Mandelbrot
    plt.imshow( M, extent=(x_min, x_max, y_min, y_max), cmap='plasma' )
    plt.colorbar()
    plt.title('Mandelbrot plot')
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.show()
