"""
Mandelbrot Set Generator
Author : [ Kim Nielsen ]
Course : Numerical Scientific Computing 2026
"""

# ----------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import statistics
import math
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
def naive_mandelbrot(c:complex, max_iter:int) -> int:
    """ This function computes Mandelbrot algorithm: z_{n+1} = z_n^2 + c
    The function checks if complex number "c" diverges by iterating over "z_{n+1} = z_n^2 + c"
    At each iteration, the function checks if the magnitude of the new z_{n+1} value exceeds 2, which means that "c" is NOT in the Mandelbrot set and results in divergence.
    The function returns the amount of iterations required before divergence or max iterations if no divergence is detected.
    
    Parameters
    ----------
    c : complex
        Point in complex plane with real and imaginary part.
    
    Returns
    ----------
    int
        Returns n if it diverges, else max_iter
    """
    
    # Initialize z value before loop
    z = 0

    for n in range(max_iter):
        # compute: z_{n+1} = z_n^2 + c
        z = z**2 + c
        # Check if new z value exceeds 2
        if math.sqrt( z.real**2 + z.imag**2 ) > 2:
            return n
    
    # If we reach max_iter and z has still not diverged, we return max_iter
    return max_iter

def create_grid() -> list:
    """Create grid or complex plane of real and imaginary axes. Represent axes as lists.
    We create two lists -> one for real part and one for imaginary part.
    
    Returns
    ----------
    list
        list of two lists. 
        - The first list contains the real axis values (x)
        - The second list contains the imaginary axis values (y)
    """
    dx = (x_max - x_min) / (N - 1)
    dy = (y_max - y_min) / (N - 1)
    x  = [ x_min + i*dx for i in range(N)]
    y  = [ 1j * (y_min + i*dy) for i in range(N)]
    C  = [x,y] # Form complex numbers
    
    return C

def naive_compute_mandelbrot(C_grid:list) -> list:
    """Compute Mandelbrot set by iterating over all pixels and applying mandelbrot function on corresponding complex number.
    
    Parameters
    ----------
    C_grid : list
        list of two lists.
        - The first list contains the real axis values (x)
        - The second list contains the imaginary axis values (y)
    
    Returns
    ----------
    list
        list with N number of lists. Each list is also N long. 
        Each element of these lists are escape counts or max_iter depending on if the pixel did not diverge before reaching max_iter.
    """
    # Loop over all grid complex values and check for divergence with mandelbrot function
    for i in range(N):
        for j in range(N):
            c = C_grid[0][j] + C_grid[1][i] # Form complex number for current pixel
            mandelbrot_set[i][j] = naive_mandelbrot( c, max_iter )
    return mandelbrot_set

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
    naive_C_grid = create_grid()
    t, M = benchmark(naive_compute_mandelbrot, naive_C_grid)

    # Plot results
    plt.imshow( M, extent=(x_min, x_max, y_min, y_max), cmap='plasma' )
    plt.colorbar()
    plt.title('Mandelbrot plot')
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.show()
