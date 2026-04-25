"""
Mandelbrot Set Generator
Author : [ Kim Nielsen ]
Course : Numerical Scientific Computing 2026
"""

# ----------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------
from dask import delayed
import dask
import numpy as np
import matplotlib.pyplot as plt
import statistics
import time
from typing import Callable, Any
from numba import njit
from dask.distributed import Client, LocalCluster

# ----------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------
N        = 1024  # Same width & height
X_MIN    = -2.0  # min point on real axis
X_MAX    = 1.0   # max point on real axis
Y_MIN    = -1.5  # min point on imaginary axis
Y_MAX    = 1.5   # max point on imaginary axis
max_iter = 100   # We set a max iteration as number not part of Mandelbrot set will result in diverging and will keep going forever.

n_workers = 4

@njit(cache=True)
def mandelbrot_pixel(c_real:float, c_imag:float, max_iter:int) -> int:
    """Finds iteration count for a single point in the complex grid.

    Parameters
    ----------
    c_real : float
        Real part of complex point
    c_imag : float
        Imaginary part of complex point
    max_iter : int
        The maximum number of iterations if trajectory does not escape.

    Returns
    ----------
    int 
        Returns n if it diverges, else max_iter
    """
    
    z_real = z_imag = 0.0

    for n in range(max_iter):
        z_real_sq = z_real * z_real
        z_imag_sq = z_imag * z_imag
        if (z_real_sq + z_imag_sq) > 4:
            return n
        z_imag = 2.0 * z_real * z_imag + c_imag
        z_real = z_real_sq - z_imag_sq + c_real
    return max_iter

@njit(cache=True)
def mandelbrot_chunk(row_start:int, row_end:int, N:int, x_min:float, x_max:float, y_min:float, y_max:float, max_iter:int) -> np.ndarray:
    """Computes a chunk of the Mandelbrot set

    Parameters
    ----------
    row_start : int
        Starting row
    row_end : int
        Ending row
    N : int
        Size of complex grid
    x_min : float
        Minimum point on real axis
    x_max : float
        Maximum point on real axis
    y_min : float
        Minimum point on imaginary axis
    y_max : float
        Maximum point on imaginary axis
    max_iter : int
        The maximum number of iterations if trajectory does not escape.

    Returns
    ----------
    np.ndarray 
        (row_end - row_start)xN numpy array
    """
    mandelbrot_set = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (x_max - x_min) / N  # Compute normalization factor for real axis
    dy = (y_max - y_min) / N  # Compute normalization factor for imaginary axis

    # Loop over all grid complex values and check for divergence with mandelbrot function
    for r in range(row_end - row_start):
        c_imag = y_min + (r + row_start) * dy
        for column in range(N):
            mandelbrot_set[r, column] = mandelbrot_pixel(
                x_min + column * dx, c_imag, max_iter
            )
    
    return mandelbrot_set

def mandelbrot_dask(N:int, x_min:float, x_max:float, y_min:float, y_max:float, max_iter:int=100, n_chunks:int=32) -> np.ndarray:
    """Computes pixel coordinates from index + bounds. We take no arrays as input.
    
    Parameters
    ----------
    N : int
        Size of resolution
    x_min : float
        Minimum point on real axis
    x_max : float
        Maximum point on real axis
    y_min : float
        Minimum point on imaginary axis
    y_max : float
        Maximum point on imaginary axis
    max_iter : int
        The maximum number of iterations if trajectory does not escape.
    n_chunks : int
        Number of chunks to divide complex grid
    
    Returns
    ----------
    np.ndarray 
        2D array of shape (N,N) containing iteration counts
    """

    chunk_size = max(1, N // n_chunks)
    print(f'N: {N}, n_chunks: {n_chunks}, Chunk sz: {chunk_size}')
    tasks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(
            delayed(mandelbrot_chunk)(
                row, row_end, N, x_min, x_max, y_min, y_max, max_iter
            )
        )
        row = row_end
    parts = dask.compute(*tasks)
    return np.vstack(parts)

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
    cluster = LocalCluster(n_workers=2, threads_per_worker=1)
    client = Client(cluster)

    t, M = benchmark(mandelbrot_dask, N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter )

    # Terminate
    client.close()
    cluster.close()

    # Plot Mandelbrot
    plt.imshow( M, extent=(X_MIN, X_MAX, Y_MIN, Y_MAX), cmap='plasma' )
    plt.colorbar()
    plt.title('Mandelbrot plot')
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.show()
