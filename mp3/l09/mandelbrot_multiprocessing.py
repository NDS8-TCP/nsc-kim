# ----------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------
from multiprocessing import Pool
import psutil
import numpy as np
from numba import njit

# ----------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------
N        = 1024  # Same width & height
x_min    = -2.0  # min point on real axis
x_max    = 1.0   # max point on real axis
y_min    = -1.5  # min point on imaginary axis
y_max    = 1.5   # max point on imaginary axis
max_iter = 100   # We set a max iteration as number not part of Mandelbrot set will result in diverging and will keep going forever.


# ----------------------------------------------------------------------------------------------------
# Methods
# ----------------------------------------------------------------------------------------------------
@njit
def mandelbrot_pixel(c_real, c_imag, max_iter) -> None:
    """
    Finds iteration count for a single point in the complex grid.
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


@njit
def mandelbrot_chunk(
    row_start, row_end, N, x_min, x_max, y_min, y_max, max_iter
) -> None:
    """
    Computes pixel coordinates from index + bounds. We take no arrays as input.
    Returns a (row_end - row_start)xN int32 numpy array
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


def _worker(args):
    return mandelbrot_chunk(*args)

# def mandelbrot_multiprocessing(N, x_min, x_max, y_min, y_max, max_iter):
#     cores = psutil.cpu_count(logical=False)
#     for n_workers in range(1, cores + 1):
#         chunk_size = max(1, N // n_workers)
#         chunks, row = [], 0
#         while row < N:
#             end = min(row + chunk_size, N)
#             chunks.append((row, end, N, x_min, x_max, y_min, y_max, max_iter))
#             row = end

#         with Pool(processes=n_workers) as pool:
#             pool.map(_worker, chunks)  # warm-up: Numba JIT in all workers
#             for _ in range(3):
#                 np.vstack(pool.map(_worker, chunks))
