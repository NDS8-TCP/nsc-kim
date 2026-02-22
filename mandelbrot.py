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
def mandelbrot(c, max_iteration) -> int:
    '''
    This function computes Mandelbrot algorithm: z_{n+1} = z_n^2 + c
    The function checks if complex number "c" diverges by iterating over "z_{n+1} = z_n^2 + c"
    At each iteration, the function checks if the magnitude of the new z_{n+1} value exceeds 2, which means that "c" is NOT in the Mandelbrot set and results in divergence.
    The function returns the amount of iterations required before divergence or max iterations if no divergence is detected.
    '''

    # Initialize z value before loop
    z = 0

    for n in range(max_iteration):
        # compute: z_{n+1} = z_n^2 + c
        z = z**2 + c
        # Check if new z value exceeds 2
        if math.sqrt( z.real**2 + z.imag**2 ) > 2:
            return n
    
    # If we reach max_iteration and z has still not diverged, we return max_iteration
    return max_iteration

def create_grid() -> list:
    '''
    Create grid or complex plane of complex numbers that correspond to pixels in output image.
    We create two lists -> one for real part and one for imaginary part.
    '''
    dx = (x_max - x_min) / (width - 1)
    dy = (y_max - y_min) / (height - 1)
    x  = [ x_min + i*dx for i in range(width)]
    y  = [ 1j * (y_min + i*dy) for i in range(height)]
    C  = [x,y] # Form complex numbers
    return C

def compute_mandelbrot(C_grid:list) -> None:
    '''
    Compute Mandelbrot set by iterating over all pixels and applying mandelbrot function on corresponding complex number.
    '''
    # Loop over all grid complex values and check for divergence with mandelbrot function
    for i in range(height):
        for j in range(width):
            c = C_grid[0][j] + C_grid[1][i] # Form complex number for current pixel
            mandelbrot_set[i][j] = mandelbrot( c, max_iteration )

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
    C_grid = create_grid()
    t, M = benchmark(compute_mandelbrot, C_grid)
    
    # Plot results
    plt.imshow( mandelbrot_set, extent=(x_min, x_max, y_min, y_max), cmap='plasma' )
    plt.colorbar()
    plt.title('Mandelbrot plot')
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.show()
