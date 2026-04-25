import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import time
import statistics
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

# ----------------------------------------------------------------------------------------------------
# Kernel codes
# ----------------------------------------------------------------------------------------------------
KERNEL_F32 = """
__kernel void mandelbrot_f32(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;   // guard against over-launch

    float c_real = x_min + col * (x_max - x_min) / (float)N;
    float c_imag = y_min + row * (y_max - y_min) / (float)N;
    
    float zr = 0.0f, zi = 0.0f;
    int count = 0;
    while (count < max_iter && zr*zr + zi*zi <= 4.0f) {
        float tmp = zr*zr - zi*zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""

KERNEL_F64 = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void mandelbrot_f64(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= N || row >= N) return;

    double c_real = x_min + col * (x_max - x_min) / (double)N;
    double c_imag = y_min + row * (y_max - y_min) / (double)N;
    double z_real = 0.0, z_imag = 0.0;
    int count     = 0;
    
    while (count < max_iter && z_real*z_real + z_imag*z_imag <= 4.0)
    {
        double tmp = z_real*z_real - z_imag*z_imag + c_real;
        z_imag = 2.0 * z_real * z_imag + c_imag;
        z_real = tmp;
        count++;
    }
    result[row * N + col] = count;
}
"""

# ----------------------------------------------------------------------------------------------------
# Methods for setup & running kernels
# ----------------------------------------------------------------------------------------------------
def init_kernel(kernel_src:str) -> dict[str, Any]:
    """
    Parameters
    ----------
    kernel_src : str
        Source code for OpenCL kernel program in string format.
    
    Returns
    ----------
    dict
        Dictionary contaning: 
        - OpenCL context
        - command queue
        - compiled kernel program
        - host-side array to write result back to host
        - device buffer to store result in kernel execution
    """
    kernel_obj              = {}
    kernel_obj['ctx']       = cl.create_some_context(interactive=False)
    kernel_obj['queue']     = cl.CommandQueue(kernel_obj['ctx'])
    kernel_obj['prog']      = cl.Program(kernel_obj['ctx'], kernel_src).build()
    kernel_obj['image']     = np.zeros((N, N), dtype=np.int32)
    kernel_obj['image_dev'] = cl.Buffer(kernel_obj['ctx'], cl.mem_flags.WRITE_ONLY, kernel_obj['image'].nbytes)
    return kernel_obj

def run_mandelbrot_f32(kernel_obj:dict[str, Any], N:int, MAX_ITER:int, X_MIN:float, X_MAX:float, Y_MIN:float, Y_MAX:float) -> np.ndarray:
    """ Compute the Mandelbrot set with float32 inputs using a Float32 OpenCL kernel.
    The kernel is executed over a 2D grid of NxN size. Each work-item computes an iteration count for one pixel in this grid.

    Parameters
    ----------
    kernel_obj : dict[str, Any]
        Dictionary with OpenCL context, command queue, compiled float 32 kernel program and device buffer to store result.
    N : int
        Size of complex grid
    MAX_ITER : int
        The maximum number of iterations if trajectory does not escape.
    X_MIN : float
        Minimum point on real axis
    X_MAX : float
        Maximum point on real axis
    Y_MIN : float
        Minimum point on imaginary axis
    Y_MAX : float
        Maximum point on imaginary axis
    
    Returns
    ----------
    np.ndarray
        NxN int32 numpy array, where each element contains the escape count of a pixel before divergence or max_iter if no divergence occured.
    """
    kernel_obj['prog'].mandelbrot_f32(
        kernel_obj['queue'], (N, N), None,      # global size (N, N); let OpenCL pick local
        kernel_obj['image_dev'],
        np.float32(X_MIN), np.float32(X_MAX),
        np.float32(Y_MIN), np.float32(Y_MAX),
        np.int32(N), np.int32(MAX_ITER),
    )

    cl.enqueue_copy(kernel_obj['queue'], kernel_obj['image'], kernel_obj['image_dev']) # Write gpu result to host
    kernel_obj['queue'].finish()
    return kernel_obj['image']

def run_mandelbrot_f64(kernel_obj:dict[str, Any], N:int, MAX_ITER:int, X_MIN:float, X_MAX:float, Y_MIN:float, Y_MAX:float) -> np.ndarray:
    """ Compute the Mandelbrot set with float64 inputs using a Float64 OpenCL kernel.
    The kernel is executed over a 2D grid of NxN size. Each work-item computes an iteration count for one pixel in this grid.

    Parameters
    ----------
    kernel_obj : dict[str, Any]
        Dictionary with OpenCL context, command queue, compiled float 64 kernel program and device buffer to store result.
    N : int
        Size of complex grid
    MAX_ITER : int
        The maximum number of iterations if trajectory does not escape.
    X_MIN : float
        Minimum point on real axis
    X_MAX : float
        Maximum point on real axis
    Y_MIN : float
        Minimum point on imaginary axis
    Y_MAX : float
        Maximum point on imaginary axis
    
    Returns
    ----------
    np.ndarray
        NxN int32 numpy array, where each element contains the escape count of a pixel before divergence or max_iter if no divergence occured.
    """
    kernel_obj['prog'].mandelbrot_f64(
        kernel_obj['queue'], (N, N), None,      # global size (N, N); let OpenCL pick local
        kernel_obj['image_dev'],
        np.float64(X_MIN), np.float64(X_MAX),
        np.float64(Y_MIN), np.float64(Y_MAX),
        np.int32(N), np.int32(MAX_ITER),
    )

    cl.enqueue_copy(kernel_obj['queue'], kernel_obj['image'], kernel_obj['image_dev']) # Write gpu result to host
    kernel_obj['queue'].finish()
    return kernel_obj['image']
    

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
    # float 32
    kernel_f32_obj = init_kernel(KERNEL_F32)
    _, M_f32 = benchmark(run_mandelbrot_f32, kernel_f32_obj, N, max_iter, x_min, x_max, y_min, y_max)
    
    plt.imshow( M_f32, extent=(x_min, x_max, y_min, y_max), cmap='plasma' )
    plt.colorbar()
    plt.title(f'Mandelbrot plot - Float 32 GPU {N}x{N}')
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.show()
    
    # float 64
    kernel_f64_obj = init_kernel(KERNEL_F64)
    dev = kernel_f64_obj['ctx'].devices[0]
    # Check support for float 64:
    if 'cl_khr_fp64' not in dev.extensions:
        print('No native fp64 -- Apple Silicon: exiting')
    else:
        _, M_f64 = benchmark(run_mandelbrot_f64, kernel_f64_obj, N, max_iter, x_min, x_max, y_min, y_max)
        plt.imshow( M_f64, extent=(x_min, x_max, y_min, y_max), cmap='plasma' )
        plt.colorbar()
        plt.title(f'Mandelbrot plot - Float 64 GPU {N}x{N}')
        plt.xlabel('Real axis')
        plt.ylabel('Imaginary axis')
        plt.show()
    