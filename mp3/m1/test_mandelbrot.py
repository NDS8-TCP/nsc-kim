# ----------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------
from mandelbrot_numba import mandelbrot_pixel # This is the same implementation in all Mandelbrot variants
import mandelbrot_numba
import mandelbrot_multiprocessing
import mandelbrot_dask
import pytest
import numpy as np
from dask.distributed import Client, LocalCluster

# ----------------------------------------------------------------------------------------------------
# Generic Mandelbrot test cases
# ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "c_real, c_imag, max_iter, expected",[
        (0.0, 0.0, 100, 100), # Inside the set
        (5.0, 0.0, 100, 1)    # Far outside the set
    ])
def test_mandelbrot_pixel(c_real, c_imag, max_iter, expected):
    """ Check if complex point is inside mandelbrot set or outside """
    assert mandelbrot_pixel(c_real, c_imag, max_iter) == expected

# ----------------------------------------------------------------------------------------------------
# Numba test cases
# ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "c_real, c_imag, max_iter, expected",[
        (0.0, 0.0, 100, 100),
        (5.0, 0.0, 100, 1),
    ])
def test_mandelbrot_pixel_identical_behavior(c_real, c_imag, max_iter, expected):
    """ Check that JIT compiled mandelbrot_pixel() matches behavior of pure Python mandelbrot_pixel() """
    assert mandelbrot_numba.mandelbrot_pixel_python(c_real, c_imag, max_iter) == expected
    assert mandelbrot_numba.mandelbrot_pixel(c_real, c_imag, max_iter) == expected

@pytest.mark.parametrize(
    "c_real, c_imag, max_iter",[
        (0.0, 0.0, 100),
        (5.0, 0.0, 100),
        (0.5, 1.0, 100),
    ])
def test_numba_matches_python(c_real, c_imag, max_iter):
    """ Check that JIT compiled mandelbrot_pixel() matches pure Python mandelbrot_pixel() """
    python_result = mandelbrot_numba.mandelbrot_pixel_python(c_real, c_imag, max_iter)
    jit_result    = mandelbrot_numba.mandelbrot_pixel(c_real, c_imag, max_iter)
    assert python_result == jit_result

@pytest.mark.parametrize(
    "N, expected_shape, expected_dtype",[
        (256, (256,256), np.int32),
        (512, (512,512), np.int32),
        (1024, (1024,1024), np.int32)
    ])
def test_numba_shape_dtype(N, expected_shape, expected_dtype):
    """ Check shape and datatype of mandelbrot_serial() """
    result :np.ndarray = mandelbrot_numba.mandelbrot_serial(N, -2.0, 1.0, -1.5, 1.5, 100)
    assert result.shape == expected_shape
    assert result.dtype == expected_dtype

def test_numba_chunk_matches_serial():
    """ Check that result from mandelbrot_serial() matches mandelbrot_chunk() """
    N = 512
    serial_result :np.ndarray = mandelbrot_numba.mandelbrot_serial(N, -2.0, 1.0, -1.5, 1.5, 100)
    chunk_result  :np.ndarray = mandelbrot_numba.mandelbrot_chunk(0, N, N, -2.0, 1.0, -1.5, 1.5, 100)
    assert np.array_equal(serial_result, chunk_result)

def test_numba_deterministic():
    """ Check that mandelbrot_serial() always produces the same result if run twice"""
    N = 512
    result_1 :np.ndarray = mandelbrot_numba.mandelbrot_serial(N, -2.0, 1.0, -1.5, 1.5, 100)
    result_2 :np.ndarray = mandelbrot_numba.mandelbrot_serial(N, -2.0, 1.0, -1.5, 1.5, 100)
    assert np.array_equal(result_1, result_2)

# ----------------------------------------------------------------------------------------------------
# Multiprocessing test cases
# ----------------------------------------------------------------------------------------------------
def test_multiprocessing_worker_matches_chunk():
    """ Check that result from mandelbrot_chunk() matches _worker() wrapper result """
    N = 512
    args = (0, N, N, -2.0, 1.0, -1.5, 1.5, 100)
    chunk_result  = mandelbrot_multiprocessing.mandelbrot_chunk(*args)
    worker_result = mandelbrot_multiprocessing._worker(args)
    assert np.array_equal(chunk_result, worker_result)

def test_multiprocessing_matches_serial():
    """ Check that multiprocessing mathces serial """
    N = 64
    args = (-2.0, 1.0, -1.5, 1.5, 100)

    # Serial baseline
    serial_result = mandelbrot_multiprocessing.mandelbrot_chunk(0, N, N, *args)

    # Simulate multiprocessing manually without pool
    n_workers = 4
    chunk_size = max(1, N // n_workers)

    chunks = []
    row = 0
    while row < N:
        end = min(row + chunk_size, N)
        chunks.append((row, end, N, *args))
        row = end
    
    results = [ mandelbrot_multiprocessing._worker(chunk) for chunk in chunks ]
    assembled = np.vstack(results)
    
    assert np.array_equal(serial_result, assembled)

def test_multiprocessing_uneven():
    """ Check that multiprocessing can handle an ueven chunk size """
    N = 50
    args = (-2.0, 1.0, -1.5, 1.5, 100)

    # Serial baseline
    serial_result = mandelbrot_multiprocessing.mandelbrot_chunk(0, N, N, *args)

    # Simulate multiprocessing manually without pool
    n_workers = 4
    chunk_size = max(1, N // n_workers)

    chunks = []
    row = 0
    while row < N:
        end = min(row + chunk_size, N)
        chunks.append((row, end, N, *args))
        row = end
    
    results = [ mandelbrot_multiprocessing._worker(chunk) for chunk in chunks ]
    assembled = np.vstack(results)
    
    assert np.array_equal(serial_result, assembled)

@pytest.mark.parametrize(
    "N",[1, 2, 8, 50, 256, 512, 1024])
def test_multiprocessing_vary_N(N):
    """ Check that multiprocessing can handle different resolution sizes """
    args = (-2.0, 1.0, -1.5, 1.5, 100)

    # Serial baseline
    serial_result = mandelbrot_multiprocessing.mandelbrot_chunk(0, N, N, *args)

    # Simulate multiprocessing manually without pool
    n_workers = 4
    chunk_size = max(1, N // n_workers)

    chunks = []
    row = 0
    while row < N:
        end = min(row + chunk_size, N)
        chunks.append((row, end, N, *args))
        row = end
    
    results = [ mandelbrot_multiprocessing._worker(chunk) for chunk in chunks ]
    assembled = np.vstack(results)
    
    assert np.array_equal(serial_result, assembled)

# ----------------------------------------------------------------------------------------------------
# Local Dask test cases
# ----------------------------------------------------------------------------------------------------
@pytest.fixture
def dask_client():
    """ Used to setup local cluster & connect to local Dask client, execute client tasks & terminate connection """
    cluster = LocalCluster(n_workers=2, threads_per_worker=1)
    client = Client(cluster)
    yield client

    # Terminate
    client.close()
    cluster.close()

def test_dask_submit_gather_chunk(dask_client):
    """ Check Dask integration using submit/gather & validating against non Dask execution """
    N = 64
    args = (0, N, N, -2.0, 1.0, -1.5, 1.5, 100)
    future   = dask_client.submit( mandelbrot_dask.mandelbrot_chunk, *args )
    result   = dask_client.gather(future)
    expected = mandelbrot_dask.mandelbrot_chunk(*args)
    assert np.array_equal(result, expected)

def test_dask_mandelbrot_matches_serial(dask_client):
    """ Validate correctness of Dask compute method """
    N = 64
    args = (-2.0, 1.0, -1.5, 1.5, 100)
    dask_result   = mandelbrot_dask.mandelbrot_dask(N, *args, n_chunks=4)
    serial_result = mandelbrot_dask.mandelbrot_chunk(0, N, N, *args)
    assert np.array_equal(dask_result, serial_result)

def test_dask_uneven_chunks(dask_client):
    """ Check if Dask can handle uneven chunk size (e.g. 50/4=12.5) """
    N = 50
    args = (-2.0, 1.0, -1.5, 1.5, 100)
    dask_result   = mandelbrot_dask.mandelbrot_dask(N, *args, n_chunks=4)
    serial_result = mandelbrot_dask.mandelbrot_chunk(0, N, N, *args)
    assert np.array_equal(dask_result, serial_result)

@pytest.mark.parametrize(
    "n_chunks",[1, 2, 32, 100])
def test_dask_various_chunk_sizes(dask_client, n_chunks):
    """ Check that local Dask implementation can handle n_chunks<N, n_chunks=N, n_chunks>N"""
    N = 32
    args = (-2.0, 1.0, -1.5, 1.5, 100)
    dask_result   = mandelbrot_dask.mandelbrot_dask(N, *args, n_chunks=n_chunks)
    serial_result = mandelbrot_dask.mandelbrot_chunk(0, N, N, *args)
    assert np.array_equal(dask_result, serial_result)
