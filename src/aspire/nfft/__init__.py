import numpy as np
from finufftpy import nufft2d2, nufft3d2, nufft2d1, nufft3d1


def nufft3(vol_f, fourier_pts, sz=None, real=False):
    if sz is None:
        sz = vol_f.shape

    dim = len(sz)
    if dim == 2:
        fn = nufft2d2
    elif dim == 3:
        fn = nufft3d2

    epsilon = max(1e-15, np.finfo(vol_f.dtype).eps)
    fourier_pts = np.asarray(np.mod(fourier_pts + np.pi, 2 * np.pi) - np.pi, order='C')

    num_pts = fourier_pts.shape[1]
    result = np.zeros(num_pts).astype('complex128')

    result_code = fn(
        *fourier_pts,
        result,
        -1,
        epsilon,
        vol_f
    )

    if result_code != 0:
        raise RuntimeError(f'FINufft transform failed. Result code {result_code}')

    return np.real(result) if real else result


def anufft3(vol_f, fourier_pts, sz, real=False):

    dim = len(sz)
    if dim == 2:
        fn = nufft2d1
    elif dim == 3:
        fn = nufft3d1
    else:
        raise RuntimeError('only 2d and 3d adjoints supported')

    epsilon = max(1e-15, np.finfo(vol_f.dtype).eps)
    fourier_pts = np.asarray(np.mod(fourier_pts + np.pi, 2 * np.pi) - np.pi, order='C')

    result = np.zeros(sz, order='F').astype('complex128')

    result_code = fn(
        *fourier_pts,
        vol_f,
        1,
        epsilon,
        *sz,
        result
    )
    if result_code != 0:
        raise RuntimeError(f'FINufft adjoint failed. Result code {result_code}')

    return np.real(result) if real else result
