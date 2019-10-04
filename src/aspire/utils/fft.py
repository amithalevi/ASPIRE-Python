"""
FFT/IFFT utilities
"""

from scipy.fftpack import ifftshift, ifft, ifft2, fftshift, fft, fft2, ifftn, fftn


def centered_ifft1(x, axis=0):
    """
    Calculate a centered, one-dimensional inverse FFT
    :param x: The one-dimensional signal to be transformed.
    :param axis: The axis along with to apply the inverse FFT.
    :return: The centered inverse Fourier transform of x.
    """
    x = ifftshift(x, axes=axis)
    x = ifft(x, axis=axis)
    x = fftshift(x, axes=axis)
    return x


def centered_fft1(x, axis=0):
    """
    Calculate a centered, one-dimensional FFT
    :param x: The one-dimensional signal to be transformed.
    :param axis: The axis along with to apply the FFT.
    :return: The centered Fourier transform of x.
    """
    x = ifftshift(x, axes=axis)
    x = fft(x, axis=axis)
    x = fftshift(x, axes=axis)
    return x


def centered_ifft2(x, axes=(0, 1)):
    """
    Calculate a centered, two-dimensional inverse FFT
    :param x: The two-dimensional signal to be transformed.
    :param axes: The axes along which we apply the inverse FFT.
    :return: The centered inverse Fourier transform of x.
    """
    ax0, ax1 = axes
    x = ifftshift(ifftshift(x, ax0), ax1)
    x = ifft2(x, axes=(ax0, ax1))
    x = fftshift(fftshift(x, ax0), ax1)
    return x


def centered_fft2(x, axes=(0, 1)):
    """
    Calculate a centered, two-dimensional FFT
    :param x: The two-dimensional signal to be transformed.
    :param axes: The axes along which we apply the FFT.
    :return: The centered Fourier transform of x.
    """
    ax0, ax1 = axes
    x = ifftshift(ifftshift(x, ax0), ax1)
    x = fft2(x, axes=(ax0, ax1))
    x = fftshift(fftshift(x, ax0), ax1)
    return x


def centered_ifft3(x, axes=(0, 1, 2)):
    """
    Calculate a centered, three-dimensional inverse FFT
    :param x: The three-dimensional signal to be transformed.
    :param axes: The axes along which we apply the inverse FFT.
    :return: The centered inverse Fourier transform of x.
    """
    ax0, ax1, ax2 = axes
    x = ifftshift(ifftshift(ifftshift(x, ax0), ax1), ax2)
    x = ifftn(x, axes=(ax0, ax1, ax2))
    x = fftshift(fftshift(fftshift(x, ax0), ax1), ax2)
    return x


def centered_fft3(x, axes=(0, 1, 2)):
    """
    Calculate a centered, three-dimensional FFT
    :param x: The three-dimensional signal to be transformed.
    :param axes: The axes along which we apply the FFT.
    :return: The centered Fourier transform of x.
    """
    ax0, ax1, ax2 = axes
    x = ifftshift(ifftshift(ifftshift(x, ax0), ax1), ax2)
    x = fftn(x, axes=(ax0, ax1, ax2))
    x = fftshift(fftshift(fftshift(x, ax0), ax1), ax2)
    return x


def mdim_ifftshift(x, dims=None):
    """
    Multi-dimensional FFT unshift
    :param x: The array to be unshifted.
    :param dims: An array of dimension indices along which the unshift should occur.
        If None, the unshift is performed along all dimensions.
    :return: The x array unshifted along the desired dimensions.
    """
    if dims is None:
        dims = range(0, x.ndim)
    for dim in dims:
        x = ifftshift(x, dim)
    return x


def mdim_fftshift(x, dims=None):
    """
    Multi-dimensional FFT shift

    :param x: The array to be shifted.
    :param dims: An array of dimension indices along which the shift should occur.
        If None, the shift is performed along all dimensions.
    :return: The x array shifted along the desired dimensions.
    """
    if dims is None:
        dims = range(0, x.ndim)
    for dim in dims:
        x = fftshift(x, dim)
    return x
