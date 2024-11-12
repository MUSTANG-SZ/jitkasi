"""
Generically useful JITted math functions
"""

import jax.numpy as jnp
from jax import Array, jit, lax


@jit
def dct_i(dat: Array, inv: bool = False, axis: int = -1) -> Array:
    r"""
    Compute the 1d discrete cosine transform (DCT) of the first kind.

    Note that since this uses `rfft` to compute the DCT this suffers from
    slightly lower accuracy than a proper DCT such at the `fftw` or `scipy` ones.
    Passing white noise back and forth through this function results in errors
    at the $1e-7$ level.

    Parameters
    ----------
    dat : Array
        The data to DCT.
        If this is multidimensional the DCT is computed along only `axis`.
    inv : bool, defualt: False
        If True apply the $\frac{1}{2*(n-1)}$ normalization that turns this into
        the inverse DCT.
    axis : int, defualt: -1
        The axis to compute the DCT along.

    Returns
    -------
    dat_dct : Array
        The DCT of the first kind of `dat`.
        Has the same shape and dtype as `dat`.
    """
    s = dat.shape
    dat = jnp.atleast_2d(dat)
    dat_tmp = jnp.hstack([dat, jnp.fliplr(dat[:, 1:-1])])
    dat_dct = jnp.real(jnp.fft.rfft(dat_tmp))
    dat_dct = lax.select(
        inv, dat_dct.at[:].multiply(1.0 / (2 * (dat.shape[axis] - 1))), dat_dct
    )
    return dat_dct.reshape(s)


@jit
def gauss_smooth_1d(dat: Array, fwhm: float) -> Array:
    """
    Smooth an array along its last axis with a gaussian.
    If you want to smooth along another axis consider using `roll` or `moveaxis`.

    Parameters
    ----------
    dat : Array
        The data to smooth.
    fwmh : float
        The full width half max of the gaussian used to smooth.

    Returns
    -------
    dat_smooth : Array
        The smoothed data.
        Has the same shape and dtype as `dat`.
    """
    smooth_kern = jnp.exp(
        -0.5 * (jnp.arange(dat.shape[-1]) * jnp.sqrt(8 * jnp.log(2)) / fwhm) ** 2
    )
    tot = smooth_kern[0] + smooth_kern[-1] + 2 * jnp.sum(smooth_kern[1:-1])
    smooth_kern = smooth_kern.at[:].multiply(1.0 / tot)
    smooth_kern = dct_i(smooth_kern)
    dat_smooth = dct_i(dat)
    dat_smooth = dat_smooth.at[:].multiply(smooth_kern)
    dat_smooth = dct_i(dat_smooth, True)

    return dat_smooth
