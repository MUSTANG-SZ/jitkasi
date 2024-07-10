from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax.numpy as jnp
from jax import Array, jit
from jax.scipy.special import erfinv
from jax.tree_util import register_pytree_node_class
from typing_extensions import Protocol, Self, runtime_checkable


@runtime_checkable
class NoiseModel(Protocol):
    def apply_noise(self, dat: Array) -> Array: ...

    @classmethod
    def compute(cls, dat: Array, /) -> Self: ...


@register_pytree_node_class
@dataclass
class NoiseWhite:
    """
    A simple noise model with only white noise.
    This is equivalent to a diagonal $N^{-1}$.
    This class is a registered pytree so it is JITable with JAX.

    Attributes
    ----------
    weights: Array
        The per-detector weights conputed from the white noise.
        This is a child of the pytree.
    """

    weights: Array

    @jit
    def apply_noise(self, dat: Array) -> Array:
        """
        Apply the noise model.
        In this case this is just rescaling each detector by its weight.

        Parameters
        ----------
        dat : Array
            The data to apply the noise model to.
            Should be 2d with `dat.shape[0] == len(self.weights)`.

        Returns
        -------
        dat_filt : Array
            The data with the noise model applied.
        """
        return dat * self.weights[..., None]

    @classmethod
    @partial(jit, static_argnums=(0,))
    def compute(cls, dat: Array) -> Self:
        """
        Compute this noise model based on some input data.
        Here we just estimate the variance of each detector in dat.

        Parameters
        ----------
        dat : Array
            The data to estimate the white noise levels from.
            Should be a 2d array.

        Returns
        -------
        noise_model : NoiseWhite
            An instance of NoiseWhite with the computed noise model.
        """
        weights = (
            2 * erfinv(0.5) / jnp.median(jnp.abs(jnp.diff(dat, axis=1)), axis=1)
        ) ** 2
        return cls(weights)

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, None]:
        children = (self.weights,)
        aux_data = None

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        return cls(*children)


@register_pytree_node_class
@dataclass
class NoiseSmoothedSVD:
    """
    The standard "Jon style" noise model.
    Here we use the SVD to compute the noise spectrum.
    This is assuming that we are operating on noise dominated data.
    This class is a registered pytree so it is JITable with JAX.

    While this model is applied in fourier space it appears to be mathematically
    the same as if we computed $N^{-1}$ directly from the SVD and then dotted that
    into the data being filtered. Math showing this (if actually true) will exist
    in the docs one day...

    Attributes
    ----------
    v : Array
        The right singular vectors to of the modeled noise.
        This is used to rotate any input data into the space of the noise.
        This is a child of the pytree.
    filt_spectrum : Array
        The inverse of the smoothed sepctrum of the noise model.
        This is used as a filter to remove the modeled noise in fourier space.
    """

    v: Array
    filt_spectrum: Array

    @jit
    def apply_noise(self, dat: Array) -> Array:
        """
        Apply the noise model.
        The procedure here is to rotate the data into the space of the noise,
        then filter out the noise in fourier space, and then rotate back.

        Parameters
        ----------
        dat : Array
            The data to apply the noise model to.
            Should be 2d with `dat.shape[0] == len(self.weights)`.

        Returns
        -------
        dat_filt : Array
            The data with the noise model applied.
        """
        dat_rot = jnp.dot(self.v, dat)
        dat_tmp = jnp.hstack([dat_rot, jnp.fliplr(dat_rot[:, 1:-1])])
        dat_rft = jnp.real(jnp.fft.rfft(dat_tmp, axis=1))
        dat_filt = jnp.fft.irfft(
            self.filt_spectrum[:, : dat_rft.shape[1]] * dat_rft, axis=1, norm="forward"
        )[:, : dat.shape[1]]
        dat_filt = jnp.dot(self.v.T, dat_filt)
        dat_filt = dat_filt.at[:, 0].multiply(0.50)
        dat_filt = dat_filt.at[:, -1].multiply(0.50)
        return dat_filt

    @classmethod
    @partial(jit, static_argnums=(0,))
    def compute(cls, dat: Array, fwhm: float) -> Self:
        """
        Compute this noise model based on some input data.
        To do this we compute the SVD of the data, rotate into its sigular space,
        and then compute the spectrum of each rotated detector.
        These spectra are then smoothed and a filter a made by squaring and
        inverting the smoothed filter.

        Parameters
        ----------
        dat : Array
            The data to estimate the white noise levels from.
            Should be a 2d array.

        Returns
        -------
        noise_model : NoiseWhite
            An instance of NoiseWhite with the computed noise model.
        """
        u, *_ = jnp.linalg.svd(dat, True)
        v = u.T
        dat_rot = jnp.dot(v, dat)
        dat_ft = jnp.real(jnp.fft.rfft(dat_rot))
        smooth_kern = jnp.exp(
            -0.5 * (jnp.arange(dat_ft.shape[1]) * jnp.sqrt(8 * jnp.log(2)) / fwhm) ** 2
        )
        for i in range(dat_ft.shape[0]):
            dat_ft = dat_ft.at[i].set(jnp.convolve(dat_ft[i], smooth_kern) ** 2)
        dat_ft = dat_ft.at[:, 1:].set(1.0 / dat_ft[:, 1:])
        dat_ft = dat_ft.at[:, 0].set(0)
        return cls(v, dat_ft)

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, None]:
        children = (self.v, self.filt_spectrum)
        aux_data = None

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        return cls(*children)
