import warnings
from dataclasses import dataclass
from functools import partial
from logging import warning
from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np
from jax import Array, ShapeDtypeStruct, jit, pure_callback
from jax.scipy.special import erfinv
from jax.tree_util import register_pytree_node_class
from numpy.typing import NDArray
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


@register_pytree_node_class
@dataclass
class NoiseWrapper:
    """
    Wrapper to use external noise models in `jitkasi`.

    Be warned that while this is provided we cannot gaurentee that the external class
    will play nicely with the rest of the library.
    In particular while the wrapper for `apply_noise` is set up to be called from a jitted function,
    `compute` is not since it includes some non-pure actions.

    Attributes
    ----------
    ext_inst : Callable
        The instance of the external noise model class we are wrapping.
    apply_func : str
        The name of the function in `ext_inst` that applies noise.
        We expect this to take a single 2d array as an argument.
    jitted : bool
        Whether or not the external noise class JITs its functions.
        If this is False you should be very caruful about how you use this class.
    shape_dtypes : ShapeDtypeStruct
        The shape and dtype of the array that the `apply_func` expects.
        This is really only used so what we can call `apply_func` when we aren't jitted.
    """

    ext_inst: Callable
    apply_func: str
    jitted: bool
    shape_dtypes: ShapeDtypeStruct

    def __post_init__(self):
        if not self.jitted:
            warnings.warn(
                "Initialized a non-JIT noise class! This is not supported in all usecases, proceed with caution."
            )

    def apply_noise(self, dat: Array) -> Array:
        """
        Apply the external noise model.
        This should be safe to call from jitted functions.

        Parameters
        ----------
        dat : Array
            The data to apply the noise model to.
            Should have shape and dtype that matches the `shape_dtypes` attribute.

        Returns
        -------
        dat_filt : Array
            The data with the noise model applied.
        """
        if self.jitted:
            return getattr(self.ext_inst, self.apply_func)(dat)
        return pure_callback(
            getattr(self.ext_inst, self.apply_func), self.shape_dtypes, np.array(dat)
        )

    @classmethod
    def compute(
        cls,
        dat: Array,
        ext_class: Callable,
        compute_func: str,
        apply_func: str,
        jitted: bool,
        *args,
        **kwargs,
    ) -> Self:
        """
        Make an instance of the external noise class and make an instance of this class to wrap it.

        Parameters
        ----------
        dat : Array
            The data to compute the noise model with.
        ext_class : Callable
            The external noise model class.
            Must have a class method (`__call__` is acceptable) that will create an instance
            and compute the noise model.
        compute_func : str
            The function that computes an instance of the external noise model.
            Should take `dat` as its first argument.
        apply_func : str
            The function that applies the noise model to some data.
        jitted : str
            Whether or not the external noise class JITs its own functions.
        *args
            Additional arguments to pass to `compute_func`.
        **kwargs
            Additional keyword arguments to pass to `compute_func`.

        Returns
        -------
        noise_model : NoiseWrapper
            An instance of `NoiseWrapper` that wraps an instance of `ext_class`.
        """
        shape_dtypes = ShapeDtypeStruct(dat.shape, dat.dtype)
        data: Array | NDArray = dat
        if not jitted:
            data = np.array(dat)
        inst = getattr(ext_class, compute_func)(data, *args, **kwargs)
        return cls(inst, apply_func, jitted, shape_dtypes)

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[None, tuple]:
        children = None
        aux_data = (self.ext_inst, self.apply_func, self.jitted, self.shape_dtypes)

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        return cls(*aux_data)
