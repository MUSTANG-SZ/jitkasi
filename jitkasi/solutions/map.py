from dataclasses import cached_property, dataclass, field
from functools import partial
from typing import Optional

import jax.numpy as jnp
import numpy as np
from astropy.wcs import WCS
from jax import Array, jit
from jax.tree_util import register_pytree_node_class
from typing_extensions import Self

from ..noise import NoiseI, NoiseModel
from ..tod import TOD, TODVec
from .core import Solution


@register_pytree_node_class
@dataclass
class WCSMap(Solution):
    """
    Class for solving for a map.
    Uses a WCS header for its pixel definitions.
    See `Solution` for inherited attributes.
    This class is a registered pytree so it is JITable with JAX.

    TODO: Caching and purging of pixelization

    Attributes
    ----------
    wcs : WCS
        The WCS header that defines the map pixelization.
        This is aux data for the pytree.
    pixelization : str
        The pixelization method used when projecting to/from TODs.
        Currently accepted values are:
        * 'nn': Nearest neighpor pixelization.
        This is aux data for the pytree.
    """

    wcs: WCS
    pixelization: str
    noise: NoiseModel = field(default_factory=NoiseI)
    hits: Optional[Array] = None
    _pix_reg: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        # Add pixelization methods to the registry
        self._pix_reg["nn"] = self.nn_pix
        # Check that we have a valid pixelization scheme
        if self.pixelization not in self._pix_reg.keys():
            raise ValueError(f"Invalid pixelization: {self.pixelization}")

    @jit
    def nn_pix(self, tod: TOD) -> Array:
        coords = jnp.column_stack(
            [jnp.ravel(tod.x * 180.0 / jnp.pi), jnp.ravel(tod.y * 180.0 / jnp.pi)]
        )
        # -1 is to go between unit offset in FITS and zero offset in python
        pix = jnp.round(jnp.array(self.wcs.wcs_world2pix(coords, 1)) - 1.0).reshape(
            (2,) + tod.data.shape
        )
        return pix.astype(jnp.int32)

    @jit
    def _nn_bin(self, bin_into: Array, data: Array, pix: Array):
        bin_into = bin_into.at[pix].add(data.at[:].get(), mode="drop")
        return bin_into

    @jit
    def _nn_unbin(self, unbin_into: Array, data: Array, pix: Array):
        unbin_into = unbin_into.at[:].add(data.at[pix].get(mode="fill", fill_value=0.0))
        return unbin_into

    @jit
    def to_tods(self, todvec: TODVec) -> TODVec:
        """
        Project the map into TODs.

        Parameters
        ----------
        todvec : TODVec
            TODVec containing TODs to project into.
            Only used for pointing and shape information,
            not modified in place.

        Returns
        -------
        todvec_out : TODVec
            A TODvec where each with the map projected into it.
            The order of TODs here is the same as the input TODVec
            and the TODs within are shallow copied to the non-data
            Arrays reference the same memory as the original TODs.
        """
        todvec_out = todvec.copy(deep=False)
        for tod in todvec_out:
            pix = self._pix_reg[self.pixelization](tod)
            tod.data = jnp.zeros_like(tod.data)
            if self.pixelization == "nn":
                tod.data = self._nn_unbin(tod.data, self.data, pix)
        return todvec_out

    @jit
    def from_tods(self, todvec: TODVec, use_filt: bool = True) -> Self:
        """
        Project TODs into a map.

        Parameters
        ----------
        todvec : TODVec
            TODVec containing TODs to project from.
        use_filt : bool, defauls: True
            If True use data_filt instead of data.

        Returns
        -------
        wcsmap_out : WCSMap
            A WCSmap with the TODs binned into it.
            This is a shallow copy of the current object so all Arrays
            except for data reference the same memory as this object.
        """
        wcsmap_out = self.copy(deep=False)
        wcsmap_out.data = jnp.zeros_like(self.data)
        for tod in todvec:
            pix = self._pix_reg[self.pixelization](tod)
            data = tod.data
            if use_filt:
                data = tod.data_filt
            if self.pixelization == "nn":
                wcsmap_out.data = self._nn_bin(wcsmap_out.data, data, pix)
        return wcsmap_out

    @jit
    def make_hits(self, todvec: TODVec, use_filt: bool = True) -> Self:
        """
        Get the hits map given a set of TODs.

        Parameters
        ----------
        todvec : TODVec
            TODVec containing TODs to project from.
        use_filt : bool, defauls: True
            If True apply the TOD noise model.

        Returns
        -------
        wcsmap : WCSMap
            A WCSmap with the hits map included.
            This is the same as the current object,
            but needs to be returned for jit to work.
        """
        self.hits = jnp.zeros_like(self.data)
        for tod in todvec:
            pix = self._pix_reg[self.pixelization](tod)
            data = jnp.ones_like(tod.data)
            if use_filt:
                data = tod.noise.apply_noise(data)
            if self.pixelization == "nn":
                self.hits = self._nn_bin(self.hits, data, pix)
        return self

    @cached_property
    def data_filt(self) -> Array:
        """
        Get a copy of the data with the noise model applied.
        This is essentially $M^{-1}m$

        Returns
        -------
        data_filt : Array
            The filtered data.
            If `self.noise` is None then this is just a copy of `self.data`.
        """
        if self.noise is None:
            return jnp.copy(self.data)
        return self.noise.apply_noise(self.data)

    def compute_noise(
        self, noise_class: NoiseModel, data: Optional[Array], *args, **kwargs
    ):
        """
        Compute and set the noise model for this map.
        This uses `noise_class.compute(dat=self.data...` to compute the noise.
        Also resets the cache on `data_filt`.

        Parameters
        ----------
        noise_class : NoiseModel
            The class to use as the noise model.
            Nominally a class from `jitkasi.noise`.
        data : Optional[Array], default: None
            Data to compute the noise model with.
            If None we use `self.data`.
        *args
            Additional arguments to pass to `noise_class.compute`.
        *kwargs
            Additional keyword arguments to pass to `noise_class.compute`.
        """
        self.__dict__.pop("data_filt", None)
        if data is None:
            data = self.data
        self.noise = noise_class.compute(data, *args, **kwargs)

    def recompute_noise(self, data: Optional[Array], *args, **kwargs):
        """
        Helper function that wraps `compute_noise` but uses the same class
        as the current instance of `self.noise`.

        Parameters
        ----------
        data : Optional[Array], default: None
            Data to compute the noise model with.
            If None we use `self.data`.
        *args
            Additional arguments to pass to `noise_class.compute`.
        *kwargs
            Additional keyword arguments to pass to `noise_class.compute`.
        """
        noise_class = self.noise.__class__
        self.compute_noise(noise_class, data, *args, **kwargs)

    @classmethod
    def empty(
        cls,
        *,
        wcs: WCS,
        lims: tuple[float, float, float, float],
        pad=0,
        square=False,
        pixelization="nn",
        **_,
    ) -> Self:
        """
        Initialize an empty map.

        Parameters
        ----------
        wcs : WCS
            The WCS kernel to use for this map.
        lims : tuple[float, float, float, float]
            The limits of the map in radians.
            Should be (RA low, RA high, Dec low, Dec high).
        pad : int, default: 0
            Number of pixels to pad the map by.
        square : bool, default: False
            If True make the map square.
        pixelization : str, default: 'nn'
            The pixelization method to use.
            See `WCSMap` documentation for more details.
        """
        corners = np.zeros([4, 2])
        corners[0, :] = [lims[0], lims[2]]
        corners[1, :] = [lims[0], lims[3]]
        corners[2, :] = [lims[1], lims[2]]
        corners[3, :] = [lims[1], lims[3]]

        pix_corners = np.array(wcs.wcs_world2pix(corners * 180 / jnp.pi, 1))
        pix_corners = np.round(pix_corners)

        if pix_corners.min() < -0.5:
            print(
                "corners seem to have gone negative in SkyMap projection.  not good, you may want to check this."
            )
        nx = int(pix_corners[:, 0].max() + pad)
        ny = int(pix_corners[:, 1].max() + pad)

        if square:
            if nx > ny:
                ny = nx
            else:
                nx = ny
        data = jnp.zeros((nx, ny))

        return cls(data, wcs, pixelization)

    def _self_check(self, other: Self):
        if self.wcs != other.wcs:
            raise ValueError("Cannot operate on WCSMaps that have different WCSs")

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (self.data,)
        aux_data = (self.wcs, self.pixelization)

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        return cls(*children, *aux_data)


# Map based noise classes
@register_pytree_node_class
@dataclass
class NoiseWhite:
    """
    A simple noise model with only white noise.
    Here we just weigh each pixel individually.
    This class is a registered pytree so it is JITable with JAX.

    Attributes
    ----------
    weights: Array
        The per-pixel weights computed from the hits map.
        This is a child of the pytree.
    """

    weights: Array

    @jit
    def apply_noise(self, dat: Array) -> Array:
        """
        Apply the noise model.
        In this case this is just rescaling each pixel by its weight.

        Parameters
        ----------
        dat : Array
            The data to apply the noise model to.
            Should be 2d with `dat.shape == self.weights.shape`.

        Returns
        -------
        dat_filt : Array
            The data with the noise model applied.
        """
        return dat * self.weights

    @classmethod
    @partial(jit, static_argnums=(0,))
    def compute(cls, dat: Array, hits: Array) -> Self:
        """
        Compute this noise model based on some input data.
        This requires you to have a hits map for your map.

        Parameters
        ----------
        dat : Array
            The map data.
            This is only here for API compatibility.
        hits : Array
            The hits map to use as weights.

        Returns
        -------
        noise_model : NoiseWhite
            An instance of NoiseWhite with the computed noise model.
        """
        _ = dat
        return cls(hits)

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, None]:
        children = (self.weights,)
        aux_data = None

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del aux_data
        return cls(*children)
