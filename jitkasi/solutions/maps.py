from dataclasses import dataclass, field
from functools import cached_property, partial
from typing import Optional, Type

import jax.numpy as jnp
import mpi4jax
import numpy as np
from astropy.wcs import WCS
from jax import Array, jit
from jax.tree_util import register_pytree_node_class
from mpi4py import MPI
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
    noise : NoiseModel, default: NoiseI
        The map space noise model.
    ivar : Array, jnp.ones(1)
        The inverse variance of the map.
        By default this is one everywhere.
    """

    wcs: WCS
    pixelization: str
    noise: NoiseModel = field(default_factory=NoiseI)
    ivar: Array = field(default_factory=partial(jnp.ones, 1))
    _pix_reg: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        # Add pixelization methods to the registry
        self._pix_reg["nn"] = self.nn_pix
        # Check that we have a valid pixelization scheme
        if self.pixelization not in self._pix_reg.keys():
            raise ValueError(f"Invalid pixelization: {self.pixelization}")
        if self.ivar.shape != self.data.shape:
            self.ivar = jnp.ones_like(self.data)

    @cached_property
    def xy(self) -> tuple[Array, Array]:
        """
        Get the ra and dex at each pixel in the map.

        Returns
        -------
        ra : Array
            The Ra at each pixel.
        dec : Array
            The dec at eahc pixel
        """
        xx, yy = np.meshgrid(
            np.arange(1, self.data.shape[0] + 1, dtype=float),
            np.arange(1, self.data.shape[1] + 1, dtype=float),
        )
        x, y = self.wcs.wcs_pix2world(np.array(xx.ravel()), np.array(yy.ravel()), 1)
        return (
            jnp.array(x).reshape(self.data.shape) * jnp.pi / 180.0,
            jnp.array(y).reshape(self.data.shape) * jnp.pi / 180.0,
        )

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
            A WCSmap with the hits map included as `wcsmap.ivar`.
            This is the same as the current object,
            but needs to be returned for jit to work.
        """
        self.ivar = jnp.zeros_like(self.data)
        for tod in todvec:
            pix = self._pix_reg[self.pixelization](tod)
            data = jnp.ones_like(tod.data)
            if use_filt:
                data = tod.noise.apply_noise(data)
            if self.pixelization == "nn":
                self.ivar = self._nn_bin(self.ivar, data, pix)
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
        self, noise_class: Type[NoiseModel], data: Optional[Array], *args, **kwargs
    ):
        """
        Compute and set the noise model for this map.
        This uses `noise_class.compute(dat=self.data...` to compute the noise.
        Also resets the cache on `data_filt`.

        Parameters
        ----------
        noise_class : Type[NoiseModel]
            The class to use as the noise model.
            Nominally a class from `jitkasi.noise`.
        data : Optional[Array], default: None
            Data to compute the noise model with.
            If None we use `self.data`.
        *args
            Additional arguments to pass to `noise_class.compute`.
            Note that any argunment that is a string that starts with `self` will be evaled.
        *kwargs
            Additional keyword .reshape(self.data.shape)arguments to pass to `noise_class.compute`.
            Note that any argument value that is a string that starts with `self` will be evaled.
        """
        args = [
            eval(arg) if (isinstance(arg, str) and arg[:4] == "self") else arg
            for arg in args
        ]
        kwargs = {
            k: (eval(v) if (isinstance(v, str) and v[:4] == "self") else v)
            for k, v in kwargs.items()
        }
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

    @jit
    def reduce(self) -> Self:
        """
        MPI reduce the solution.
        This adds up all the data in all insances of this solution.
        """
        self.data, token = mpi4jax.allreduce(self.data, op=MPI.SUM, comm=self.comm)
        self.ivar, _ = mpi4jax.allreduce(
            self.ivar, op=MPI.SUM, comm=self.comm, token=token
        )
        return self

    @classmethod
    def empty(
        cls,
        *,
        name: str,
        comm: MPI.Intracomm,
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
        name : str
            The name of the map.
        comm : MPI.Intracomm
            The MPI communicator to use.
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

        return cls(name, data, comm, wcs, pixelization)

    def _self_check(self, other: Self):
        if self.wcs != other.wcs:
            raise ValueError("Cannot operate on WCSMaps that have different WCSs")

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (self.data, self.ivar, self.noise)
        aux_data = (self.name, self.comm, self.wcs, self.pixelization)

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        name, comm, wcs, pixelization = aux_data
        data, ivar, noise = children
        return cls(name, data, comm, wcs, pixelization, noise, ivar)


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
        The per-pixel weights computed from the ivar map.
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
    def compute(cls, dat: Array, ivar: Array) -> Self:
        """
        Compute this noise model based on some input data.
        This requires you to have a ivar map for your map.

        Parameters
        ----------
        dat : Array
            The map data.
            This is only here for API compatibility.
        ivar : Array
            The ivar map to use as weights.

        Returns
        -------
        noise_model : NoiseWhite
            An instance of NoiseWhite with the computed noise model.
        """
        _ = dat
        return cls(ivar)

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
