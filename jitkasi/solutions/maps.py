"""
Class for solving for a map.

Only supports T for now.
"""

from dataclasses import dataclass, field
from functools import cached_property, partial
from typing import Optional, Type

import jax.numpy as jnp
import mpi4jax
import numpy as np
from astropy.wcs import WCS
from jax import Array, jit
from jax.tree_util import register_pytree_node_class, tree_leaves
from mpi4py import MPI
from typing_extensions import Self

from ..noise import NoiseI, NoiseModel
from ..proj import map2tod, tod2map
from ..tod import TODVec
from ..wcs import JAXWCS
from .core import Solution


@register_pytree_node_class
@dataclass
class WCSMap(Solution):
    """
    Class for solving for a map using a WCS pixelization.

    See `Solution` for inherited attributes.

    This class is a registered pytree and can be used with JAX.

    Attributes
    ----------
    wcs : JAXWCS
        JAX-compatible WCS projector defining the map pixelization.
    noise : NoiseModel, default: NoiseI
        Map-space noise model.
    ivar : Array
        Inverse-variance or hit-weight map. Defaults to one everywhere.
    """

    wcs: JAXWCS
    noise: NoiseModel = field(default_factory=NoiseI)
    ivar: Array = field(default_factory=partial(jnp.ones, 1))

    def __post_init__(self):
        if self.ivar.shape != self.data.shape:
            self.ivar = jnp.ones_like(self.data)

    @cached_property
    def xy(self) -> tuple[Array, Array]:
        """
        Get the right ascension and declination at every map pixel.

        Returns
        -------
        ra : Array
            Right ascension in radians, with shape `self.data.shape`.
        dec : Array
            Declination in radians, with shape `self.data.shape`.
        """
        xx, yy = jnp.meshgrid(
            jnp.arange(self.data.shape[0], dtype=jnp.float32),
            jnp.arange(self.data.shape[1], dtype=jnp.float32),
            indexing="ij",
        )
        return self.wcs.pixel_to_world(xx, yy)

    @jit
    def to_tods(self, todvec: TODVec) -> TODVec:
        """
        Project the map into TODs using bilinear interpolation.

        Parameters
        ----------
        todvec : TODVec
            TODs containing the pointing information. Their data arrays
            are replaced with the projected map values in a shallow copy.

        Returns
        -------
        todvec_out : TODVec
            A shallow copy of the input TODVec containing the projected map values.
        """
        todvec_out = todvec.copy(deep=False)
        for tod in todvec_out:
            tod.data = map2tod(self.data, tod.x, tod.y, self.wcs)
        return todvec_out

    @jit
    def from_tods(self, todvec: TODVec, use_filt: bool = True) -> Self:
        """
        Accumulate TOD samples into a map using bilinear projection.

        Parameters
        ----------
        todvec : TODVec
            TODs containing the samples and pointing information.
        use_filt : bool, default: True
            If True, use `tod.data_filt` instead of `tod.data`.

        Returns
        -------
        wcsmap_out : WCSMap
            A shallow copy of this map with the accumulated TOD samples in its data array.
        """
        data_out = jnp.zeros_like(self.data)
        for tod in todvec:
            data = tod.data_filt if use_filt else tod.data
            data_out = tod2map(data_out, data, tod.x, tod.y, self.wcs)
        wcsmap_out = self.copy(deep=False)
        wcsmap_out.data = data_out
        return wcsmap_out

    def make_hits(self, todvec: TODVec, use_filt: bool = True) -> Self:
        """
        Accumulate a bilinearly weighted coverage map.

        Parameters
        ----------
        todvec : TODVec
            TODs containing the pointing information.
        use_filt : bool, default: True
            If True, apply each TOD's noise model to a vector of ones
            before accumulation.

        Returns
        -------
        wcsmap_out : WCSMap
            A shallow copy of this map with the accumulated coverage
            weights stored in `ivar`.
        """
        ivar = jnp.zeros_like(self.data)
        for tod in todvec:
            data = jnp.ones_like(tod.data)
            if use_filt:
                data = tod.noise.apply_noise(data)
            ivar = tod2map(ivar, data, tod.x, tod.y, self.wcs)
        wcsmap_out = self.copy(deep=False)
        wcsmap_out.ivar = ivar
        return wcsmap_out

    @cached_property
    def data_filt(self) -> Array:
        """
        Get the map data with the map-space noise model applied.

        Returns
        -------
        data_filt : Array
            Noise-weighted map data. If `self.noise` is None, this is
            a copy of `self.data`.
        """
        if self.noise is None:
            return jnp.copy(self.data)
        return self.noise.apply_noise(self.data)

    def compute_noise(
        self, noise_class: Type[NoiseModel], data: Optional[Array], *args, **kwargs
    ):
        """
        Compute and set the map-space noise model.

        String arguments beginning with ``self`` are evaluated as
        attributes or expressions, preserving the existing API.

        Parameters
        ----------
        noise_class : Type[NoiseModel]
            Noise model class to instantiate.
        data : Array or None
            Data used to compute the noise model. Defaults to `self.data`.
        *args
            Additional positional arguments for `noise_class.compute`.
        **kwargs
            Additional keyword arguments for `noise_class.compute`.
        """
        args = [
            eval(arg) if isinstance(arg, str) and arg[:4] == "self" else arg
            for arg in args
        ]
        kwargs = {
            k: eval(v) if isinstance(v, str) and v[:4] == "self" else v
            for k, v in kwargs.items()
        }

        self.__dict__.pop("data_filt", None)

        if data is None:
            data = self.data

        self.noise = noise_class.compute(data, *args, **kwargs)

    def recompute_noise(self, data: Optional[Array], *args, **kwargs):
        """
        Recompute the current map-space noise model.

        Parameters
        ----------
        data : Array or None
            Data used to compute the noise model. Defaults to `self.data`.
        *args
            Additional positional arguments for the noise model.
        **kwargs
            Additional keyword arguments for the noise model.
        """
        self.compute_noise(self.noise.__class__, data, *args, **kwargs)

    @jit
    def reduce(self) -> Self:
        """
        MPI-reduce the map data and inverse-variance map.

        Returns
        -------
        WCSMap
            This map with globally summed data and inverse variance.
        """
        self.data = mpi4jax.allreduce(self.data, op=MPI.SUM, comm=self.comm)
        self.ivar = mpi4jax.allreduce(self.ivar, op=MPI.SUM, comm=self.comm)
        return self

    @classmethod
    def empty(
        cls,
        *,
        name: str,
        comm: MPI.Intracomm,
        wcs: WCS,
        lims: tuple[float, float, float, float],
        pad: int = 0,
        square: bool = False,
        **_,
    ) -> Self:
        """
        Initialize an empty map using an Astropy WCS.

        Parameters
        ----------
        name : str
            Name of the map.
        comm : MPI.Intracomm
            MPI communicator associated with the map.
        wcs : WCS
            Astropy WCS describing the desired pixelization.
        lims : tuple of float
            Sky bounds in radians, ordered as
            (RA low, RA high, Dec low, Dec high).
        pad : int, default: 0
            Number of extra pixels to add to each map dimension.
        square : bool, default: False
            If True, make both map dimensions equal.

        Returns
        -------
        WCSMap
            An empty map using a JAX-compatible WCS projector.
        """
        corners = np.array(
            [
                [lims[0], lims[2]],
                [lims[0], lims[3]],
                [lims[1], lims[2]],
                [lims[1], lims[3]],
            ]
        ) * (180.0 / np.pi)

        # Astropy is used only during construction, outside JIT.
        pix_corners = np.asarray(wcs.wcs_world2pix(corners, 0))

        if not np.all(np.isfinite(pix_corners)):
            raise ValueError("Map corners project to invalid pixel coordinates.")

        # Include the full range of pixels touched by the corner bounds.
        lower = np.floor(pix_corners.min(axis=0)).astype(int)
        upper = np.ceil(pix_corners.max(axis=0)).astype(int)

        nx = int(upper[0] - lower[0] + 1 + 2 * pad)
        ny = int(upper[1] - lower[1] + 1 + 2 * pad)

        if nx <= 0 or ny <= 0:
            raise ValueError(f"Invalid map dimensions: {(nx, ny)}")

        if square:
            nx = ny = max(nx, ny)

        data = jnp.zeros((nx, ny))
        jax_wcs = JAXWCS.from_astropy(wcs)

        return cls(name, data, comm, jax_wcs)

    def _self_check(self, other: Self):
        if type(self.wcs) is not type(other.wcs):
            raise ValueError("Cannot operate on WCSMaps with different projections.")

        for a, b in zip(tree_leaves(self.wcs), tree_leaves(other.wcs)):
            if not np.allclose(np.asarray(a), np.asarray(b)):
                raise ValueError("Cannot operate on WCSMaps with different WCSs.")

    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (self.data, self.ivar, self.noise, self.wcs)
        aux_data = (self.name, self.comm)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        name, comm = aux_data
        data, ivar, noise, wcs = children
        return cls(name, data, comm, wcs, noise, ivar)
