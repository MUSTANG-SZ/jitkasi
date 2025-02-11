from copy import copy, deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterator, Optional

import jax.numpy as jnp
import mpi4jax
from jax import Array, jit
from jax.tree_util import register_pytree_node_class
from mpi4py import MPI
from typing_extensions import Self

from . import noise as n


@register_pytree_node_class
@dataclass
class TOD:
    """
    Class for storing time ordered data.
    This class is a registered pytree so it is JITable with JAX.

    Attributes
    ----------
    data : Array
        The detector data for this TOD.
        This is a child of the pytree.
    x : Array
        The x sky coordinates for this TOD.
        Nominally this should be RA in radians.
        Note that eventually this may become a cached property
        to enable on the fly pointing reconstruction for large expiriments.
        This is a child of the pytree.
    y : Array
        The y sky coordinates for this TOD.
        Nominally this should be Dec in radians.
        Note that eventually this may become a cached property
        to enable on the fly pointing reconstruction for large expiriments.
        This is a child of the pytree.
    noise : Optional[NoiseModel]
        The noise model for this TOD.
        Should be None when no model is initialized.
        This is aux data for the pytree.
    meta : dict
        Additional metadata associated with the TOD.
        This is supplied **only** for user convenience and
        should **not** be relied on in any `jitkasi` library code.
        It is intended to be used in user made scripts only.
        This is aux data for the pytree
    """

    data: Array
    x: Array
    y: Array
    noise: n.NoiseModel = field(default_factory=n.NoiseI)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        # Check that all sizes are the same
        shapes = [self.data.shape, self.x.shape, self.y.shape]
        if not (
            all(s[0] == shapes[0][0] for s in shapes)
            and all(s[1] == shapes[0][1] for s in shapes)
        ):
            raise ValueError(
                f"Expected 'data', 'x', and 'y' to have the same shape but got {shapes}."
            )

        # Now make everything jax arrays in case we were given numpy arrays
        self.data = jnp.array(self.data)
        self.x = jnp.array(self.x)
        self.y = jnp.array(self.y)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Get the shape of the TOD's data.

        Returns
        -------
        shape : tuple[int, ....]
            The shape of the TOD.
        """
        return self.data.shape

    @property
    def lims(self) -> Array:
        """
        Get the limits of the TODs coordinates.

        Returns
        -------
        lims : Array
            The limits in the order:
            (x min, x max, y min, y max).
        """
        return jnp.array(
            (jnp.min(self.x), jnp.max(self.x), jnp.min(self.y), jnp.max(self.y))
        )

    @cached_property
    def data_filt(self) -> Array:
        """
        Get a copy of the data with the noise model applied.
        This is essentially $N^{-1}d$

        Returns
        -------
        data_filt : Array
            The filtered data.
            If `self.noise` is None then this is just a copy of `self.data`.
        """
        if self.noise is None:
            return jnp.copy(self.data)
        return self.noise.apply_noise(self.data)

    def copy(self, deep: bool = False) -> Self:
        """
        Return of copy of the TOD.

        Parameters
        ----------
        deep : bool, default: False
            If True do a deepcopy so that all contained
            data is also copied. Otherwise a shallow copy is
            made and the new TOD will reference the same arrays.

        Returns
        -------
        copy : TOD
            A copy of this TOD.
        """
        if deep:
            return deepcopy(self)
        return copy(self)

    def compute_noise(
        self, noise_class: n.NoiseModel, data: Optional[Array], *args, **kwargs
    ):
        """
        Compute and set the noise model for this TOD.
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
            Note that any argunment that is a string that starts with `self` will be evaled.
        *kwargs
            Additional keyword arguments to pass to `noise_class.compute`.
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

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (self.data, self.x, self.y)
        aux_data = (self.noise, self.meta)

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        return cls(*children, *aux_data)


@register_pytree_node_class
@dataclass
class TODVec:
    """
    Class to store collections of TODs.
    Eventually this will be responsible for handling most
    collective (and MPI aware) operations on TODs.

    Attributes
    ----------
    tods : list[TOD]
        The TODs that belong to this TODVec.
        Indexing and interating the TODVec operates on this.
    comm : MPI.Intracomm
        The MPI communicator to use.
    """

    tods: list[TOD] = field(default_factory=list)
    comm: MPI.Intracomm = field(default_factory=MPI.COMM_WORLD.Clone)

    @property
    @jit
    def nsamp(self) -> int:
        nsamp = 0
        for tod in self.tods:
            nsamp += int(jnp.product(jnp.array(tod.shape), axis=None))
        return nsamp

    @property
    def lims(self) -> Array:
        """
        Get the global limits of all the TODs coordinates.
        Will eventually be MPI aware.

        Returns
        -------
        lims : Array
            The limits in the order:
            (x min, x max, y min, y max).
        """
        all_lims = [tod.lims for tod in self.tods]
        all_lims = jnp.array(all_lims).reshape(-1, 4)
        # Order not important here, no need for tokens
        x0, _ = mpi4jax.allreduce(jnp.min(all_lims[:, 0]), MPI.MIN, comm=self.comm)
        x1, _ = mpi4jax.allreduce(jnp.max(all_lims[:, 1]), MPI.MAX, comm=self.comm)
        y0, _ = mpi4jax.allreduce(jnp.min(all_lims[:, 2]), MPI.MIN, comm=self.comm)
        y1, _ = mpi4jax.allreduce(jnp.max(all_lims[:, 3]), MPI.MAX, comm=self.comm)
        lims = jnp.array((x0, x1, y0, y1))
        return jnp.ravel(lims)

    def copy(self, deep: bool = False, copy_comm=False) -> Self:
        """
        Return of copy of the TODVec.

        Parameters
        ----------
        deep : bool, default: False
            If True do a deepcopy so that all contained
            TODs are also copied. Otherwise a shallow copy is
            made and the new TODVec will reference the same TODs.

        Returns
        -------
        copy : TODVec
            A copy of this TODVec.
        """
        comm = self.comm
        if copy_comm:
            comm = MPI.COMM_WORLD.Clone()
        if deep:
            tods = deepcopy(self.tods)
        else:
            tods = copy(self.tods)
        return self.__class__(tods, comm)

    # Functions to make this list like
    def __getitem__(self, key: int) -> TOD:
        if not isinstance(key, int):
            raise TypeError("TODVec is indexed by ints")
        return self.tods[key]

    def __setitem__(self, key: int, value: TOD):
        if not isinstance(key, int):
            raise TypeError("TODVec is indexed by ints")
        if not isinstance(value, TOD):
            raise TypeError("TODVec can only store instances of TOD")

        self.tods[key] = value

    def __delitem__(self, key: int):
        if not isinstance(key, int):
            raise TypeError("TODVec is indexed by ints")
        del self.tods[key]

    def __iter__(self) -> Iterator[TOD]:
        return self.tods.__iter__()

    def __add__(self, other: Self) -> Self:
        if not isinstance(other, TODVec):
            raise TypeError("Can only add other TODVecs to a TODVec")
        new = self.copy()
        new += other
        return new

    def __radd__(self, other: Self) -> Self:
        if not isinstance(other, TODVec):
            raise TypeError("Can only add other TODVecs to a TODVec")
        new = other.copy()
        new += self
        return new

    def __iadd__(self, other: Self):
        if not isinstance(other, TODVec):
            raise TypeError("Can only add other TODVecs to a TODVec")
        self.tods += other.tods
        return self

    def insert(self, key: int, value: TOD):
        if not isinstance(key, int):
            raise TypeError("TODVec is indexed by ints")
        if not isinstance(value, TOD):
            raise TypeError("TODVec can only store instances of TOD")

        self.tods.insert(key, value)

    def append(self, value):
        if not isinstance(value, TOD):
            raise TypeError("TODVec can only store instances of TOD")

        self.tods.append(value)

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = tuple(self.tods)
        aux_data = (self.comm,)

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        return cls(list(children), *aux_data)
