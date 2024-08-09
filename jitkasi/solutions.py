from copy import copy, deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Iterator, Optional

import jax.numpy as jnp
import numpy as np
from astropy.wcs import WCS
from jax import Array, jit
from jax.tree_util import register_pytree_node_class
from typing_extensions import Self

from .tod import TOD, TODVec


# TODO: Figure out math for cuts and jumps
@partial(jit, donate_argnames=["val1"])
def _iadd(val1, val2):
    val1 = val1.at[:].add(val2)
    return val1


@partial(jit, donate_argnames=["val1"])
def _isub(val1, val2):
    val1 = val1.at[:].add(-1.0 * val2)
    return val1


@partial(jit, donate_argnames=["val1"])
def _imul(val1, val2):
    val1 = val1.at[:].multiply(val2)
    return val1


@register_pytree_node_class
@dataclass
class Solution:
    """
    Base class defining required functionality for a solution.
    For mapmaking all that we really need is code to project to and from TODs,
    additional helper functions are probably needed but anything that needs to
    be included in a mapmaking script explicitly should be clearly documented.

    This class is non functional but all actual solutions (maps, cuts, etc.) should
    subclass it and ensure that the required functions are implemented.

    Attributes
    ----------
    params : Array
        The model params we are solving for.
        For example if we are solving for a map then this is the map,
        if we are solving for cuts then this is the modeled offsets.
        Mathematically this is $m$ in $d = Pm + n$.
        This should always be a child of the pytree.
    """

    params: Array

    @jit
    def to_tods(self, todvec: TODVec) -> TODVec:  # type: ignore
        pass

    @jit
    def from_tods(self, todvec: TODVec, use_filt: bool = True) -> Self:  # type: ignore
        pass

    @classmethod
    def empty(cls, **kwargs) -> Self:  # type: ignore
        pass

    def copy(self, deep: bool = False) -> Self:
        """
        Return of copy of the Solution.

        Parameters
        ----------
        deep : bool, default: False
            If True do a deepcopy so that all contained
            data is also copied. Otherwise a mostly shallow copy is
            made and the new Solution will reference the same objects
            except for `params` which will be a copy.

        Returns
        -------
        copy : Solution
            A copy of this Solution.
        """
        if deep:
            return deepcopy(self)
        else:
            new = copy(self)
            new.params = new.params.copy()
            return new

    # Math functions
    def _self_check(self, other: Self):
        pass

    def _get_to_op(self, other: Any) -> Array | float:
        if isinstance(other, type(self)):
            to_op = other.params
            self._self_check(other)
        elif isinstance(other, (float, int)):
            to_op = float(other)
        else:
            raise TypeError(f"Cannot use type {type(other)} to operate on {type(self)}")
        return to_op

    def __iadd__(self, other: Self | float) -> Self:
        to_add = self._get_to_op(other)
        self.params = _iadd(self.params, to_add)
        return self

    def __add__(self, other: Self | float) -> Self:
        to_ret = self.copy()
        to_ret += other
        return to_ret

    def __radd__(self, other: Self | float) -> Self:
        return self.__add__(other)

    def __isub__(self, other: Any) -> Self:
        to_sub = self._get_to_op(other)
        self.params = _isub(self.params, to_sub)
        return self

    def __sub__(self, other: Any) -> Self:
        to_ret = self.copy()
        to_ret -= other
        return to_ret

    def __rsub__(self, other: Any) -> Self:
        return self.__sub__(other)

    def __imul__(self, other: Any) -> Self:
        to_sub = self._get_to_op(other)
        self.params = _imul(self.params, to_sub)
        return self

    def __mul__(self, other: Any) -> Self:
        to_ret = self.copy()
        to_ret *= other
        return to_ret

    def __rmul__(self, other: Any) -> Self:
        return self.__mul__(other)

    @jit
    def __matmul__(self, other: Self) -> float:
        if not isinstance(other, type(self)):
            raise ValueError(f"Can't dot {type(other)} and {type(self)}")
        return float(jnp.sum(self.params * other.params, axis=None))

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, Optional[tuple]]:
        children = (self.params,)
        aux_data = None

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        return cls(*children)


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
                tod.data = tod.data.at[:].add(
                    self.params.at[pix].get(mode="fill", fill_value=0.0)
                )
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
            except for params reference the same memory as this object.
        """
        wcsmap_out = self.copy(deep=False)
        wcsmap_out.params = jnp.zeros_like(self.params)
        for tod in todvec:
            pix = self._pix_reg[self.pixelization](tod)
            data = tod.data
            if use_filt:
                data = tod.data_filt
            if self.pixelization == "nn":
                wcsmap_out.params = wcsmap_out.params.at[pix].add(
                    data.at[:].get(), mode="drop"
                )
        return wcsmap_out

    @classmethod
    def empty(
        cls,
        wcs: WCS,
        lims: tuple[float, float, float, float],
        pad=0,
        square=False,
        pixelization="nn",
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
        children = (self.params,)
        aux_data = (self.wcs, self.pixelization)

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        return cls(*children, *aux_data)


@dataclass
class SolutionSet:
    """
    Class to store collections of Solutions.
    Eventually this will be responsible for handling most
    collective (and MPI aware) operations on Solutions.

    Note that while this can mostly be treated like a list, '+' acts
    in a math like fasion adding Solutions within two SolutionSets together
    rather than appending them (unlike TODVec which does append).

    Attributes
    ----------
    solutions : list[Solution]
        The solutions that belong to this SolutionSet.
        Indexing and interating the SolutionSet operates on this.
    """

    solutions: list[Solution] = field(default_factory=list)

    def copy(self, deep: bool = False) -> Self:
        """
        Return of copy of the SolutionSet.

        Parameters
        ----------
        deep : bool, default: False
            If True do a deepcopy so that all contained
            Solutions are also copied. Otherwise a shallow copy is
            made and the new SolutionSet will reference the same Solutions.

        Returns
        -------
        copy : TODVec
            A copy of this TODVec.
        """
        if deep:
            return deepcopy(self)
        return copy(self)

    @jit
    def to_tods(self, todvec: TODVec) -> TODVec:
        """
        Project the all the Solutions into the same set of TODs.

        Parameters
        ----------
        todvec : TODVec
            TODVec containing TODs to project into.
            Only used for pointing and shape information,
            not modified in place.

        Returns
        -------
        todvec_out : TODVec
            A TODvec each with the Solutions projected into it.
            The order of TODs here is the same as the input TODVec
            and the TODs within are shallow copied to the non-data
            Arrays reference the same memory as the original TODs.
        """
        todvec_out = todvec.copy(deep=False)
        for tod in todvec_out:
            tod.data = jnp.zeros_like(tod.data)
        for solution in self:
            tmp = solution.to_tods(todvec)
            for todout, todin in zip(todvec_out, tmp):
                todout.data = todout.data.at[:].add(todin.data.at[:].get())
        return todvec_out

    @jit
    def from_tods(self, todvec: TODVec, use_filt: bool = True) -> Self:
        """
        Project TODs into the solutions.

        Parameters
        ----------
        todvec : TODVec
            TODVec containing TODs to project from.
        use_filt : bool, default: True
            If True use data_filt instead of data.

        Returns
        -------
        solutionset_out : SolutionSet
            A SolutionSet with the TODs projected into each Solution.
            The new Solutions are shallow copies of the current ones so all Arrays
            except for params reference the same memory as before.
        """
        solutionset_out = self.copy(deep=False)
        for i, solution in enumerate(self):
            solutionset_out[i] = solution.from_tods(todvec, use_filt)
        return solutionset_out

    # Math functions
    def __add__(self, other: Self) -> Self:
        if len(self.solutions) != other.solutions:
            raise ValueError(
                "SolutionSets can only be added if they contain the same number of solutions"
            )
        summed = self.copy(deep=False)
        for i in range(len(self.solutions)):
            summed[i] = self[i] + other[i]
        return summed

    def __sub__(self, other: Self) -> Self:
        if len(self.solutions) != other.solutions:
            raise ValueError(
                "SolutionSets can only be subtracted if they contain the same number of solutions"
            )
        subbed = self.copy(deep=False)
        for i in range(len(self.solutions)):
            subbed[i] = self[i] - other[i]
        return subbed

    def __mul__(self, other: Self) -> Self:
        if len(self.solutions) != other.solutions:
            raise ValueError(
                "SolutionSets can only be multiplied if they contain the same number of solutions"
            )
        product = self.copy(deep=False)
        for i in range(len(self.solutions)):
            product[i] = self[i] * other[i]
        return product

    @jit
    def __matmul__(self, other: Self) -> float:
        if len(self.solutions) != other.solutions:
            raise ValueError(
                "SolutionSets can only be dotted if they contain the same number of solutions"
            )
        tot = 0.0
        for lsol, rsol in zip(self, other):
            tot += lsol @ rsol
        return tot

    # Functions to make this list like
    def __getitem__(self, key: int) -> Solution:
        if not isinstance(key, int):
            raise TypeError("SolutionSet is indexed by ints")
        return self.solutions[key]

    def __setitem__(self, key: int, value: Solution):
        if not isinstance(key, int):
            raise TypeError("SolutionSet is indexed by ints")
        if not isinstance(value, Solution):
            raise TypeError("SolutionSet can only store instances of Solution")

        self.solutions[key] = value

    def __delitem__(self, key: int):
        if not isinstance(key, int):
            raise TypeError("SolutionSet is indexed by ints")
        del self.solutions[key]

    def __iter__(self) -> Iterator[Solution]:
        return self.solutions.__iter__()

    def insert(self, key: int, value: Solution):
        if not isinstance(key, int):
            raise TypeError("SolutionSet is indexed by ints")
        if not isinstance(value, Solution):
            raise TypeError("SolutionSet can only store instances of Solution")

        self.solutions.insert(key, value)

    def append(self, value):
        if not isinstance(value, Solution):
            raise TypeError("SolutionSet can only store instances of Solution")

        self.solutions.append(value)

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, None]:
        children = tuple(self.solutions)
        aux_data = None

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        return cls(list(children))
