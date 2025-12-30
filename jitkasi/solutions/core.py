from copy import copy, deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Iterator, Optional

import jax.numpy as jnp
import mpi4jax
from jax import Array, jit
from jax.tree_util import register_pytree_node_class
from mpi4py import MPI
from typing_extensions import Self

from ..tod import TODVec


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
    name : str
        An identifying string for this solution.
    data : Array
        The model data we are solving for.
        For example if we are solving for a map then this is the map,
        if we are solving for cuts then this is the modeled offsets.
        Mathematically this is $m$ in $d = Pm + n$.
        This should always be a child of the pytree.
    comm : MPI.Intracomm
        The MPI communicator to use.
    """

    name: str
    data: Array
    comm: MPI.Intracomm

    @jit
    def to_tods(self, todvec: TODVec) -> TODVec:  # type: ignore
        _ = todvec
        pass

    @jit
    def from_tods(self, todvec: TODVec, use_filt: bool = True) -> Self:  # type: ignore
        _ = (todvec, use_filt)
        pass

    @classmethod
    def empty(cls, **_) -> Self:  # type: ignore
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
            except for `data` which will be a copy.

        Returns
        -------
        copy : Solution
            A copy of this Solution.
        """
        if deep:
            return deepcopy(self)
        else:
            new = copy(self)
            new.data = new.data.copy()
            return new

    # Math functions
    def _self_check(self, other: Self):
        _ = other
        pass

    def _get_to_op(self, other: Any) -> Array | float:
        if isinstance(other, type(self)):
            to_op = other.data
            self._self_check(other)
        elif isinstance(other, (float, int)):
            to_op = float(other)
        else:
            raise TypeError(f"Cannot use type {type(other)} to operate on {type(self)}")
        return to_op

    def __iadd__(self, other: Self | float) -> Self:
        to_add = self._get_to_op(other)
        self.data = _iadd(self.data, to_add)
        return self

    def __add__(self, other: Self | float) -> Self:
        to_ret = self.copy()
        to_ret += other
        return to_ret

    def __radd__(self, other: Self | float) -> Self:
        return self.__add__(other)

    def __isub__(self, other: Any) -> Self:
        to_sub = self._get_to_op(other)
        self.data = _isub(self.data, to_sub)
        return self

    def __sub__(self, other: Any) -> Self:
        to_ret = self.copy()
        to_ret -= other
        return to_ret

    def __rsub__(self, other: Any) -> Self:
        return self.__sub__(other)

    def __imul__(self, other: Any) -> Self:
        to_sub = self._get_to_op(other)
        self.data = _imul(self.data, to_sub)
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
        return float(jnp.sum(self.data * other.data, axis=None))

    @jit
    def reduce(self) -> Self:
        """
        MPI reduce the solution.
        This adds up all the data in all insances of this solution.
        """
        dat_sum = mpi4jax.allreduce(self.data, op=MPI.SUM, comm=self.comm)
        self.data = dat_sum
        return self

    # Functions for making this a pytree
    # Don't call this on your own
    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = (self.data,)
        aux_data = (self.name, self.comm)

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        name, comm = aux_data
        return cls(name, children[0], comm)


@register_pytree_node_class
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
    comm : MPI.Intracomm
        The MPI communicator to use.
    """

    solutions: list[Solution] = field(default_factory=list)
    comm: MPI.Intracomm = field(default_factory=MPI.COMM_WORLD.Clone)

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
            except for data reference the same memory as before.
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
    def tree_flatten(self) -> tuple[tuple, tuple]:
        children = tuple(self.solutions)
        aux_data = (self.comm,)

        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        return cls(list(children), *aux_data)
