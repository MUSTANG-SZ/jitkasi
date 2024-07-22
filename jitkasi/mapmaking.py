"""
Core mapmaking functions
"""

from jax import jit

from .solutions import SolutionSet
from .tod import TODVec


@jit
def make_lhs(solutionset: SolutionSet, todvec: TODVec) -> SolutionSet:
    """
    Make the left hand side of the map maker equation: $P^{T}N^{-1}Pm$.
    To do this we project the Solutions into TODs, apply the noise model,
    and project back to Solutions.

    Parameters
    ----------
    solutionset : SolutionSet
        SolutionSet containing the things you are trying to solve for.
        This is not modified in place.
    todvec : TODVec
        TODVec containing TODs to project into and from.
        Only used for pointing and shape information,
        not modified in place.

    Returns
    -------
    lhs : SolutionSet
        A SolutionSet for the left hand side of the map maker equation.
    """
    projected = solutionset.to_tods(todvec)
    lhs = solutionset.from_tods(projected, True)
    return lhs


@jit
def make_rhs(solutionset: SolutionSet, todvec: TODVec) -> SolutionSet:
    """
    Make the right hand side of the map maker equation: $P^{T}N^{-1}d$.
    To do this we apply the noise model to some TODs and project into Solutions.

    Parameters
    ----------
    solutionset : SolutionSet
        SolutionSet containing the things you are trying to solve for.
        This is not modified in place.
    todvec : TODVec
        TODVec containing TODs to project from.
        This is not modified in place.

    Returns
    -------
    rhs : SolutionSet
        A SolutionSet for the right hand side of the map maker equation.
    """
    rhs = solutionset.from_tods(todvec, True)
    return rhs
