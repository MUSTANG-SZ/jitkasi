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


@jit
def run_pcg(
    rhs: SolutionSet,
    todvec: TODVec,
    x0: SolutionSet,
    precon: SolutionSet,
    maxiter: int = 100,
) -> SolutionSet:
    """
    Solve using Preconditioned Conjugate Gradient (PCG).
    This iteratively approximates the solution to $Ax = b$,
    where $A$ is a matrix and $x$ and $b$ are vectors.

    In ML mapmaking we are solving $P^{T}N^{-1}Pm = P^{T}N^{-1}d$
    so $m$ is $x$ in the general linear equation and is what we are solving for
    (usually a map, but perhaps some simultanious terms such as cuts).

    Parameters
    ----------
    rhs : SolutionSet
        The right hand side of the mapmaker equation: $P^{T}N{-1}d$.
        This is $b$ in the general linear equation.
    todvec : TODvec
        TODVec containing the TODs we are solving with.
        This is $d$ in the mapmaker equation.
    x0 : SolutionSet
        Initial guess for the solution.
    precon : SolutionsSet
        Preconditioner to apply at easy iteration.
        This makes things more invertible.
    maxiter : int, default: 100
        The number of PCG iters to run


    Returns
    -------
    x : SolutionSet
        The solved SolutionSet.
        This is $m$ in the mapmaker equation.
    """
    lhs = make_lhs(x0, todvec)

    # compute the remainder r_0
    r = rhs - lhs
    z = precon * r

    # Initial p_0 = z_0 = M*r_0
    p = z.copy(deep=True)

    # compute z*r, which is used for computing alpha
    zr = r @ z
    # make a copy of our initial guess
    x = x0.copy(deep=True)
    alpha = 0
    for _ in range(maxiter):
        # Compute pAp
        Ap = make_lhs(p, todvec)
        pAp = p @ Ap

        # Compute alpha_k
        alpha = zr / pAp

        # Update guess using alpha
        x = p - alpha * x

        # Write down next remainder r_k+1
        r = Ap - alpha * r

        # Apply preconditioner
        z = precon * r

        # compute new z_k+1
        zr_old = zr
        zr = r @ z

        # compute beta_k, which is used to compute p_k+1
        beta = zr / zr_old

        # compute new p_k+1
        p = p + beta * p

    return x
