"""
Core mapmaking functions.
"""

from functools import partial

from jax import jit, lax

from .solutions import SolutionSet
from .tod import TODVec


@jit
def make_lhs(solutionset: SolutionSet, todvec: TODVec) -> SolutionSet:
    r"""
    Make the left-hand side of the mapmaker equation: $P^T N^{-1} P m$.

    Parameters
    ----------
    solutionset : SolutionSet
        Current mapmaking vector. Not modified in place.
    todvec : TODVec
        TODs providing pointing and noise information.

    Returns
    -------
    lhs : SolutionSet
        The result of applying P.T N^-1 P to `solutionset`.
    """
    projected = solutionset.to_tods(todvec)
    return solutionset.from_tods(projected, use_filt=True)


@jit
def make_rhs(solutionset: SolutionSet, todvec: TODVec) -> SolutionSet:
    r"""
    Make the right-hand side of the mapmaker equation: $P^T N^{-1} d$.

    Parameters
    ----------
    solutionset : SolutionSet
        Solution templates defining the unknowns.
    todvec : TODVec
        TODs containing the observed data and noise models.

    Returns
    -------
    rhs : SolutionSet
        The noise-weighted data projected into solution space.
    """
    return solutionset.from_tods(todvec, use_filt=True)


def _pcg_step(step, pars):
    r"""
    Perform one preconditioned conjugate-gradient iteration.

    The system solved is $A x = b$, where $A$ is assumed to be
    symmetric positive definite and the preconditioner represents
    an approximation to $A$ inverse.
    """
    _ = step

    todvec, precon, p, x, r, z, rz = pars

    # Apply A = P.T N^-1 P to the current search direction.
    Ap = make_lhs(p, todvec)

    # Compute the step length.
    pAp = p @ Ap
    alpha = rz / pAp

    # Update the solution and residual.
    x = x + alpha * p
    r = r - alpha * Ap

    # Apply the inverse preconditioner to the new residual.
    z = precon * r

    # Compute the new residual-preconditioned-residual product.
    rz_new = r @ z

    # Update the search direction.
    beta = rz_new / rz
    p = z + beta * p

    return todvec, precon, p, x, r, z, rz_new


@partial(jit, static_argnames=("maxiter",))
def run_pcg(
    rhs: SolutionSet,
    todvec: TODVec,
    x0: SolutionSet,
    precon: SolutionSet,
    maxiter: int = 100,
) -> SolutionSet:
    r"""
    Solve the mapmaking system using preconditioned conjugate gradients.

    The system we are solving is:

    $$
    P^T N^{-1} P m = P^T N^{-1} d
    $$

    Parameters
    ----------
    rhs : SolutionSet
        Right-hand side, $P^T N^{-1} d$.
    todvec : TODVec
        TODs providing pointing, data, and noise information.
    x0 : SolutionSet
        Initial solution estimate.
    precon : SolutionSet
        Approximation to the inverse system operator. Applied as
        `precon * residual`.
    maxiter : int, default: 100
        Maximum number of iterations.

    Returns
    -------
    x : SolutionSet
        Approximate solution after `maxiter` iterations.
    """
    if maxiter < 0:
        raise ValueError("maxiter must be non-negative.")

    # Initial residual: r_0 = b - A x_0.
    lhs = make_lhs(x0, todvec)
    r = rhs - lhs

    # Apply the inverse preconditioner.
    z = precon * r

    # Initial search direction.
    p = z.copy(deep=True)

    # Initial inner product used to compute alpha and beta.
    rz = r @ z

    # Preserve the initial guess.
    x = x0.copy(deep=True)

    carry = (todvec, precon, p, x, r, z, rz)

    # Fixed iteration count; keep the loop rolled to avoid code
    # expansion for large iteration counts.
    carry = lax.fori_loop(0, maxiter, _pcg_step, carry)

    _, _, _, x, _, _, _ = carry
    return x
