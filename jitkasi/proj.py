"""
Projection operations.
Note that here we do things with just simple arrays rather than the
container classes to keep things generic. The container classes in
`solutions` have the container specific wrappers.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .wcs import JAXWCS


@jax.jit
def tod2map(
    map_data: Float[Array, "nx ny"],
    tod_data: Float[Array, "..."],
    ra_rad: Float[Array, "..."],
    dec_rad: Float[Array, "..."],
    wcs: JAXWCS,
) -> Float[Array, "nx ny"]:
    """
    Accumulate TOD samples into an existing map using bilinear weights.

    Parameters
    ----------
    map_data : Float[Array, "nx ny"]
        Initial map to which the TOD contributions are added.
    tod_data : Float[Array, "..."]
        TOD samples to accumulate.
    ra_rad : Float[Array, "..."]
        Right ascension of each TOD sample in radians.
    dec_rad : Float[Array, "..."]
        Declination of each TOD sample in radians.
    wcs : JAXWCS
        WCS projector used to convert sky coordinates to pixels.

    Returns
    -------
    map_out : Float[Array, "nx ny"]
        Updated map containing the accumulated TOD contributions.
    """
    x, y = wcs.world_to_pixel(ra_rad, dec_rad)

    x = jnp.ravel(x)
    y = jnp.ravel(y)
    data = jnp.ravel(tod_data)

    valid = jnp.isfinite(x) & jnp.isfinite(y)
    x = jnp.where(valid, x, -1e6)
    y = jnp.where(valid, y, -1e6)
    data = jnp.where(valid, data, 0.0)

    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    dx = x - x0
    dy = y - y0

    w00 = (1.0 - dx) * (1.0 - dy)
    w10 = dx * (1.0 - dy)
    w01 = (1.0 - dx) * dy
    w11 = dx * dy

    out = map_data
    out = out.at[x0, y0].add(data * w00, mode="drop")
    out = out.at[x1, y0].add(data * w10, mode="drop")
    out = out.at[x0, y1].add(data * w01, mode="drop")
    out = out.at[x1, y1].add(data * w11, mode="drop")

    return out


@jax.jit
def map2tod(
    map_data: Float[Array, "nx ny"],
    ra_rad: Float[Array, "..."],
    dec_rad: Float[Array, "..."],
    wcs: JAXWCS,
) -> Float[Array, "..."]:
    """
    Project a map into TOD samples using bilinear interpolation.

    Parameters
    ----------
    map_data : Float[Array, "nx ny"]
        Input map, indexed as map_data[x, y].
    ra_rad : Float[Array, "..."]
        Right ascension of each TOD sample in radians.
    dec_rad : Float[Array, "..."]
        Declination of each TOD sample in radians.
    wcs : JAXWCS
        WCS projector used to convert sky coordinates to pixels.

    Returns
    -------
    tod : Float[Array, "..."]
        Interpolated map values at the TOD pointing positions.
    """
    x, y = wcs.world_to_pixel(ra_rad, dec_rad)

    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    dx = x - x0
    dy = y - y0

    w00 = (1.0 - dx) * (1.0 - dy)
    w10 = dx * (1.0 - dy)
    w01 = (1.0 - dx) * dy
    w11 = dx * dy

    p00 = map_data.at[x0, y0].get(mode="fill", fill_value=0.0)
    p10 = map_data.at[x1, y0].get(mode="fill", fill_value=0.0)
    p01 = map_data.at[x0, y1].get(mode="fill", fill_value=0.0)
    p11 = map_data.at[x1, y1].get(mode="fill", fill_value=0.0)

    tod = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11

    return jnp.where(jnp.isfinite(x) & jnp.isfinite(y), tod, 0.0)
