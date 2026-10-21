from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from astropy.wcs import WCS
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float


@register_pytree_node_class
@dataclass(frozen=True)
class JAXWCS:
    """Base class for JAX-compatible WCS projections."""

    crpix: Float[Array, "2"]
    crval: Float[Array, "2"]
    cdelt: Float[Array, "2"]
    pc: Float[Array, "2 2"]

    _registry: ClassVar[dict[str, type["JAXWCS"]]] = {}

    @classmethod
    def register_projection(cls, *codes: str):
        """
        Register a WCS subclass for one or more projection codes.

        Parameters
        ----------
        *codes : str
            FITS projection codes, such as ``"CAR"`` or ``"TAN"``.

        Returns
        -------
        Callable
            A class decorator that registers the projection.
        """

        def decorator(subclass: type["JAXWCS"]):
            for code in codes:
                key = code.upper()
                if key in cls._registry:
                    raise ValueError(f"Projection {key!r} is already registered.")
                cls._registry[key] = subclass
            return subclass

        return decorator

    @classmethod
    def from_astropy(cls, wcs: WCS) -> "JAXWCS":
        """
        Construct a JAX WCS implementation from an Astropy WCS.

        Parameters
        ----------
        wcs : WCS
            The Astropy WCS object to convert.

        Returns
        -------
        JAXWCS
            A projector selected from the registered projection classes.

        Raises
        ------
        ValueError
            If the WCS does not have exactly two axes.
        NotImplementedError
            If its projection is not registered or its axis ordering
            is unsupported by the selected implementation.
        """
        ctype = tuple(str(c).upper() for c in wcs.wcs.ctype)

        if len(ctype) != 2:
            raise ValueError("Only 2D WCS objects are supported.")

        # Extract the FITS projection code from each CTYPE
        codes = tuple(axis[-3:] for axis in ctype)

        if codes[0] != codes[1]:
            raise NotImplementedError(f"Mixed projections are unsupported: {ctype}.")

        code = codes[0]
        subclass = cls._registry.get(code)

        if subclass is None:
            supported = ", ".join(sorted(cls._registry)) or "(none)"
            raise NotImplementedError(
                f"Unsupported projection {code!r}. "
                f"Registered projections: {supported}."
            )

        return subclass._from_astropy(wcs)

    @classmethod
    def _from_astropy(cls, wcs: WCS) -> "JAXWCS":
        return cls(**cls._from_astropy_parameters(wcs))

    @classmethod
    def _from_astropy_parameters(cls, wcs: WCS) -> dict:
        """
        Extract common linear WCS parameters.

        Parameters
        ----------
        wcs : WCS
            The input Astropy WCS.

        Returns
        -------
        dict
            Constructor parameters shared by projection subclasses.
        """
        return {
            "crpix": jnp.asarray(wcs.wcs.crpix[:2]),
            "crval": jnp.asarray(wcs.wcs.crval[:2]),
            "cdelt": jnp.asarray(wcs.wcs.cdelt[:2]),
            "pc": jnp.asarray(wcs.wcs.get_pc()[:2, :2]),
        }

    def tree_flatten(self):
        children = (self.crpix, self.crval, self.cdelt, self.pc)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)

    @jax.jit
    def world_to_pixel(
        self,
        ra_rad: Float[Array, "..."],
        dec_rad: Float[Array, "..."],
    ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        _ = ra_rad, dec_rad
        raise NotImplementedError

    @jax.jit
    def pixel_to_world(
        self,
        xpix: Float[Array, "..."],
        ypix: Float[Array, "..."],
    ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        _ = xpix, ypix
        raise NotImplementedError


@register_pytree_node_class
@JAXWCS.register_projection("CAR")
@dataclass(frozen=True)
class JAXCAR(JAXWCS):
    """
    JAX implementation of the CAR celestial map projection.

    This implementation handles the linear CAR case with RA and Dec
    as the first and second world axes, respectively. The projection
    parameters are extracted from an Astropy WCS during construction.

    Pixel coordinates returned by this class are zero-based, consistent
    with NumPy indexing.

    This implementation does not implement the full celestial-to-native
    spherical rotation for every legal FITS CAR WCS. Validate against
    Astropy before using it with a new WCS header.

    Attributes
    ----------
    crpix : Float[Array, "2"]
        Reference pixel coordinates in FITS one-based convention.
    crval : Float[Array, "2"]
        Reference right ascension and declination in degrees.
    cdelt : Float[Array, "2"]
        Coordinate increments in degrees per pixel.
    pc : Float[Array, "2 2"]
        Linear transformation matrix.
    """

    projection: ClassVar[str] = "CAR"

    @classmethod
    def _from_astropy(cls, wcs: WCS) -> "JAXCAR":
        ctype = tuple(str(c).upper() for c in wcs.wcs.ctype)

        if not (ctype[0].startswith("RA---CAR") and ctype[1].startswith("DEC--CAR")):
            raise NotImplementedError(f"Unsupported CAR axis ordering: {ctype}.")

        if (
            getattr(wcs, "sip", None) is not None
            or getattr(wcs, "cpdis1", None) is not None
            or getattr(wcs, "cpdis2", None) is not None
            or getattr(wcs, "det2im1", None) is not None
            or getattr(wcs, "det2im2", None) is not None
        ):
            raise NotImplementedError("CAR distortion terms are not supported.")
        if not np.isclose(wcs.wcs.lonpole, 180.0):
            raise NotImplementedError("Nonstandard LONPOLE is unsupported.")
        if not np.isclose(wcs.wcs.latpole, 90.0):
            raise NotImplementedError("Nonstandard LATPOLE is unsupported.")

        return cls(**cls._from_astropy_parameters(wcs))

    @jax.jit
    def world_to_pixel(
        self,
        ra_rad: Float[Array, "..."],
        dec_rad: Float[Array, "..."],
    ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        r"""
        Convert celestial coordinates to CAR pixel coordinates.
        The linear transformation follows the FITS convention:

        $$
        \mathrm{intermediate} = \mathrm{PC} \mathrm{diag}(\mathrm{CDELT})(\mathrm{pixel} - \mathrm{CRPIX})
        $$

        Longitude differences are wrapped into the interval [-180, 180) degrees.

        Parameters
        ----------
        ra_rad : Float[Array, "..."]
            Right ascension in radians.
        dec_rad : Float[Array, "..."]
            Declination in radians. Must have the same shape as `ra_rad`.

        Returns
        -------
        xpix : Float[Array, "..."]
            Zero-based pixel coordinates along the first map axis.
        ypix : Float[Array, "..."]
            Zero-based pixel coordinates along the second map axis.
        """
        ra = jnp.rad2deg(ra_rad)
        dec = jnp.rad2deg(dec_rad)

        dra = (ra - self.crval[0] + 180.0) % 360.0 - 180.0
        ddec = dec - self.crval[1]

        intermediate = jnp.stack([dra, ddec], axis=0)

        scaled = intermediate / self.cdelt.reshape((2,) + (1,) * ra.ndim)

        offset = jnp.einsum(
            "ij,j...->i...",
            jnp.linalg.inv(self.pc),
            scaled,
        )

        pix = offset + self.crpix.reshape((2,) + (1,) * ra.ndim) - 1.0

        return pix[0], pix[1]

    @jax.jit
    def pixel_to_world(
        self,
        xpix: Float[Array, "..."],
        ypix: Float[Array, "..."],
    ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        """
        Convert CAR pixel coordinates to celestial coordinates.
        This is the inverse of the linear CAR transformation implemented
        by `world_to_pixel`. It does not perform a general spherical
        celestial-to-native rotation.

        Parameters
        ----------
        xpix : Float[Array, "..."]
            Zero-based pixel coordinates along the first map axis.
        ypix : Float[Array, "..."]
            Zero-based pixel coordinates along the second map axis.

        Returns
        -------
        ra_rad : Float[Array, "..."]
            Right ascension in radians, wrapped into [0, 2 pi).
        dec_rad : Float[Array, "..."]
            Declination in radians.
        """
        pix = jnp.stack([xpix, ypix], axis=0)

        offset = pix + 1.0 - self.crpix.reshape((2,) + (1,) * xpix.ndim)

        intermediate = jnp.einsum(
            "ij,j...->i...",
            self.pc,
            offset * self.cdelt.reshape((2,) + (1,) * xpix.ndim),
        )

        ra = self.crval[0] + intermediate[0]
        dec = self.crval[1] + intermediate[1]

        return jnp.deg2rad(ra) % (2.0 * jnp.pi), jnp.deg2rad(dec)
