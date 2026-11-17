from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from astropy.wcs import WCS
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float


def _unit_vector(lon, lat):
    r"""
    Convert spherical coordinates to Cartesian unit vectors.
    The spherical-to-Cartesian transformation is

    $$
    \begin{aligned}
    x &= \cos(\mathrm{lat})\cos(\mathrm{lon}), \\
    y &= \cos(\mathrm{lat})\sin(\mathrm{lon}), \\
    z &= \\sin(\mathrm{lat}).
    \end{aligned}
    $$

    Parameters
    ----------
    lon : Float[Array, "..."]
        Longitude in radians.
    lat : Float[Array, "..."]
        Latitude in radians.

    Returns
    -------
    Float[Array, "3 ..."]
        Cartesian unit vectors, with the Cartesian coordinate axis first.
    """
    cos_lat = jnp.cos(lat)
    return jnp.stack(
        (cos_lat * jnp.cos(lon), cos_lat * jnp.sin(lon), jnp.sin(lat)), axis=0
    )


def _native_to_celestial_matrix(
    crval: Float[Array, "2"],
    lonpole: Float[Array, ""],
    latpole: Float[Array, ""],
    projection: str,
) -> Float[Array, "3 3"]:
    r"""
    Construct the rotation matrix from native to celestial coordinates.
    The rotation matrix transforms a native Cartesian unit vector
    $\boldsymbol{n}$ into a celestial Cartesian unit vector
    $\boldsymbol{c}$:

    $$
    \boldsymbol{c} = R\boldsymbol{n}.
    $$

    The rotation is fixed by two conditions (Calabretta & Greisen 2002, sect. 2):

    1. The native reference point maps to CRVAL. This is native
       $(\phi_0, \theta_0) = (0, 0)$ for CAR and the native pole,
       $(\phi_0, \theta_0) = (0, 90^\circ)$, for zenithal projections such as TAN.
    2. The celestial north pole sits at native coordinates
       $(\phi_p, \theta_p)$ = (`lonpole`, `latpole`).

    Parameters
    ----------
    crval : Float[Array, "2"]
        Reference celestial longitude and latitude in radians.
    lonpole : Float[Array, ""]
        Native longitude of the celestial north pole in radians.
    latpole : Float[Array, ""]
        Native latitude of the celestial north pole in radians.
        This is the resolved LATPOLE for CAR, and $\delta_0$ (the CRVAL
        latitude) for zenithal projections such as TAN.
    projection : str
        Projection code, such as ``"CAR"`` or ``"TAN"``.
        Sets which native vector is mapped to CRVAL.

    Returns
    -------
    Float[Array, "3 3"]
        Rotation matrix mapping native Cartesian coordinates to celestial
        Cartesian coordinates.

    Raises
    ------
    ValueError
        If the projection code is unsupported.
    """
    alpha0, delta0 = crval

    n = _unit_vector(lonpole, latpole)
    r = _unit_vector(alpha0, delta0)

    if projection == "CAR":
        e0 = jnp.array([1.0, 0.0, 0.0], dtype=crval.dtype)
    elif projection == "TAN":
        e0 = jnp.array([0.0, 0.0, 1.0], dtype=crval.dtype)
    else:
        raise ValueError(f"Unsupported projection {projection!r}.")
    z = jnp.array([0.0, 0.0, 1.0], dtype=crval.dtype)

    native_dot = jnp.dot(e0, n)
    celestial_dot = jnp.dot(r, z)

    native_perp = n - native_dot * e0
    celestial_perp = z - celestial_dot * r

    native_norm = jnp.linalg.norm(native_perp)
    celestial_norm = jnp.linalg.norm(celestial_perp)

    # Do not put a floor on these norms. For a map centered exactly on a celestial
    # pole both perpendicular vectors are cos(fl(pi/2)) ~ 6.1e-17 times the correct
    # unit direction (fl(pi/2) is not exactly pi/2, so the cosine is not exactly 0).
    # Dividing by the true norm recovers that direction, a floor (e.g. 1e-15) would
    # leave them far from unit length and give a wrong rotation.
    # Only an exactly zero norm is guarded, to avoid NaNs.
    u1 = native_perp / jnp.where(native_norm > 0, native_norm, 1.0)
    v1 = celestial_perp / jnp.where(celestial_norm > 0, celestial_norm, 1.0)

    u2 = jnp.cross(e0, u1)
    v2 = jnp.cross(r, v1)

    native_basis = jnp.stack((e0, u1, u2), axis=1)
    celestial_basis = jnp.stack((r, v1, v2), axis=1)

    return celestial_basis @ native_basis.T


def _rotate_native_to_celestial(phi, theta, matrix):
    r"""
    Transform native spherical coordinates to celestial coordinates.

    Parameters
    ----------
    phi : Float[Array, "..."]
        Native longitude in radians.
    theta : Float[Array, "..."]
        Native latitude in radians.
    matrix : Float[Array, "3 3"]
        Rotation matrix from native to celestial Cartesian coordinates.

    Returns
    -------
    lon : Float[Array, "..."]
        Celestial longitude in radians, wrapped into $[0, 2\pi)$.
    lat : Float[Array, "..."]
        Celestial latitude in radians.
    """
    native = _unit_vector(phi, theta)
    celestial = jnp.einsum("ij,j...->i...", matrix, native)

    x, y, z = celestial[0], celestial[1], celestial[2]
    lon = jnp.mod(jnp.arctan2(y, x), 2.0 * jnp.pi)
    lat = jnp.arctan2(z, jnp.sqrt(x * x + y * y))

    return lon, lat


def _rotate_celestial_to_native(lon, lat, matrix):
    """
    Transform celestial spherical coordinates to native coordinates.

    Parameters
    ----------
    lon : Float[Array, "..."]
        Celestial longitude in radians.
    lat : Float[Array, "..."]
        Celestial latitude in radians.
    matrix : Float[Array, "3 3"]
        Rotation matrix from native to celestial Cartesian coordinates.

    Returns
    -------
    phi : Float[Array, "..."]
        Native longitude in radians, wrapped into $[0, 2\\pi)$.
    theta : Float[Array, "..."]
        Native latitude in radians.
    """
    celestial = _unit_vector(lon, lat)
    native = jnp.einsum("ji,j...->i...", matrix, celestial)

    x, y, z = native[0], native[1], native[2]
    phi = jnp.mod(jnp.arctan2(y, x), 2.0 * jnp.pi)
    theta = jnp.arctan2(z, jnp.sqrt(x * x + y * y))

    return phi, theta


@register_pytree_node_class
@dataclass(frozen=True)
class JAXWCS:
    r"""
    Base class for JAX-compatible two-dimensional World Coordinate Systems (WCS).

    This class defines a common interface for mapping between pixel
    coordinates and world coordinates using JAX-compatible transformations.
    Projection-specific subclasses implement the coordinate transformations
    and may define additional projection parameters.

    The WCS parameters follow the FITS convention, with reference pixel
    coordinates specified in a one-based coordinate system. Angular
    parameters are stored in degrees unless otherwise specified by a
    subclass. Pixel coordinates returned by the transformation methods
    follow the conventions documented by each subclass.

    The linear transformation follows the FITS convention:

    $$
    \boldsymbol{q}
    =
    \mathrm{diag}(\mathrm{CDELT})\,\mathrm{PC}
    (\boldsymbol{p}-\mathrm{CRPIX}),
    $$

    where $\boldsymbol{p}$ denotes FITS one-based pixel coordinates and
    $\boldsymbol{q}$ denotes intermediate world coordinates.

    Celestial projections may additionally require a nonlinear projection
    and a spherical rotation between native and celestial coordinates.
    These operations are implemented by the corresponding subclasses.

    Attributes
    ----------
    crpix : Float[Array, "2"]
        Reference pixel coordinates in FITS one-based convention.
    crval : Float[Array, "2"]
        World coordinates at the reference pixel, in the units specified
        by the WCS axes. For celestial coordinates, these are typically
        right ascension and declination in degrees.
    cdelt : Float[Array, "2"]
        Coordinate increments along each intermediate world axis, typically
        in degrees per pixel for celestial WCS.
    pc : Float[Array, "2 2"]
        Linear transformation matrix relating pixel offsets to intermediate
        world-coordinate offsets.
    lonpole : Float[Array, ""]
        Native longitude of the celestial north pole (LONPOLE), in degrees,
        following the FITS WCS convention. Defaults to NaN, meaning the FITS
        default for the projection is used: 0 if the CRVAL latitude is
        at least `theta0`, otherwise 180 (Calabretta & Greisen 2002, sect. 2.4).
        See `_lonpole`.
    latpole : Float[Array, ""]
        Latitude of the pole (LATPOLE), in degrees, following the FITS WCS
        convention. After WCSLIB has resolved it this is also the native
        latitude of the celestial north pole. Only used by projections whose
        reference point is not the native pole (such as CAR), zenithal
        projections such as TAN take it from the CRVAL latitude instead.
    radesys : str
        Celestial reference frame, such as `ICRS` or `FK5`.
    equinox : float
        Equinox of the celestial reference frame, in years. Used for
        frames such as FK5 for which an equinox is applicable. If NaN,
        no equinox is explicitly specified.

    Class Attributes
    ----------------
    projection : str
        FITS projection code of the subclass, such as ``"CAR"`` or ``"TAN"``.
    theta0 : float
        Native latitude of the reference point in degrees, set by each
        subclass: 0 for CAR and 90 for zenithal projections such as TAN.
        Used for the default LONPOLE.
    _registry : dict[str, type[JAXWCS]]
        Registry mapping FITS projection codes to their corresponding
        JAX-compatible WCS implementations.
    """

    crpix: Float[Array, "2"]
    crval: Float[Array, "2"]
    cdelt: Float[Array, "2"]
    pc: Float[Array, "2 2"]
    lonpole: float = jnp.nan  # NaN = use the FITS default for this projection
    theta0: ClassVar[float]  # native latitude of the reference point (deg)
    latpole: float = 90.0
    radesys: str = "ICRS"
    equinox: float = jnp.nan

    _registry: ClassVar[dict[str, type["JAXWCS"]]] = {}

    def _lonpole(self):
        """
        LONPOLE with the FITS default applied if it was not specified (NaN).

        Returns
        -------
        Float[Array, ""]
            LONPOLE in degrees: `lonpole` if it is set, otherwise 0 if the
            CRVAL latitude is at least `theta0` and 180 if it is not
            (Calabretta & Greisen 2002, sect. 2.4).
        """
        default = jnp.where(self.crval[1] >= self.theta0, 0.0, 180.0)
        return jnp.where(jnp.isnan(self.lonpole), default, self.lonpole)

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
        proj : Callable
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
        jaxwcs : JAXWCS
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

        codes = tuple(axis[-3:] for axis in ctype)

        if codes[0] != codes[1]:
            raise NotImplementedError(f"Mixed projections are unsupported: {ctype}.")

        code = codes[0]
        subclass = cls._registry.get(code)

        if subclass is None:
            supported = ", ".join(sorted(cls._registry)) or "(none)"
            raise NotImplementedError(
                f"Unsupported projection {code!r}. Registered projections: {supported}."
            )

        return subclass._from_astropy(wcs)

    @classmethod
    def _from_astropy(cls, wcs: WCS) -> "JAXWCS":
        """
        Construct an instance from an Astropy WCS.
        Note that this is a placeholder for subclasses to implement.

        Parameters
        ----------
        wcs : WCS
            The Astropy WCS object to convert.

        Returns
        -------
        jaxwcs : JAXWCS
            A projector selected from the registered projection classes.
        """
        return cls(**cls._from_astropy_parameters(wcs))

    @classmethod
    def _from_astropy_parameters(cls, wcs: WCS) -> dict:
        """
        Extract common linear and spherical WCS parameters.

        Parameters
        ----------
        wcs : WCS
            The input Astropy WCS.

        Returns
        -------
        wcspars : dict
            Constructor parameters shared by projection subclasses.
            Angular parameters are stored in degrees.
        """
        return {
            "crpix": jnp.asarray(wcs.wcs.crpix[:2]),
            "crval": jnp.asarray(wcs.wcs.crval[:2]),
            "cdelt": jnp.asarray(wcs.wcs.cdelt[:2]),
            "pc": jnp.asarray(wcs.wcs.get_pc()[:2, :2]),
            "lonpole": jnp.asarray(wcs.wcs.lonpole),
            "latpole": jnp.asarray(wcs.wcs.latpole),
            "radesys": wcs.wcs.radesys,
            "equinox": float(wcs.wcs.equinox),
        }

    def to_header(self):
        """
        Convert the WCS to an Astropy FITS header.

        Returns
        -------
        astropy.io.fits.Header
            FITS header containing the WCS metadata.
        """
        projection = getattr(type(self), "projection", None)

        if projection is None:
            raise NotImplementedError("The WCS subclass must define a projection code.")
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = [
            f"RA---{projection}",
            f"DEC--{projection}",
        ]
        wcs.wcs.cunit = ["deg", "deg"]
        wcs.wcs.crpix = np.asarray(self.crpix)
        wcs.wcs.crval = np.asarray(self.crval)

        cdelt = np.asarray(self.cdelt)
        pc = np.asarray(self.pc)
        wcs.wcs.cd = np.diag(cdelt) @ pc

        wcs.wcs.lonpole = float(self._lonpole())
        wcs.wcs.latpole = float(self.latpole)
        wcs.wcs.radesys = self.radesys

        if not np.isnan(self.equinox):
            wcs.wcs.equinox = float(self.equinox)

        return wcs.to_header(relax=True)

    def tree_flatten(self):
        children = (
            self.crpix,
            self.crval,
            self.cdelt,
            self.pc,
            self.lonpole,
            self.latpole,
        )
        aux_data = (self.radesys, self.equinox)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        radesys, equinox = aux_data
        return cls(*children, radesys=radesys, equinox=equinox)

    @jax.jit
    def world_to_pixel(
        self, ra_rad: Float[Array, "..."], dec_rad: Float[Array, "..."]
    ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        """
        Convert celestial coordinates to pixel coordinates.
        Note that this is a placeholder for subclasses to implement.

        Parameters
        ----------
        ra_rad : Float[Array, "..."]
            Right ascension in radians.
        dec_rad : Float[Array, "..."]
            Declination in radians. Must be broadcast-compatible with
            `ra_rad`.

        Returns
        -------
        xpix : Float[Array, "..."]
            Zero-based pixel coordinates along the first map axis.
        ypix : Float[Array, "..."]
            Zero-based pixel coordinates along the second map axis.
        """
        _ = ra_rad, dec_rad
        raise NotImplementedError

    @jax.jit
    def pixel_to_world(
        self, xpix: Float[Array, "..."], ypix: Float[Array, "..."]
    ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        r"""
        Convert pixel coordinates to celestial coordinates.
        Note that this is a placeholder for subclasses to implement.

        Parameters
        ----------
        xpix : Float[Array, "..."]
            Zero-based pixel coordinates along the first map axis.
        ypix : Float[Array, "..."]
            Zero-based pixel coordinates along the second map axis.

        Returns
        -------
        ra_rad : Float[Array, "..."]
            Right ascension in radians, wrapped into $[0, 2\pi)$.
        dec_rad : Float[Array, "..."]
            Declination in radians.
        """
        _ = xpix, ypix
        raise NotImplementedError


@register_pytree_node_class
@JAXWCS.register_projection("CAR")
@dataclass(frozen=True)
class JAXCAR(JAXWCS):
    r"""
    JAX implementation of the CAR celestial map projection.

    This implementation transforms between celestial and native spherical
    coordinates using a three-dimensional rotation matrix. The native CAR
    projection is linear in longitude and latitude, after which the PC and
    CDELT matrices map intermediate coordinates to pixels.

    Pixel coordinates returned by this class are zero-based, consistent
    with NumPy indexing. CRPIX follows the FITS one-based convention.

    This implementation does not support SIP distortions or distortion
    lookup tables.

    The reference point (CRVAL) is native $(\phi_0, \theta_0) = (0, 0)$
    and the celestial north pole sits at native (LONPOLE, LATPOLE).

    See `JAXWCS` for details on the attributes of this class.
    """

    projection: ClassVar[str] = "CAR"
    theta0: ClassVar[float] = 0.0  # native latitude of the reference point (deg)

    @classmethod
    def _from_astropy(cls, wcs: WCS) -> "JAXCAR":
        """
        Construct a JAXCAR instance from an Astropy WCS.

        Parameters
        ----------
        wcs : WCS
            The input Astropy CAR WCS.

        Returns
        -------
        JAXCAR
            The corresponding JAX CAR projector.

        Raises
        ------
        NotImplementedError
            If the axis ordering is unsupported or distortion terms
            are present.
        ValueError
            If the pole parameters are geometrically inconsistent with
            the reference coordinate.
        """
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

        result = cls(**cls._from_astropy_parameters(wcs))

        crval = np.deg2rad(np.asarray(result.crval))
        lonpole = np.deg2rad(float(result.lonpole))
        latpole = np.deg2rad(float(result.latpole))

        native_dot = np.cos(latpole) * np.cos(lonpole)
        celestial_dot = np.sin(crval[1])

        if not np.isclose(native_dot, celestial_dot, atol=1e-8, rtol=1e-8):
            raise ValueError(
                "Inconsistent CRVAL/LONPOLE/LATPOLE geometry for this "
                "CAR reference point. Check the pole solution supplied "
                "by Astropy/WCSLIB."
            )

        if abs(np.sin(crval[1])) > 1.0 - 1e-12 and abs(np.cos(latpole)) < 1e-12:
            raise ValueError(
                "Degenerate pole configuration: the reference point lies "
                "at a celestial pole. A separate limiting rotation is needed."
            )

        return result

    def _rotation_matrix(self):
        """
        Construct the rotation matrix from native to celestial coordinates.

        Returns
        -------
        Float[Array, "3 3"]
            Rotation matrix constructed from CRVAL, LONPOLE, and LATPOLE.
        """
        return _native_to_celestial_matrix(
            jnp.deg2rad(self.crval),
            jnp.deg2rad(self._lonpole()),
            jnp.deg2rad(self.latpole),
            projection=self.projection,
        )

    @jax.jit
    def world_to_pixel(
        self, ra_rad: Float[Array, "..."], dec_rad: Float[Array, "..."]
    ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        r"""
        Convert celestial coordinates to CAR pixel coordinates.
        Celestial coordinates are first rotated into the native CAR
        coordinate system. Native longitude and latitude are then
        converted to intermediate coordinates in degrees.

        The linear transformation follows the FITS convention:

        $$
        \mathrm{intermediate}
        =
        \mathrm{diag}(\mathrm{CDELT})\,\mathrm{PC}
        (\mathrm{pixel}-\mathrm{CRPIX}).
        $$

        The inverse linear transformation gives the pixel coordinates.
        FITS one-based reference pixels are converted to zero-based
        coordinates by subtracting one.

        Parameters
        ----------
        ra_rad : Float[Array, "..."]
            Right ascension in radians.
        dec_rad : Float[Array, "..."]
            Declination in radians. Must be broadcast-compatible with
            `ra_rad`.

        Returns
        -------
        xpix : Float[Array, "..."]
            Zero-based pixel coordinates along the first map axis.
        ypix : Float[Array, "..."]
            Zero-based pixel coordinates along the second map axis.
        """
        matrix = self._rotation_matrix()
        phi, theta = _rotate_celestial_to_native(ra_rad, dec_rad, matrix)
        phi = (phi + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        intermediate = jnp.stack((jnp.rad2deg(phi), jnp.rad2deg(theta)), axis=0)

        scaled = intermediate / self.cdelt.reshape((2,) + (1,) * ra_rad.ndim)
        offset = jnp.einsum("ij,j...->i...", jnp.linalg.inv(self.pc), scaled)

        pix = offset + self.crpix.reshape((2,) + (1,) * ra_rad.ndim) - 1.0

        return pix[0], pix[1]

    @jax.jit
    def pixel_to_world(
        self, xpix: Float[Array, "..."], ypix: Float[Array, "..."]
    ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        r"""
        Convert CAR pixel coordinates to celestial coordinates.

        Pixel coordinates are converted to FITS one-based offsets and
        transformed to intermediate CAR coordinates:

        $$
        \mathrm{intermediate}
        =
        \mathrm{diag}(\mathrm{CDELT})\,\mathrm{PC}
        (\mathrm{pixel}-\mathrm{CRPIX}).
        $$

        The intermediate coordinates are interpreted as native CAR
        longitude and latitude. The spherical rotation defined by
        CRVAL, LONPOLE, and LATPOLE then converts them to celestial
        coordinates.

        Parameters
        ----------
        xpix : Float[Array, "..."]
            Zero-based pixel coordinates along the first map axis.
        ypix : Float[Array, "..."]
            Zero-based pixel coordinates along the second map axis.
            Must be broadcast-compatible with ``xpix``.

        Returns
        -------
        ra_rad : Float[Array, "..."]
            Right ascension in radians, wrapped into $[0, 2\pi)$.
        dec_rad : Float[Array, "..."]
            Declination in radians.
        """
        pix = jnp.stack((xpix, ypix), axis=0)

        offset = pix + 1.0 - self.crpix.reshape((2,) + (1,) * xpix.ndim)
        intermediate = self.cdelt.reshape((2,) + (1,) * xpix.ndim) * jnp.einsum(
            "ij,j...->i...", self.pc, offset
        )

        phi = jnp.deg2rad(intermediate[0])
        theta = jnp.deg2rad(intermediate[1])

        matrix = self._rotation_matrix()
        ra, dec = _rotate_native_to_celestial(phi, theta, matrix)

        return ra, dec


@register_pytree_node_class
@JAXWCS.register_projection("TAN")
@dataclass(frozen=True)
class JAXTAN(JAXWCS):
    r"""
    JAX implementation of the TAN celestial map projection.

    TAN is a zenithal projection: the reference point (CRVAL) is the native
    pole, $(\phi_0, \theta_0) = (0, 90^\circ)$, and the celestial north pole
    sits at native $(\mathrm{LONPOLE}, \delta_0)$ where $\delta_0$ is the CRVAL
    latitude. LATPOLE is therefore not used.

    Celestial coordinates are transformed into native spherical coordinates
    using the rotation matrix defined by CRVAL and LONPOLE.
    The native spherical coordinates are then projected onto the tangent
    plane using the gnomonic projection.

    Pixel coordinates returned by this class are zero-based, consistent
    with NumPy indexing. CRPIX follows the FITS one-based convention.

    This implementation does not support SIP distortions or distortion
    lookup tables.

    See `JAXWCS` for details on the attributes of this class.
    """

    projection: ClassVar[str] = "TAN"
    theta0: ClassVar[float] = 90.0  # native latitude of the reference point (deg)

    @classmethod
    def _from_astropy(cls, wcs: WCS) -> "JAXTAN":
        """
        Construct a JAXTAN instance from an Astropy WCS.

        Parameters
        ----------
        wcs : WCS
            The input Astropy TAN WCS.

        Returns
        -------
        JAXTAN
            The corresponding JAX TAN projector.

        Raises
        ------
        NotImplementedError
            If the axis ordering is unsupported or distortion terms
            are present.
        """
        ctype = tuple(str(c).upper() for c in wcs.wcs.ctype)

        if not (ctype[0].startswith("RA---TAN") and ctype[1].startswith("DEC--TAN")):
            raise NotImplementedError(f"Unsupported TAN axis ordering: {ctype}.")

        if (
            getattr(wcs, "sip", None) is not None
            or getattr(wcs, "cpdis1", None) is not None
            or getattr(wcs, "cpdis2", None) is not None
            or getattr(wcs, "det2im1", None) is not None
            or getattr(wcs, "det2im2", None) is not None
        ):
            raise NotImplementedError("TAN distortion terms are not supported.")

        params = cls._from_astropy_parameters(wcs)
        # Zenithal: the celestial pole is at native latitude delta0, store that
        # so that to_header writes the LATPOLE that is actually used
        params["latpole"] = params["crval"][1]
        return cls(**params)

    def _rotation_matrix(self):
        """
        Construct the rotation matrix from native to celestial coordinates.

        Returns
        -------
        Float[Array, "3 3"]
            Rotation matrix constructed from CRVAL and LONPOLE.
        """
        _, delta0 = jnp.deg2rad(self.crval)
        return _native_to_celestial_matrix(
            jnp.deg2rad(self.crval),
            jnp.deg2rad(self._lonpole()),
            delta0,  # Celestial pole sits at native latitude delta0 regardless of LATPOLE
            projection=self.projection,
        )

    @jax.jit
    def world_to_pixel(
        self, ra_rad: Float[Array, "..."], dec_rad: Float[Array, "..."]
    ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        r"""
        Convert celestial coordinates to TAN pixel coordinates.

        Celestial coordinates are first rotated into the native coordinate
        system. The native spherical coordinates are then projected onto
        the tangent plane using the gnomonic projection.

        The linear transformation follows the FITS convention:

        $$
        \mathrm{intermediate}
        =
        \mathrm{diag}(\mathrm{CDELT})\,\mathrm{PC}
        (\mathrm{pixel}-\mathrm{CRPIX}),
        $$

        which is inverted to get the pixel coordinates.
        Pixel coordinates are returned in zero-based convention.

        Points more than 90 degrees from the reference point (behind the
        tangent plane) have no TAN projection and are not flagged, they
        return finite but meaningless pixel coordinates.

        Parameters
        ----------
        ra_rad : Float[Array, "..."]
            Right ascension in radians.
        dec_rad : Float[Array, "..."]
            Declination in radians.

        Returns
        -------
        xpix, ypix : Float[Array, "..."]
            Zero-based pixel coordinates.
        """
        matrix = self._rotation_matrix()

        phi, theta = _rotate_celestial_to_native(ra_rad, dec_rad, matrix)
        sin_theta = jnp.sin(theta)

        # Gnomonic projection
        x = jnp.cos(theta) * jnp.sin(phi) / sin_theta
        y = -jnp.cos(theta) * jnp.cos(phi) / sin_theta
        intermediate = jnp.stack((jnp.rad2deg(x), jnp.rad2deg(y)), axis=0)
        scaled = intermediate / self.cdelt.reshape((2,) + (1,) * ra_rad.ndim)
        offset = jnp.einsum("ij,j...->i...", jnp.linalg.inv(self.pc), scaled)
        pix = offset + self.crpix.reshape((2,) + (1,) * ra_rad.ndim) - 1.0

        return pix[0], pix[1]

    @jax.jit
    def pixel_to_world(
        self, xpix: Float[Array, "..."], ypix: Float[Array, "..."]
    ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        r"""
        Convert TAN pixel coordinates to celestial coordinates.

        Pixel coordinates are first transformed into intermediate
        tangent-plane coordinates:

        $$
        \mathrm{intermediate}
        =
        \mathrm{diag}(\mathrm{CDELT})\,\mathrm{PC}
        (\mathrm{pixel}-\mathrm{CRPIX}).
        $$

        The inverse gnomonic projection then produces native spherical
        coordinates, which are rotated into celestial coordinates.

        Parameters
        ----------
        xpix : Float[Array, "..."]
            Zero-based pixel coordinates along the first map axis.
        ypix : Float[Array, "..."]
            Zero-based pixel coordinates along the second map axis.

        Returns
        -------
        ra_rad : Float[Array, "..."]
            Right ascension in radians, wrapped into $[0, 2\pi)$.
        dec_rad : Float[Array, "..."]
            Declination in radians.
        """
        pix = jnp.stack((xpix, ypix), axis=0)
        offset = pix + 1.0 - self.crpix.reshape((2,) + (1,) * xpix.ndim)
        intermediate = self.cdelt.reshape((2,) + (1,) * xpix.ndim) * jnp.einsum(
            "ij,j...->i...", self.pc, offset
        )
        x = jnp.deg2rad(intermediate[0])
        y = jnp.deg2rad(intermediate[1])

        # Inverse gnomonic projection
        phi = jnp.arctan2(x, -y)
        theta = jnp.arctan2(1.0, jnp.sqrt(x * x + y * y))
        matrix = self._rotation_matrix()
        ra, dec = _rotate_native_to_celestial(phi, theta, matrix)

        return ra, dec
