"""
Compare the JAX WCS projections against astropy (WCSLIB).

Note that round trips (pixel -> world -> pixel) are not enough on their own,
since a wrong spherical rotation used consistently in both directions still
round trips. Every case is therefore checked against astropy in both directions.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from astropy.wcs import WCS

from jitkasi.wcs import JAXCAR, JAXTAN, JAXWCS

ARCSEC = 3600.0

# Tolerances (WCSLIB agreement in float64 is ~1e-10" and ~1e-6 px)
SKY_TOL = 1e-4  # arcsec
PIX_TOL = 1e-4  # pixels


def make_wcs(proj, crval, cdelt, crpix, shape, rot=0.0):
    """
    Build a 2D celestial WCS with square pixels, RA increasing to the left
    and an optional rotation (in degrees) through the PC matrix.
    """
    w = WCS(naxis=2)
    w.wcs.ctype = [f"RA---{proj}", f"DEC--{proj}"]
    w.wcs.crval = crval
    w.wcs.crpix = crpix
    w.wcs.cdelt = [-cdelt, cdelt]
    c, s = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))
    w.wcs.pc = [[c, -s], [s, c]]
    w.wcs.set()
    return w, shape


# (name, kwargs for make_wcs)
TAN_CASES = [
    # Same WCS as the MOO1142 XMM maps (data/exp/back.fits): 113x85, 2.5" pixels
    (
        "xmm_moo1142",
        dict(
            crval=[175.685833333333, 15.4793888888889],
            cdelt=0.000694444444444445,
            crpix=[70.91000000000003, 78.91000000000003],
            shape=(113, 85),
        ),
    ),
    # Same WCS as the MOO1142 XMM PSF files: 101x101, 1" pixels
    (
        "xmm_psf",
        dict(
            crval=[3.21475641510593, -8.94838149967263],
            cdelt=0.000277777777777778,
            crpix=[50.5, 50.5],
            shape=(101, 101),
        ),
    ),
    (
        "rotated_pc",
        dict(crval=[175.7, 15.5], cdelt=2.5 / ARCSEC, crpix=[50.5, 40.5], shape=(100, 80), rot=30.0),
    ),
    ("high_dec", dict(crval=[40.0, 75.0], cdelt=2.5 / ARCSEC, crpix=[50.5, 40.5], shape=(100, 80))),
    ("south", dict(crval=[300.0, -60.0], cdelt=2.5 / ARCSEC, crpix=[50.5, 40.5], shape=(100, 80))),
    ("ra_wrap", dict(crval=[0.01, 5.0], cdelt=2.5 / ARCSEC, crpix=[50.5, 40.5], shape=(100, 80))),
    ("wide_field", dict(crval=[175.7, 15.5], cdelt=30 / ARCSEC, crpix=[60.5, 60.5], shape=(120, 120))),
    (
        "crpix_off_map",
        dict(crval=[175.7, 15.5], cdelt=2.5 / ARCSEC, crpix=[-200.0, 300.0], shape=(100, 80)),
    ),
]

CAR_CASES = [
    # ACT-like: reference on the equator, far from the map
    (
        "act_like",
        dict(crval=[175.7, 0.0], cdelt=30 / ARCSEC, crpix=[-5000.5, 400.5], shape=(100, 80)),
    ),
    (
        "rotated_pc",
        dict(crval=[175.7, 0.0], cdelt=30 / ARCSEC, crpix=[50.5, 40.5], shape=(100, 80), rot=30.0),
    ),
]

ALL_CASES = [("TAN", n, kw) for n, kw in TAN_CASES] + [("CAR", n, kw) for n, kw in CAR_CASES]
IDS = [f"{proj}-{name}" for proj, name, _ in ALL_CASES]


def angular_sep_arcsec(ra1, dec1, ra2, dec2):
    """
    Great circle distance (Vincenty formula) in arcseconds, inputs in radians.
    """
    dra = ra2 - ra1
    num = np.hypot(
        np.cos(dec2) * np.sin(dra),
        np.cos(dec1) * np.sin(dec2) - np.sin(dec1) * np.cos(dec2) * np.cos(dra),
    )
    den = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(dra)
    return np.rad2deg(np.arctan2(num, den)) * ARCSEC


def pixel_grid(shape):
    return np.meshgrid(
        np.arange(shape[0], dtype=float), np.arange(shape[1], dtype=float), indexing="ij"
    )


def astropy_world(w, xx, yy):
    ra, dec = w.wcs_pix2world(xx, yy, 0)
    return np.deg2rad(ra % 360.0), np.deg2rad(dec)


@pytest.fixture(params=ALL_CASES, ids=IDS)
def case(request):
    proj, _, kwargs = request.param
    w, shape = make_wcs(proj, **kwargs)
    return proj, w, JAXWCS.from_astropy(w), shape


def test_from_astropy_picks_projection(case):
    proj, _, jw, _ = case
    assert isinstance(jw, {"TAN": JAXTAN, "CAR": JAXCAR}[proj])


def test_pixel_to_world_matches_astropy(case):
    _, w, jw, shape = case
    xx, yy = pixel_grid(shape)
    ra_ref, dec_ref = astropy_world(w, xx, yy)

    ra, dec = jw.pixel_to_world(jnp.asarray(xx), jnp.asarray(yy))
    ra, dec = np.asarray(ra), np.asarray(dec)

    assert np.all(np.isfinite(ra)) and np.all(np.isfinite(dec))
    sep = angular_sep_arcsec(ra, dec, ra_ref, dec_ref)
    assert np.max(sep) < SKY_TOL, f"max offset from astropy {np.max(sep):.3g} arcsec"


def test_world_to_pixel_matches_astropy(case):
    _, w, jw, shape = case
    xx, yy = pixel_grid(shape)
    ra_ref, dec_ref = astropy_world(w, xx, yy)

    xpix, ypix = jw.world_to_pixel(jnp.asarray(ra_ref), jnp.asarray(dec_ref))
    xpix, ypix = np.asarray(xpix), np.asarray(ypix)

    assert np.all(np.isfinite(xpix)) and np.all(np.isfinite(ypix))
    dpix = np.hypot(xpix - xx, ypix - yy)
    assert np.max(dpix) < PIX_TOL, f"max offset from astropy {np.max(dpix):.3g} px"


def test_reference_pixel_is_crval(case):
    # The (zero-based) reference pixel has to map to CRVAL
    _, w, jw, _ = case
    x0, y0 = (float(c) - 1.0 for c in w.wcs.crpix)
    ra, dec = jw.pixel_to_world(jnp.asarray(x0), jnp.asarray(y0))
    crval = np.deg2rad(w.wcs.crval)
    sep = angular_sep_arcsec(float(ra), float(dec), crval[0], crval[1])
    assert sep < SKY_TOL, f"reference pixel is {sep:.3g} arcsec from CRVAL"


def test_round_trip(case):
    _, _, jw, shape = case
    xx, yy = pixel_grid(shape)
    ra, dec = jw.pixel_to_world(jnp.asarray(xx), jnp.asarray(yy))
    xpix, ypix = jw.world_to_pixel(ra, dec)
    dpix = np.hypot(np.asarray(xpix) - xx, np.asarray(ypix) - yy)
    assert np.max(dpix) < PIX_TOL


def test_to_header_matches_original(case):
    # A WCS rebuilt from to_header() has to put every pixel where the original does
    _, w, jw, shape = case
    xx, yy = pixel_grid(shape)
    ra_ref, dec_ref = astropy_world(w, xx, yy)
    ra, dec = astropy_world(WCS(jw.to_header()), xx, yy)
    sep = angular_sep_arcsec(ra, dec, ra_ref, dec_ref)
    assert np.max(sep) < SKY_TOL
