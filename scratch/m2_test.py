import glob
import os
import time
from copy import deepcopy

import jax
import jax.numpy as jnp
import minkasi
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from minkasi.tools import presets_by_source as pbs
from mpi4py import MPI
from witch.external.minkasi.utils import from_minkasi_tod

import jitkasi.mapmaking as jm
from jitkasi import noise as jn
from jitkasi.solutions import SolutionSet, maps
from jitkasi.tod import TODVec
from jitkasi.wcs import JAXCAR

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

TOD_DIR = "/mnt/welch/MUSTANG/M2-TODs/RXJ1347/mustang2/"
SOURCE = "RXJ1347"
PIX_SIZE = 2.0 / 3600 * np.pi / 180
PCG_ITERATIONS = 200


def log(message):
    if rank == 0:
        print(message, flush=True)


def array_stats(name, value):
    a = np.asarray(jax.device_get(value))
    finite = np.isfinite(a)

    if not finite.any():
        log(f"{name}: shape={a.shape}, NO FINITE VALUES")
        return

    good = a[finite]
    log(
        f"{name}: shape={a.shape}, dtype={a.dtype}, "
        f"finite={finite.sum()}/{a.size}, "
        f"min={good.min():.6e}, max={good.max():.6e}"
    )


def require_finite(name, value):
    a = np.asarray(jax.device_get(value))
    if not np.isfinite(a).all():
        raise ValueError(
            f"{name} contains NaN/Inf: "
            f"{np.count_nonzero(~np.isfinite(a))} invalid values "
            f"out of {a.size}"
        )


def compare_with_astropy(wcs, x, y):
    projector = JAXCAR.from_astropy(wcs)
    ra_jax, dec_jax = projector.pixel_to_world(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
    )
    ra_jax = np.rad2deg(np.asarray(jax.device_get(ra_jax)))
    dec_jax = np.rad2deg(np.asarray(jax.device_get(dec_jax)))
    ra_ref, dec_ref = wcs.all_pix2world(x, y, 0)
    dra = (ra_jax - ra_ref + 180.0) % 360.0 - 180.0
    ddec = dec_jax - dec_ref
    np.testing.assert_allclose(dra, 0.0, atol=1e-7)
    np.testing.assert_allclose(ddec, 0.0, atol=1e-7)


def save_fits(filename, data, wcs):
    if rank != 0:
        return
    hdu = fits.PrimaryHDU(data=np.asarray(jax.device_get(data)), header=wcs.to_header())
    hdu.writeto(filename, overwrite=True)


log("Loading TODs")
tod_fnames = sorted(glob.glob(os.path.join(TOD_DIR, "*.fits")))
if not tod_fnames:
    raise FileNotFoundError(f"No TOD FITS files found in {TOD_DIR}")
bad_tod, _ = pbs.get_bad_tods(SOURCE)
tod_fnames = sorted(minkasi.tods.io.cut_blacklist(tod_fnames, bad_tod))
if not tod_fnames:
    raise RuntimeError("No TODs remain after applying the blacklist")

tods = []
mtodvec = minkasi.tods.TodVec()
for i, fname in enumerate(tod_fnames):
    dat = minkasi.tods.io.read_tod_from_fits(fname)
    minkasi.tods.processing.truncate_tod(dat)
    minkasi.tods.processing.downsample_tod(dat)
    mtod = minkasi.tods.Tod(dat)
    mtodvec.add_tod(mtod)
    tods.append(from_minkasi_tod(deepcopy(mtod)))
todvec = TODVec(tods, comm)
log(f"Loaded {len(tods)} TODs")

log("Initializing map")
lims: tuple[float, float, float, float] = tuple(float(v) for v in todvec.lims)  # type: ignore
mmap = minkasi.maps.SkyMap(lims, PIX_SIZE)
jmap = maps.WCSMap.empty(name="rxj1347_map", comm=comm, wcs=mmap.wcs, lims=lims, pad=1)
log(f"JAX map shape: {jmap.data.shape}")
log(f"Minkasi map shape: {mmap.map.shape}")

log("Testing WCS")
ny, nx = jmap.data.shape[-2:]
test_x = np.array([0.0, nx / 2, nx - 1.0])
test_y = np.array([0.0, ny / 2, ny - 1.0])
compare_with_astropy(mmap.wcs, test_x, test_y)
log("WCS test passed")

for i, tod in enumerate(todvec):
    xpix, ypix = jmap.wcs.world_to_pixel(tod.x, tod.y)
    xpix_np = np.asarray(jax.device_get(xpix))
    ypix_np = np.asarray(jax.device_get(ypix))

    valid = (
        np.isfinite(xpix_np)
        & np.isfinite(ypix_np)
        & (xpix_np >= 0)
        & (xpix_np < nx)
        & (ypix_np >= 0)
        & (ypix_np < ny)
    )

    log(
        f"TOD {i}: in-bounds samples "
        f"{valid.sum()}/{valid.size}; "
        f"x range [{np.nanmin(xpix_np):.2f}, "
        f"{np.nanmax(xpix_np):.2f}], "
        f"y range [{np.nanmin(ypix_np):.2f}, "
        f"{np.nanmax(ypix_np):.2f}]"
    )

log("Computing noise")
t0 = time.time()
for tod in todvec:
    tod.compute_noise(jn.NoiseSmoothedSVD, None, fwhm=50)
todvec = jax.block_until_ready(todvec)
t1 = time.time()
log(f"jitkasi noise {t1 - t0}")
t0 = time.time()
for tod in mtodvec.tods:
    tod.info["ipix"] = mmap.get_pix(tod)
    tod.set_noise(minkasi.mapmaking.noise.NoiseSmoothedSVD)
t1 = time.time()
log(f"minkasi noise {t1 - t0}")


log("Computing hits and precon")
jmap = jmap.make_hits(todvec, False)
precon = jmap.copy(deep=True)
precon.data = jnp.where(
    jnp.isfinite(jmap.hits) & (jmap.hits > 1),
    1.0 / jnp.where(jmap.hits > 1, jmap.hits, 1.0),
    0.0,
)
array_stats("Preconditioner", precon.data)
require_finite("Preconditioner", precon.data)
save_fits("hits.fits", jmap.hits, jmap.wcs)

hits = minkasi.mapmaking.make_hits(mtodvec, mmap)
mapset = minkasi.maps.Mapset()
mapset.add_map(mmap)
mprecon = mapset.copy()
tmp = hits.map.copy()
ii = tmp > 0
tmp[ii] = 1.0 / tmp[ii]
mprecon.maps[0].map[:] = tmp[:]

log("Building naive map")

naive = jmap.from_tods(todvec, True)
binned = jnp.asarray(naive.data)

jmap = jmap.make_ivar(todvec)
ivar = jmap.ivar
valid_ivar = jnp.isfinite(ivar) & (ivar >= 1)
naive.data = jnp.where(
    valid_ivar,
    binned / jnp.where(valid_ivar, ivar, 1.0),
    0.0,
)
naive.data = jnp.nan_to_num(naive.data, nan=0.0, posinf=0.0, neginf=0.0)

array_stats("Naive map", naive.data)
require_finite("Naive map", naive.data)
save_fits("naive.fits", naive.data, naive.wcs)

log("Setting up SolutionSets")

sset = SolutionSet([jmap], comm)
pset = SolutionSet([precon], comm)
naive.data = naive.data.at[:].set(0.0)
x0 = SolutionSet([naive], comm)

mx0 = mapset.copy()
mx0.clear()

log("Computing RHS")

rhs = jm.make_rhs(sset, todvec)
mrhs = mapset.copy()
mtodvec.make_rhs(mrhs)

array_stats("JAX RHS map", rhs[0].data)
require_finite("JAX RHS map", rhs[0].data)

log("Running PCG")
array_stats("Initial guess", x0[0].data)
require_finite("Initial guess", x0[0].data)

t0 = time.time()
solved = jm.run_pcg(
    rhs,
    todvec,
    x0,
    pset,
    maxiter=PCG_ITERATIONS,
)
solved = jax.block_until_ready(solved)
t1 = time.time()
print(f"jitkasi: {t1 - t0}")

t0 = time.time()
mapset_out = minkasi.mapmaking.run_pcg(
    mrhs, mx0, mtodvec, mprecon, maxiter=PCG_ITERATIONS, save_iters=[], outroot="."
)
t1 = time.time()
print(f"minkasi: {t1 - t0}")

array_stats("Solved map", solved[0].data)
require_finite("Solved map", solved[0].data)

log("Saving results")
save_fits("solved.fits", solved[0].data, solved[0].wcs)

log("Done")
