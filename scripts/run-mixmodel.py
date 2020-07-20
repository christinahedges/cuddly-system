# Standard library
import os
import sys
userpath = os.path.expanduser('~/')
os.environ["THEANO_FLAGS"] = f'base_compiledir={userpath}/.theano/{os.getpid()}'

# Third-party
from astropy.utils import iers
iers.conf.auto_download = False
import astropy.table as at
import astropy.units as u
import numpy as np
from pyia import GaiaData
import pymc3 as pm
import exoplanet as xo

from model import ComovingHelper


def worker(task):
    model, y, M, Cinv, test_r, test_vxyz, source_id, filename = task

    with model:
        pm.set_data({'y': y,
                     'Cinv': Cinv,
                     'M': M})

        test_pt = {'vxyz': test_vxyz,
                   'r': test_r,
                   'f': np.array([0.5, 0.5])}
        try:
            print("starting optimize")
            res = xo.optimize(start=test_pt, progress_bar=False, verbose=False)

            print("done optimize - starting sample")
            trace = pm.sample(
                init=res,
                tune=1000,
                draws=1000,
                cores=1,
                chains=1,
                step=xo.get_dense_nuts_step(target_accept=0.95),
                progressbar=False
            )
        except Exception as e:
            print(str(e))
            return source_id, np.nan, filename

        # print("done sample - computing prob")
        ll_fg = trace.get_values(model.group_logp)
        ll_bg = trace.get_values(model.field_logp)
        post_prob = np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
        post_prob = post_prob.sum() / len(post_prob)

        return source_id, post_prob, filename


def callback(result):
    source_id, prob, filename = result
    done = at.Table.read(filename)
    done.add_row({'source_id': source_id,
                  'prob': prob})
    done.write(filename, overwrite=True)


def main(pool):
    filename = os.path.abspath('../cache/probs.fits')
    _path, _ = os.path.split(filename)
    os.makedirs(_path, exist_ok=True)

    # Load already done stars:
    if os.path.exists(filename):
        done = at.Table.read(filename)
        print(f'{len(done)} already done')
    else:
        done = at.Table({'source_id': [], 'prob': []},
                        dtype=(np.int64, np.float64))
        done.write(filename)

    g = GaiaData('../data/150pc_MG12-result.fits.gz')
    the_og = g[g.source_id == 1490845584382687232]

    print("data loaded")

    # Only stars within 100 pc of the OG:
    c = g.get_skycoord()
    the_og_c = the_og.get_skycoord()[0]
    sep3d_mask = c.separation_3d(the_og_c) < 100*u.pc
    subg = g[sep3d_mask & ~np.isin(g.source_id, done['source_id'])]
    subc = subg.get_skycoord()

    # The OG!
    v0 = np.array([-6.932, 24.301, -9.509])  # km/s
    sigma_v0 = 0.6  # km/s

    # For stars with reported radial velocities, remove very different vxyz:
    vxyz = subc[np.isfinite(subg.radial_velocity)].velocity.d_xyz
    vxyz = vxyz.to_value(u.km/u.s).T
    dv_mask = np.linalg.norm(vxyz - v0, axis=1) > 15.
    dv_mask = ~np.isin(
        subg.source_id,
        subg.source_id[np.isfinite(subg.radial_velocity)][dv_mask])
    subg = subg[dv_mask]

    # Results from Field-velocity-distribution.ipynb:
    vfield = np.array([[-1.49966296, 14.54365055, -9.39127686],
                       [-8.78150468, 22.08294278, -22.9100212],
                       [-112.0987016, 120.8536385, -179.84992332]])
    sigvfield = np.array([15.245, 37.146, 109.5])
    wfield = np.array([0.53161301, 0.46602227, 0.00236472])

    print("setting up model")
    helper = ComovingHelper(subg)
    model = helper.get_model(v0=v0,
                             sigma_v0=sigma_v0,
                             vfield=vfield, sigma_vfield=sigvfield,
                             wfield=wfield)

    print(f"done init model - making {len(subg)} tasks")
    tasks = [(model, helper.ys[n], helper.Ms[n], helper.Cinvs[n],
              helper.test_r[n], helper.test_vxyz[n], subg.source_id[n],
              filename)
             for n in range(helper.N)]

    for _, prob, _ in pool.map(worker, tasks, callback=callback):
        pass


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser()

    # vq_group = parser.add_mutually_exclusive_group()
    # vq_group.add_argument('-v', '--verbose', action='count', default=0,
    #                       dest='verbosity')
    # vq_group.add_argument('-q', '--quiet', action='count', default=0,
    #                       dest='quietness')

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    group.add_argument("--mpiasync", dest="mpiasync", default=False,
                       action="store_true", help="Run with MPI async.")

    parsed = parser.parse_args()

    # deal with multiproc:
    if parsed.mpi:
        from schwimmbad.mpi import MPIPool
        Pool = MPIPool
        kw = dict()
    elif parsed.mpiasync:
        from schwimmbad.mpi import MPIAsyncPool
        Pool = MPIAsyncPool
        kw = dict()
    elif parsed.n_procs > 1:
        from schwimmbad import MultiPool
        Pool = MultiPool
        kw = dict(processes=parsed.n_procs)
    else:
        from schwimmbad import SerialPool
        Pool = SerialPool
        kw = dict()
    Pool = Pool
    Pool_kwargs = kw

    with threadpool_limits(limits=1, user_api='blas'):
        with Pool(**Pool_kwargs) as pool:
            main(pool=pool)

    sys.exit(0)
