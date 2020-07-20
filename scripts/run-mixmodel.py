# Standard library
import os
import sys
userpath = os.path.expanduser('~/')
os.environ["THEANO_FLAGS"] = f'base_compiledir={userpath}/.theano/{os.getpid()}'

# Third-party
import astropy.units as u
import numpy as np
from pyia import GaiaData
import pymc3 as pm
import exoplanet as xo

from model import ComovingHelper


def worker(task):
    model, y, M, Cinv, test_r, test_vxyz = task

    with model:
        pm.set_data({'y': y,
                     'Cinv': Cinv,
                     'M': M})

        test_pt = {'vxyz': test_vxyz,
                   'r': test_r,
                   'f': np.array([0.5, 0.5])}
        try:
            # print("starting optimize")
            res = xo.optimize(start=test_pt, progress_bar=False, verbose=False)

            # print("done optimize - starting sample")
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
            return np.nan

        # print("done sample - computing prob")
        ll_fg = trace.get_values(model.group_logp)
        ll_bg = trace.get_values(model.field_logp)
        post_prob = np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
        post_prob = post_prob.sum() / len(post_prob)

        return post_prob


def main(pool):
    g = GaiaData('../data/150pc_MG12-result.fits.gz')
    the_og = g[g.source_id == 1490845584382687232]

    # Only stars within 100 pc of the OG:
    c = g.get_skycoord()
    the_og_c = the_og.get_skycoord()[0]
    sep3d_mask = c.separation_3d(the_og_c) < 100*u.pc
    subg = g[sep3d_mask]

    # Results from Field-velocity-distribution.ipynb:
    vfield = np.array([[-1.49966296, 14.54365055, -9.39127686],
                       [-8.78150468, 22.08294278, -22.9100212],
                       [-112.0987016, 120.8536385, -179.84992332]])
    sigvfield = np.array([15.245, 37.146, 109.5])
    wfield = np.array([0.53161301, 0.46602227, 0.00236472])

    helper = ComovingHelper(subg)
    model = helper.get_model(v0=np.array([-6.932, 24.301, -9.509]),
                             sigma_v0=0.6,
                             vfield=vfield, sigma_vfield=sigvfield,
                             wfield=wfield)

    tasks = [(model, helper.ys[n], helper.Ms[n], helper.Cinvs[n],
              helper.test_r[n], helper.test_vxyz[n])
             for n in range(helper.N)]

    probs = []
    for prob in pool.map(worker, tasks):
        probs.append(prob)
    probs = np.array(probs)

    print(probs)


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
