# Standard library
import os
import sys

# Third-party
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
        res = xo.optimize(start=test_pt, progressbar=False, verbose=False)

        trace = pm.sample(
            init=res,
            tune=1000,
            draws=1000,
            cores=1,
            chains=1,
            step=xo.get_dense_nuts_step(target_accept=0.95),
            progressbar=False
        )

        ll_fg = trace.get_values(model.group_logp)
        ll_bg = trace.get_values(model.field_logp)
        post_prob = np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
        post_prob = post_prob.sum() / len(post_prob)

        return post_prob


def main(pool):
    g = GaiaData('../data/150pc_MG12-result.fits.gz')

    # TODO: replace this with results from Field-velocity-distribution.ipynb
    vfield = np.array([[   1.72666906,   17.96839664,   -5.97107496],
                       [ -15.0730518 ,   33.09260716,  -31.36846921],
                       [-113.89493924,  122.05855141, -180.76490601]])
    sigvfield = np.array([[ 20.,  20,  20],
                          [ 50.,  50,  50],
                          [125., 125, 125]])
    wfield = np.array([0.65, 0.33, 0.02])

    helper = ComovingHelper(g)
    model = helper.get_model(v0=np.array([-6.932, 24.301, -9.509]),
                             sigma_v0=0.6,
                             vfield=vfield, sigma_vfield=sigvfield,
                             wfield=wfield)

    import pickle
    with open('test.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('test.pkl', 'rb') as f:
        model = pickle.load(f)

    return


    tasks = [(model, helper.ys[n], helper.Ms[n], helper.Cinvs[n],
              helper.test_r[n], helper.test_vxyz[n])
             for n in range(helper.N)]

    probs = []
    for prob in pool.map(worker, tasks[:10]):
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
