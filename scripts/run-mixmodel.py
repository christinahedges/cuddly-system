# Standard library
import atexit
import os
import sys
import time
# userpath = os.path.expanduser('~/')
# os.environ["THEANO_FLAGS"] = f'base_compiledir={userpath}/.theano/{os.getpid()}'
# os.environ["THEANO_FLAGS"] = os.environ["THEANO_FLAGS"] + ',blas.ldflags="-L/cm/shared/sw/pkg/base/openblas/0.3.6-haswell/lib -lopenblas"'

compilepath = './theano-compile'
pid = os.environ.get('OMPI_COMM_WORLD_RANK', os.getpid())
os.environ["THEANO_FLAGS"] = f'base_compiledir={compilepath}/{pid}'

# https://github.com/numpy/numpy/issues/11734
# os.environ["KMP_INIT_AT_FORK"] = "False"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_THREADING_LAYER"] = "TBB"

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
    (i1, i2), data, model_kw = task

    g = GaiaData(data)

    cache_filename = os.path.abspath(f'../cache/tmp_{i1}-{i2}.fits')
    if os.path.exists(cache_filename):
        print(f"({pid}) cache filename exists for index range: "
              f"{cache_filename}")
        return cache_filename

    print(f"({pid}) setting up model")
    helper = ComovingHelper(g)

    niter = 0
    while niter < 10:
        try:
            model = helper.get_model(**model_kw)
            break
        except OSError:
            print(f"{pid} failed to compile - trying again in 2sec...")
            time.sleep(5)
            niter += 1
            continue
    else:
        print(f"{pid} never successfully compiled. aborting")
        import socket
        print(socket.gethostname(), socket.getfqdn(),
              os.path.exists("/cm/shared/sw/pkg/devel/gcc/7.4.0/bin/g++"))
        return ''

    print(f"({pid}) done init model - running {len(g)} stars")

    probs = np.full(helper.N, np.nan)
    for n in range(helper.N):
        with model:
            pm.set_data({'y': helper.ys[n],
                         'Cinv': helper.Cinvs[n],
                         'M': helper.Ms[n]})

            test_pt = {'vxyz': helper.test_vxyz[n],
                       'r': helper.test_r[n],
                       'w': np.array([0.5, 0.5])}
            try:
                print("starting optimize")
                res = xo.optimize(start=test_pt,
                                  progress_bar=False,
                                  verbose=False)

                print("done optimize - starting sample")
                trace = pm.sample(
                    start=res,
                    tune=2000,
                    draws=1000,
                    cores=1,
                    chains=1,
                    step=xo.get_dense_nuts_step(target_accept=0.95),
                    progressbar=False
                )
            except Exception as e:
                print(str(e))
                continue

            # print("done sample - computing prob")
            ll_fg = trace.get_values(model.group_logp)
            ll_bg = trace.get_values(model.field_logp)
            post_prob = np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
            probs[n] = post_prob.sum() / len(post_prob)

    # write probs to cache filename
    tbl = at.Table()
    tbl['source_id'] = g.source_id
    tbl['prob'] = probs
    tbl.write(cache_filename)

    return cache_filename


def combine_output():
    import glob

    all_filename = '../cache/probs.fits'
    if os.path.exists(all_filename):
        prev_table = at.Table.read(all_filename)
    else:
        prev_table = None

    # combine the individual worker cache files
    all_tables = []
    remove_filenames = []
    for filename in glob.glob('../cache/tmp*.fits'):
        all_tables.append(at.Table.read(filename))
        remove_filenames.append(filename)
    all_table = at.vstack(all_tables)

    if prev_table:
        all_table = at.vstack((prev_table, all_table))

    _, idx = np.unique(all_table['source_id'], return_index=True)
    all_table[idx].write(all_filename, overwrite=True)

    for filename in remove_filenames:
        os.unlink(filename)


def main(pool, data_file):
    # When this exits on the main process, combine any output files
    atexit.register(combine_output)

    basename = os.path.basename(data_file).split('.')[0]

    from schwimmbad.utils import batch_tasks

    filename = os.path.abspath(f'../cache/probs-{basename}.fits')
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

    g = GaiaData(data_file)
    # the_og = g[g.source_id == 1490845584382687232]

    print("data loaded")

    # Only stars within 100 pc of the OG:
    # EDIT: turning this off...
    # c = g.get_skycoord()
    # the_og_c = the_og.get_skycoord()[0]
    # sep3d_mask = c.separation_3d(the_og_c) < 100*u.pc
    # subg = g[sep3d_mask & ~np.isin(g.source_id, done['source_id'])]
    subg = g[~np.isin(g.source_id, done['source_id']) &
             ((g.parallax / g.parallax_error) > 5)]
    subc = subg.get_skycoord()

    # The OG!
    v0 = np.array([-6.932, 24.301, -9.509])  # km/s
    sigma_v0 = 0.6  # km/s

    # For stars with reported radial velocities, remove very different vxyz:
    vxyz = subc[np.isfinite(subg.radial_velocity)].velocity.d_xyz
    vxyz = vxyz.to_value(u.km/u.s).T
    dv_mask = np.linalg.norm(vxyz - v0, axis=1) > 25.
    dv_mask = ~np.isin(
        subg.source_id,
        subg.source_id[np.isfinite(subg.radial_velocity)][dv_mask])
    subg = subg[dv_mask]

    # Results from Field-velocity-distribution.ipynb:
    model_kw = dict()
    model_kw['v0'] = v0
    model_kw['sigma_v0'] = sigma_v0
    model_kw['vfield'] = np.array([[-1.49966296, 14.54365055, -9.39127686],
                                   [-8.78150468, 22.08294278, -22.9100212],
                                   [-112.0987016, 120.8536385, -179.84992332]])
    model_kw['sigma_vfield'] = np.array([15.245, 37.146, 109.5])
    model_kw['wfield'] = np.array([0.53161301, 0.46602227, 0.00236472])

    tasks = batch_tasks(n_batches=8 * pool.size,
                        arr=subg.data,
                        args=(model_kw, ))

    sub_filenames = []
    for sub_filename in pool.map(worker, tasks):
        sub_filenames.append(sub_filename)


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser()

    parser.add_argument("--data", dest="data_file", required=True,
                        type=str, help="the source data file")

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
            main(pool=pool, data_file=parsed.data_file)

    sys.exit(0)
