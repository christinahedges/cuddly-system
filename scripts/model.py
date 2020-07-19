# Third-party
import astropy.units as u
import numpy as np
import pymc3 as pm
import theano.tensor as tt

pc_mas_yr_per_km_s = (1 * u.km/u.s).to(u.pc*u.mas/u.yr,
                                       u.dimensionless_angles()).value
km_s_per_pc_mas_yr = 1 / pc_mas_yr_per_km_s

__all__ = ['UniformSpaceDensity']


class UniformSpaceDensity(pm.Continuous):

    def __init__(self, rlim, **kwargs):
        """A uniform space density prior over a distance, r, between (0, rlim)
        """

        self.rlim = float(rlim)
        assert (self.rlim > 0)
        self._fac = np.log(3.) - 3 * np.log(self.rlim)

        shape = kwargs.get("shape", None)
        if shape is None:
            testval = 0.5 * self.rlim
        else:
            testval = 0.5 * self.rlim + np.zeros(shape)
        kwargs["testval"] = kwargs.pop("testval", testval)
        super(UniformSpaceDensity, self).__init__(**kwargs)

    def _random(self, size=None):
        uu = np.random.uniform(size=size)
        return np.cbrt(uu) * self.rlim

    def random(self, point=None, size=None):
        return pm.distributions.generate_samples(
            self._random,
            dist_shape=self.shape,
            broadcast_shape=self.shape,
            size=size,
        )

    def logp(self, value):
        return 2 * tt.log(tt.as_tensor_variable(value)) + self._fac


def get_tangent_basis(ra, dec):
    """
    row vectors are the tangent-space basis at (alpha, delta, r)
    ra, dec in radians
    """
    M = np.array([
        [-np.sin(ra), np.cos(ra), 0.],
        [-np.sin(dec)*np.cos(ra), -np.sin(dec)*np.sin(ra), np.cos(dec)],
        [np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)]
    ])
    return M


class BaseHelper:

    def __init__(self, g):
        self.N = len(g)

        ra = g.ra.to_value(u.rad)
        dec = g.dec.to_value(u.rad)
        Ms = np.stack([get_tangent_basis(ra[i], dec[i])
                       for i in range(self.N)])
        ys = np.stack([g.parallax.value,
                       g.pmra.value,
                       g.pmdec.value,
                       g.radial_velocity.value],
                      axis=1)
        ys[np.isnan(ys[:, 3]), 3] = 0.

        Cs = g.get_cov(units=dict(parallax=u.mas,
                                  pmra=u.mas/u.yr,
                                  pmdec=u.mas/u.yr,
                                  radial_velocity=u.km/u.s))
        Cs = Cs[:, 2:, 2:]
        Cinvs = np.stack([np.linalg.inv(Cs[n])
                          for n in range(self.N)], axis=0)

        assert np.isfinite(ys).all()
        assert np.isfinite(Ms).all()
        assert np.isfinite(Cinvs).all()

        self.ys = ys
        self.Ms = Ms
        self.Cs = Cs
        self.Cinvs = Cinvs

        # Set up test values:
        self.test_r = g.distance.value.reshape(-1, 1)

        rv = g.radial_velocity.value
        rv[~np.isfinite(rv)] = 0.
        c = g.get_skycoord(radial_velocity=rv*u.km/u.s)
        self.test_vxyz = c.velocity.d_xyz.value.T


class ComovingHelper(BaseHelper):

    def get_model(self, v0, sigma_v0, vfield, sigma_vfield, wfield):
        # Number of prior mixture components:
        with pm.Model() as model:
            # Data per star:
            M = pm.Data('M', np.eye(3))
            Cinv = pm.Data('Cinv', np.eye(4))
            y = pm.Data('y', np.zeros(4))

            # True distance:
            r = UniformSpaceDensity("r", 250, shape=(1, ))

            # Group velocity distribution
            pvgroup = pm.Normal.dist(v0, sigma_v0, shape=3)

            # Milky Way velocity distribution
            K = vfield.shape[0]
            pvdists = []
            for k in range(K):
                pvtmp = pm.Normal.dist(vfield[k], sigma_vfield[k], shape=3)
                pvdists.append(pvtmp)
            pvfield = pm.Mixture.dist(w=np.array(wfield),
                                      comp_dists=pvdists,
                                      shape=3,
                                      testval=self.test_vxyz)

            # Mixture model for 3D velocity
            f = pm.Dirichlet('f', a=np.ones(2))

            def logp(v):
                a = pvgroup.logp(v).sum() + tt.log(f[0])
                b = pvfield.logp(v).sum() + tt.log(f[1])
                return pm.logsumexp([a, b])
            vxyz = pm.DensityDist('vxyz', logp, shape=3)

            # Store log probs for each mixture component:
            pm.Deterministic('group_logp',
                             pvgroup.logp(vxyz).sum() + tt.log(f[0]))
            pm.Deterministic('field_logp',
                             pvfield.logp(vxyz).sum() + tt.log(f[1]))

            # Velocity in tangent plane coordinates
            vtan = tt.dot(M, vxyz)

            model_pm = vtan[:2] / r * pc_mas_yr_per_km_s
            model_rv = vtan[2:3]
            model_y = tt.concatenate((1000 / r, model_pm, model_rv), axis=0)

            pm.Deterministic('model_y', model_y)
            # val = pm.MvNormal('like', mu=model_y, tau=Cinv, observed=y)
            dy = y - model_y
            pm.Potential('chisq', -0.5 * tt.dot(dy, tt.dot(Cinv, dy)))

        return model


class FieldHelper(BaseHelper):

    def get_model(self, vfield0, sigma_vfield0):
        # Number of prior mixture components:
        with pm.Model() as model:

            # True distance:
            rlim = 250
            BoundedR = pm.Bound(UniformSpaceDensity, lower=0, upper=rlim)
            r = BoundedR("r", rlim, shape=(self.N, 1),
                         testval=self.test_r)

            # Milky Way velocity distribution
            K = vfield0.shape[0]
            w = pm.Dirichlet('w', a=np.ones(K))

            # Set up means and variances:
            meanvs = []
            sigvs = []
            for k in range(K):
                vtmp = pm.Normal(f'vmean{k}', vfield0[k], 10., shape=3)  # HACK
                lnstmp = pm.Normal(f'lns{k}',
                                   np.log(sigma_vfield0[k]), 1.,  # HACK
                                   shape=3)
                stmp = pm.Deterministic(f'vsig{k}', tt.exp(lnstmp))

                meanvs.append(vtmp)
                sigvs.append(stmp)

            pvdists = []
            for k in range(K):
                pvtmp = pm.Normal.dist(meanvs[k], sigvs[k], shape=3)
                pvdists.append(pvtmp)
            vxyz = pm.Mixture('vxyz', w=w,
                              comp_dists=pvdists, shape=(self.N, 3))

            # Velocity in tangent plane coordinates
            vtan = tt.batched_dot(self.Ms, vxyz)

            model_pm = vtan[:, :2] / r * pc_mas_yr_per_km_s
            model_rv = vtan[:, 2:3]
            model_y = tt.concatenate((1000 / r, model_pm, model_rv), axis=1)

            pm.Deterministic('model_y', model_y)
            # val = pm.MvNormal('like', mu=model_y, tau=Cinv, observed=y)
            dy = self.ys - model_y
            pm.Potential('chisq',
                         -0.5 * tt.batched_dot(dy,
                                               tt.batched_dot(self.Cinvs, dy)))

        return model
