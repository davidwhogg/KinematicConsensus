"""
This file is part of the Kinematic Consensus project.
Copyright 2013 David W. Hogg (NYU)

bugs / issues:
- Doesn't add observational noise to ObservedStar.
- Could sample velocity and position independently.
- Could sample velocity analytically!
"""

import numpy as np
import emcee
import triangle as tri

class SixPosition:

    def __init__(self, sixvector):
        """
        input:
        - `sixvector`: 6-element np array Galactocentric phase-space position (kpc, km s^{-1})

        output:
        - `SixPosition` object

        bugs:
        - `potential_amplitude` should not be part of this `class`.
        - `potential_amplitude` is a magic number.
        """
        assert len(sixvector) == 6
        self.pos = np.array(sixvector)
        self.xhat = np.array([1., 0., 0.])
        self.yhat = np.array([0., 1., 0.])
        self.zhat = np.array([0., 0., 1.])
        self.potential_amplitude = 173.**2 # km^2 s^{-2}
        self.potential_scale = 100. # kpc
        return None

    def get_sixpos(self):
        """
        output:
        - 6-element np array Galactocentric phase-space position (kpc, km s^{-1})
        """
        return self.pos

    def get_sixpos_names(self):
        return np.array([r"$x$", r"$y$", r"$z$", r"$v_x$", r"$v_y$", r"$v_z$"])

    def get_sixpos_extents(self):
        """
        bugs:
        - Way hard-coded.
        """
        return [(-120., 120.), (-120., 120.), (-120., 120.), (-500., 500.), (-500., 500.), (-500., 500.)]

    def get_sun_sixpos(self):
        """
        output:
        - the position of the Sun in 6-vector phase space (kpc, kpc, kpc, km s^{-1}, km s^{-1}, km s^{-1})
        """
        return np.array([-8., 0., 0., 0., 220., 0.])

    def get_helio_sixpos(self):
        """
        output:
        - heliocentric 6-vector phase-space position (kpc, km s^{-1})
        """
        return self.get_sixpos() - self.get_sun_sixpos()

    def get_helio_3pos(self):
        """
        output:
        - heliocentric 3-vector spatial position (kpc)
        """
        return self.get_helio_sixpos()[:3]

    def get_helio_3vel(self):
        """
        output:
        - heliocentric 3-vector velocity (km s^{-1})
        """
        return self.get_helio_sixpos()[3:]

    def get_lb(self):
        """
        output:
        - Galactic celestial coordinates (deg)
        """
        x = self.get_helio_3pos()
        l = np.rad2deg(np.arctan2(x[1], x[0]))
        while l < 0.:
            l += 360.
        b = np.rad2deg(np.arcsin(x[2] / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)))
        return np.array([l, b])

    def get_helio_distance(self):
        """
        output:
        - heliocentric distance (kpc)
        """
        return np.sqrt(np.sum(self.get_helio_3pos() ** 2))

    def get_helio_unit_vectors(self):
        """
        Compute and return unit vectors in spherical heliocentric coordinate system r, l, b.

        outputs:
        - `rhat`, `lhat`, `bhat`:

        bugs:
        - Handedness of the system has not been tested; probably wrong!
        - Handles pole issues stupidly.
        """
        rhat = self.get_helio_3pos()
        rhat /= np.sqrt(np.sum(rhat ** 2))
        lhat = np.cross(rhat, self.zhat)
        if np.dot(lhat, lhat) < 1e-15:
            return rhat, self.yhat, self.xhat
        lhat /= np.sqrt(np.sum(lhat ** 2))
        bhat = np.cross(lhat, rhat)
        return rhat, lhat, bhat

    def get_observables(self):
        """
        output:
        - measurement inputs to `ObservedStar` object: `(lb, dm, pm, rv)` (units deg, mag, mas yr^{-1}, km s^{-1})

        comments:
        - https://www.google.com/search?q=(1+kpc)+%2F+(1+km+%2F+s)+%2F+206254.81+%2F+1000.
        - 4.74080147 years

        bugs:
        - UNITS SUCK -- Units for `pm` need checking.
        """
        lb = self.get_lb()
        d = self.get_helio_distance()
        dm = 5. * np.log10(100. * d) # magic number 100 for kpc -> (10 pc)
        rhat, lhat, bhat = self.get_helio_unit_vectors()
        v = self.get_helio_3vel()
        rv = np.dot(v, rhat)
        pm = np.array([np.dot(v, lhat), np.dot(v, bhat)]) / d / 4.7408 # UNITS WRONG?
        return lb, dm, pm, rv

    def get_observables_array(self):
        """
        Same as `get_observables()` but returned as a 6-array.
        """
        lb, dm, pm, rv = self.get_observables()
        return np.array([lb[0], lb[1], dm, pm[0], pm[1], rv])

    def get_observables_names(self):
        return np.array([r"$l$", r"$b$", r"$DM$", r"$\dot{l}$", r"$\dot{b}$", r"$v_r$"])

    def get_observables_extents(self):
        """
        bugs:
        - Way hard-coded.
        """
        return [(0., 360.), (-90., 90.), (12.5, 20.5), (-2., 2.), (-2., 2.), (-500., 500.)]

    def get_potential_energy(self):
        """
        bugs:
        - Hard-coded to this potential.
        """
        return self.potential_amplitude * 0.5 * np.log(np.sum(self.get_sixpos()[:3] ** 2) / self.potential_scale ** 2)

    def get_kinetic_energy(self):
        """
        Duh.
        """
        return 0.5 * np.sum(self.get_sixpos()[3:] ** 2)

    def get_energy(self):
        """
        Duh.
        """
        return self.get_potential_energy() + self.get_kinetic_energy()

    def get_semimajor_axis(self):
        """
        Return the radius at which a star with this *energy* would be on a circular orbit.

        bugs:
        - Hard-coded to this potential.
        """
        return self.potential_scale * np.exp(self.get_energy() / self.potential_amplitude - 0.5)

    def get_angular_momentum(self):
        """
        Return the current value of the Galactocentric angular momentum (kpc km s^{-1}).
        """
        return np.cross(self.get_helio_3pos(), self.get_helio_3vel())

    def get_dimensionless_angular_momentum(self):
        """
        Return a dimensionless angular momentum quantity.

        bugs:
        - Hard-coded to this potential.
        """
        return np.sqrt(np.sum(self.get_angular_momentum() ** 2)
                       / self.potential_amplitude) / self.get_semimajor_axis()

    def _root_find(self, Q, x1, x2):
        """
        Very brittle root finder.
        """
        tol = 0.001
        Q1 = Q(x1)
        while x2 - x1 > tol:
            x3 = 0.5 * (x1 + x2)
            if Q(x3) / Q1 > 0.:
                x1 = x3
            else:
                x2 = x3
        return x3

    def get_eccentricity(self):
        """
        Return the generalized eccentricity.

        bugs:
        - Hard-coded to this potential (and doesn't need to be).
        - Uses root finding!
        """
        lna = np.log(self.get_semimajor_axis())
        L = np.sqrt(np.sum(self.get_angular_momentum() ** 2))
        Q = lambda lnr: lnr + 0.5 * L ** 2 / (self.potential_amplitude * np.exp(2. * lnr)) - lna - 0.5
        rperi = self._root_find(Q, lna - 64., lna)
        rap = self._root_find(Q, lna, lna + 64)
        return (rap - rperi) / (rap + rperi)

    def get_angular_momentum_angles(self):
        L = self.get_angular_momentum()
        absL = np.sqrt(np.sum(L ** 2))
        Lhat = L / absL
        lpole = np.rad2deg(np.arctan2(L[1], L[0]))
        while lpole < 0.:
            lpole += 360.
        bpole = np.rad2deg(np.arcsin(Lhat[2]))
        return np.array([lpole, bpole])

    def get_integrals_of_motion(self):
        return np.concatenate((np.array([self.get_semimajor_axis(),
                                         self.get_dimensionless_angular_momentum()]),
                               self.get_angular_momentum_angles()))

    def get_integrals_of_motion_names(self):
        return np.array([r"$a$", r"$|L|/(v_0^2\,a)$", r"$l^{(0)}$", r"$b^{(0)}$"])

    def get_integrals_of_motion_extents(self):
        """
        bugs:
        - Way hard-coded.
        """
        return [(0., 150.), (0., 3.), (0., 360.), (-90., 90.)]

class ObservedStar:

    def __init__(self, lb, lb_ivar, dm, dm_ivar, pm, pm_ivar, rv, rv_ivar):
        """
        inputs:
        - `lb`: measured celestial position (l, b) (deg)
        - `lb_ivar`: inverse variance tensor (2x2) for `lb` measurement (deg^{-2})
        - `dm`: inferred distance modulus (mag)
        - `dm_ivar`: inverse variance for `dm` measurement (mag^{-2})
        - `pm`: measured proper motion vector (l and b directions) (mas yr^{-1})
        - `pm_ivar`: inverse variance tensor (2x2) for `pm` measurement (mas^{-2} yr^2)
        - `rv`: measured radial velocity (km s^{-1}), positive red-shifted, negative blue-shifted
        - `rv_ivar`: inverse variance for `rv` measurement (km^{-2} s^2)

        output:
        - initialized `ObservedStar` object

        comments:
        - If a quantity (eg `pm`) is unmeasured, set corresponding inverse variance (eg `pm_ivar`) to zero.
        - pm quantity should be the isotropic angular velocity components, not just l-dot, b-dot (ie, there is a cosine in there).
        - ONLY works for distance modulus measurements, DOESN'T work for parallax measurements (yet).
        """
        self.prior_dbreak = 10. # kpc
        self.prior_vvariance = 100. * 100. # km^2 s^{-2}
        self.lb = lb
        self.lb_ivar = lb_ivar
        self.dm = dm
        self.dm_ivar = dm_ivar
        self.pm = pm
        self.pm_ivar = pm_ivar
        self.rv = rv
        self.rv_ivar = rv_ivar
        return None

    def get_fiducial_sixpos(self):
        """
        output:
        - SixPosition object corresponding to (lb, dm, pm, rv)

        bugs:
        - Units for `pm` need checking.
        """
        result = SixPosition(np.zeros(6))
        x = result.get_sixpos()
        x += result.get_sun_sixpos()
        d = 0.01 * 10. ** (0.2 * self.dm) # note (10 pc) -> kpc conversion
        x[0] += d * np.cos(np.deg2rad(self.lb[0])) * np.cos(np.deg2rad(self.lb[1]))
        x[1] += d * np.sin(np.deg2rad(self.lb[0])) * np.cos(np.deg2rad(self.lb[1]))
        x[2] += d * np.sin(np.deg2rad(self.lb[1]))
        rhat, lhat, bhat = result.get_helio_unit_vectors()
        x[3:] += self.rv * rhat
        x[3:] += self.pm[0] * d * lhat * 4.7408
        x[3:] += self.pm[1] * d * bhat * 4.7408
        return result

    def ln_prior(self, sixpos):
        """
        inputs:
        - `sixpos`: A `SixPosition` object.

        output:
        - log prior pdf evaluated at this sixposition

        comments:
        - totally made up and sucks!
        """
        foo = sixpos.get_sixpos()
        d = np.sqrt(np.sum(foo[:3] ** 2))
        if (d < self.prior_dbreak):
            ln_pos_prior = 0. # constant on the inside
        else:
            ln_pos_prior = 3. * np.log(self.prior_dbreak / d) # 1 / distance^3 on the outside
        ln_vel_prior = -0.5 * np.sum(foo[3:] ** 2) / self.prior_vvariance
        return ln_pos_prior + ln_vel_prior

    def ln_likelihood(self, sixpos):
        """
        inputs:
        - `sixpos`: A `SixPosition` object.

        output:
        - log likelihood that this star is explained by this sixposition.

        comments:
        - assumes all noise is Gaussian, duh.
        - assumes all `_ivar` variables are constants (not free parameters).
        """
        lb, dm, pm, rv = sixpos.get_observables()
        return -0.5 * (np.dot(np.dot(lb - self.lb, self.lb_ivar), lb - self.lb)
                       + self.dm_ivar * (dm - self.dm) ** 2
                       + np.dot(np.dot(pm - self.pm, self.pm_ivar), pm - self.pm)
                       + self.rv_ivar * (rv - self.rv) ** 2)

    def ln_posterior(self, sixpos):
        """
        The usual, now with `inf` catches.

        comments:
        - input is a `ndarray` not a `SixPosition` because of `emcee` requirements.
        """
        lnp = self.ln_prior(sixpos)
        if np.isfinite(lnp):
            return lnp + self.ln_likelihood(sixpos)
        return -np.inf

    def _get_samples(self, lnp, f=1):
        """
        Run emcee to generate samples.

        bugs:
        - Lots of things hard-coded.
        """
        ndim, nw, ns, nburn, nwburn, nsburn = 6, 32, f * 1024, 8, 16, 512
        pf = self.get_fiducial_sixpos().get_sixpos()
        amp = np.zeros_like(pf) + 0.1 # km s^{-1}
        amp[:3] = 1.e-6 # kpc
        p0 = [pf + amp * np.random.normal(ndim) for i in range(nwburn)]
        for b in range(nburn):
            sampler = emcee.EnsembleSampler(nwburn, ndim, lnp)
            sampler.run_mcmc(p0, nsburn)
            rindx = np.argsort(sampler.flatlnprobability)[-nsburn:]
            p0 = sampler.flatchain[rindx[np.random.randint(nsburn, size=(nwburn))], :]
        p1 = sampler.flatchain[np.random.randint(nwburn * nsburn, size=(nw)), :]
        sampler = emcee.EnsembleSampler(nw, ndim, lnp)
        sampler.run_mcmc(p1, ns)
        return sampler.chain, sampler.lnprobability

    def get_posterior_samples(self):
        """
        Run emcee to generate posterior samples of true position given measured position.
        """
        lnp = lambda sp: self.ln_posterior(SixPosition(sp))
        return self._get_samples(lnp)

    def get_prior_samples(self):
        """
        Run emcee to generate posterior samples of true position given measured position.
        """
        lnp = lambda sp: self.ln_prior(SixPosition(sp))
        return self._get_samples(lnp, f=32)

def unit_tests():
    """
    Run unit tests.

    output:
    - `True` or `False`, depending!

    bugs:
    - Needs left-handed vs right-handed coordinate system tests.
    - Needs to show that `pm` coordinate system is consistent with `lb` system.
    - Needs to test `pm` units by doing some fiducial 100 km / s at 100 kpc.
    """
    sixpos = SixPosition([0., 0., 0., 0., 0., 0.])
    lb, dm, pm, rv = sixpos.get_observables()
    if np.any(lb != [0., 0.]) or (rv != 0.):
        print lb, dm, pm, rv
        print "unit_tests(): (l,b) failed (0, 0) test"
        return False
    bar = sixpos.get_sun_sixpos()
    foo = 1. * bar # copy
    foo[3:] = [0., 0., 0.]
    sixpos = SixPosition(foo + np.array([0.,  10., 0., 0., 0., 0.]))
    lb, dm, pm, rv = sixpos.get_observables()
    if np.any(lb != [90., 0.]) or (rv != -bar[4]):
        print lb, dm, pm, rv
        print "unit_tests(): (l,b) failed (90, 0) test"
        return False
    sixpos = SixPosition(foo + np.array([0., -10., 0., 0., 0., 0.]))
    lb, dm, pm, rv = sixpos.get_observables()
    if np.any(lb != [270., 0.]) or (rv != bar[4]):
        print lb, dm, pm, rv
        print "unit_tests(): (l,b) failed (270, 0) test"
        return False
    sixpos = SixPosition(foo + np.array([0., 0.,  10., 0., 0., 0.]))
    lb, dm, pm, rv = sixpos.get_observables()
    if np.any(lb != [0., 90.]) or (rv != 0.):
        print lb, dm, pm, rv
        print "unit_tests(): (l,b) failed (0, 90) test"
        return False
    sixpos = SixPosition(foo + np.array([0., 0., -10., 0., 0., 0.]))
    lb, dm, pm, rv = sixpos.get_observables()
    if np.any(lb != [0., -90.]) or (rv != 0.):
        print lb, dm, pm, rv
        print "unit_tests(): (l,b) failed (0, 90) test"
        return False
    sixpos = SixPosition([-32., 45., 12., 115., 95., 160.])
    rhat, lhat, bhat = sixpos.get_helio_unit_vectors()
    tiny = 1e-15
    if ((np.abs(np.dot(rhat, rhat) - 1.) > tiny) or
        (np.abs(np.dot(lhat, lhat) - 1.) > tiny) or
        (np.abs(np.dot(bhat, bhat) - 1.) > tiny)):
        print np.dot(rhat, rhat) - 1., np.dot(lhat, lhat) - 1., np.dot(bhat, bhat) - 1.
        print "unit tests(): failed unit-vector normalization test"
        return False
    if ((np.abs(np.dot(rhat, lhat)) > tiny) or
        (np.abs(np.dot(rhat, bhat)) > tiny) or
        (np.abs(np.dot(lhat, bhat)) > tiny)):
        print rhat, lhat, bhat
        print "unit tests(): failed unit-vector orthogonality test"
        return False
    lb, dm, pm, rv = sixpos.get_observables()
    lb_ivar = np.diag([1e8, 1e8]) # deg^{-2}
    dm_ivar = 1. / (0.15 ** 2) # mag^{-2}
    pm_ivar = np.diag([0., 0.]) # mas^{-2} yr^2
    rv_ivar = 1. / (2. ** 2) # km^{-2} s^2
    star = ObservedStar(lb, lb_ivar, dm, dm_ivar, pm, pm_ivar, rv, rv_ivar)
    if np.any(np.abs(sixpos.get_sixpos() - star.get_fiducial_sixpos().get_sixpos()) > 1000. * tiny):
        print sixpos.get_sixpos(), star.get_fiducial_sixpos().get_sixpos()
        print "unit tests(): failed fiducial to star and back test"
        return False
    print "unit_tests(): all tests passed"
    return True

def triangle_plot_chain(chain, lnprob, prefix, truth=None):
    """
    Make a 7x7 triangle.
    """
    nx, nq = chain.shape
    maxlnp = np.max(lnprob)
    lnpextent = [(maxlnp-14.5, maxlnp+0.5)]
    bar = SixPosition(chain[0]) # temporary variable to get names
    foo = np.concatenate((chain, lnprob.reshape((nx, 1))), axis=1)
    if truth is None:
        truths = None
    else:
        truths = (np.append(truth, [maxlnp]), )
    labels = np.append(bar.get_sixpos_names(), [r"$\ln p$"])
    extents = bar.get_sixpos_extents() + lnpextent
    fig = tri.corner(foo, labels=labels, extents=extents,
                     truths=truths, plot_contours=False)
    fn = prefix + "a.png"
    print "triangle_plot_chain(): writing " + fn
    fig.savefig(fn)
    obsfoo = 1. * foo # copy
    for i in range(nx):
        obsfoo[i,:6] = SixPosition(foo[i,:6]).get_observables_array()
    if truth is None:
        trueobs = None
    else:
        trueobs = 1. * truths # copy
        trueobs[0,:6] = SixPosition(truth[:6]).get_observables_array()
    labels = np.append(bar.get_observables_names(), [r"$\ln p$"])
    extents = bar.get_observables_extents() + lnpextent
    fig = tri.corner(obsfoo, labels=labels, extents=extents,
                     truths=trueobs, plot_contours=False)
    fn = prefix + "b.png"
    print "triangle_plot_chain(): writing " + fn
    fig.savefig(fn)
    intfoo = 1. * foo[:,2:] # copy
    for i in range(nx):
        intfoo[i,:4] = SixPosition(foo[i,:6]).get_integrals_of_motion()
    if truth is None:
        trueint = None
    else:
        trueint = 1. * truths[:,2:] # copy
        trueint[0,:4] = SixPosition(truth).get_integrals_of_motion()
    labels = np.append(bar.get_integrals_of_motion_names(), [r"$\ln p$"])
    extents = bar.get_integrals_of_motion_extents() + lnpextent
    fig = tri.corner(intfoo, labels=labels, extents=extents,
                     truths=trueint, plot_contours=False)
    fn = prefix + "c.png"
    print "triangle_plot_chain(): writing " + fn
    fig.savefig(fn)
    return None

def figure_01():
    """
    Make figure 1. and maybe 2.

    comments:
    - Cut out stars from prior that are at |b| < 30 deg.
    - Cut out stars from prior that have heliocentric distances > 120 kpc.

    bugs:
    - HACK: NOT YET WRITTEN
    """
    print "hello world"
    sixpos = SixPosition([-12., 5., 22., 115., 20., 160.])
    lb, dm, pm, rv = sixpos.get_observables()
    lb_ivar = np.diag([1e9 * np.cos(np.deg2rad(lb[1])) ** 2, 1e9]) # deg^{-2}
    dm_ivar = 1. / (0.15 ** 2) # mag^{-2}
    pm_ivar = np.diag([0., 0.]) # mas^{-2} yr^2
    rv_ivar = 1. # km^{-2} s^2
    star = ObservedStar(lb, lb_ivar, dm, dm_ivar, pm, pm_ivar, rv, rv_ivar)
    chain, lnprob = star.get_prior_samples()
    nx, ny, nd = chain.shape
    chain = chain.reshape((nx * ny, nd))
    lnprob = lnprob.reshape((nx * ny))
    prior_obs = np.array([SixPosition(sp).get_observables_array() for sp in chain])
    good = np.logical_and(np.abs(prior_obs[:,1]) > 30., # deg
                          np.logical_and(prior_obs[:,2] > 13.5, # mag
                                         prior_obs[:,2] < 20.4)) # mag
    good[: nx * ny / 2] = False
    ngood = np.sum(good)
    indx = (np.arange(nx * ny))[good]
    chain = chain[indx, :]
    lnprob = lnprob[indx, :]
    triangle_plot_chain(chain, lnprob, "figure_01")
    for fig in range(8):
        sixpos = SixPosition(chain[np.random.randint(ngood)])
        lb, dm, pm, rv = sixpos.get_observables()
        star = ObservedStar(lb, lb_ivar, dm, dm_ivar, pm, pm_ivar, rv, rv_ivar)
        chain1, lnprob1 = star.get_posterior_samples()
        nx, ny, nd = chain1.shape
        chain1 = chain1.reshape((nx * ny, nd))
        lnprob1 = lnprob1.reshape((nx * ny))
        prefix = "figure_%02d" % (fig + 2)
        triangle_plot_chain(chain1[nx * ny / 2 :, :], lnprob1[nx * ny / 2 :],
                            prefix) # , truth=sixpos.get_sixpos())
    return None

if __name__ == "__main__":
    assert unit_tests()
    figure_01()
