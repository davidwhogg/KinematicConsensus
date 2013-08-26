"""
This file is part of the Kinematic Consensus project.
Copyright 2013 David W. Hogg (NYU)
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
        self.potential_amplitude = 200.**2 # km^2 s^{-2}
        return None

    def get_sixpos(self):
        """
        output:
        - 6-element np array Galactocentric phase-space position (kpc, km s^{-1})
        """
        return self.pos

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
        return np.array([l, np.rad2deg(np.arcsin(x[2] / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)))])

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

    def get_potential_energy(self):
        return self.potential_amplitude * 0.5 * np.log(np.sum(self.get_sixpos()[:3] ** 2) / 200. ** 2) # MAGIC NUMBER 200 kpc

    def get_kinetic_energy(self):
        return 0.5 * np.sum(self.get_sixpos()[3:] ** 2)

    def get_energy(self):
        return self.get_potential_energy() + self.get_kinetic_energy()

    def get_angular_momentum(self):
        return np.cross(self.get_helio_3pos(), self.get_helio_3vel())

    def get_integrals_of_motion(self):
        return np.concatenate(([self.get_energy()], self.get_angular_momentum()))

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
        self.prior_dmin = 0.02 # kpc
        self.prior_dmax = 200. # kpc
        self.prior_vvariance = 150. * 150. # km^2 s^{-2}
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
        d2 = np.sum(foo[:3] ** 2)
        if (d2 > self.prior_dmax ** 2):
            return -np.inf
        if (d2 < self.prior_dmin ** 2):
            ln_pos_prior = 1. / self.prior_dmin ** 2
        else:
            ln_pos_prior = 1. / d2
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

    def ln_posterior(self, sp):
        """
        The usual, now with `inf` catches.

        comments:
        - input is a `ndarray` not a `SixPosition` because of `emcee` requirements.
        """
        sixpos = SixPosition(sp)
        lnp = self.ln_prior(sixpos)
        if np.isfinite(lnp):
            return lnp + self.ln_likelihood(sixpos)
        return -np.inf

    def get_posterior_samples(self, nsamples):
        """
        Run emcee to generate posterior samples of true position given measured position.

        bugs:
        - Lots of things hard-coded.
        """
        ndim, nwalkers = 6, 16
        pf = self.get_fiducial_sixpos().get_sixpos()
        p0 = [pf + 1e-6 * np.random.normal(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_posterior)
        sampler.run_mcmc(p0, nsamples)
        return sampler.chain, sampler.lnprobability

    def get_prior_samples(self, nsamples):
        """
        Run emcee to generate posterior samples of true position given measured position.

        bugs:
        - Lots of things hard-coded.
        """
        ndim, nwalkers = 6, 16
        pf = self.get_fiducial_sixpos().get_sixpos()
        p0 = [pf + 1e-6 * np.random.normal(ndim) for i in range(nwalkers)]
        lnp = lambda sp: self.ln_prior(SixPosition(sp))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnp)
        sampler.run_mcmc(p0, nsamples)
        return sampler.chain, sampler.lnprobability

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
    lb_ivar = np.diag([1e9, 1e9]) # deg^{-2}
    dm_ivar = 1. / (0.15 ** 2) # mag^{-2}
    pm_ivar = np.diag([0., 0.]) # mas^{-2} yr^2
    rv_ivar = 1. # km^{-2} s^2
    star = ObservedStar(lb, lb_ivar, dm, dm_ivar, pm, pm_ivar, rv, rv_ivar)
    if np.any(np.abs(sixpos.get_sixpos() - star.get_fiducial_sixpos().get_sixpos()) > 1000. * tiny):
        print sixpos.get_sixpos(), star.get_fiducial_sixpos().get_sixpos()
        print "unit tests(): failed fiducial to star and back test"
        return False
    print "unit_tests(): all tests passed"
    return True

def triangle_plot_chain(chain, lnprob, prefix):
    """
    Make a 7x7 triangle.
    """
    nx, nq = chain.shape
    foo = np.concatenate((chain, lnprob.reshape((nx, 1))), axis=1)
    labels = [r"$x$", r"$y$", r"$z$", r"$v_x$", r"$v_y$", r"$v_z$", r"$\ln p$"]
    fig = tri.corner(foo.transpose(), labels=labels)
    fn = prefix + "a.png"
    print "triangle_plot_chain(): writing " + fn
    fig.savefig(fn)
    obsfoo = 1. * foo # copy
    for i in range(nx):
        obsfoo[i,:6] = SixPosition(foo[i,:6]).get_observables_array()
    labels = [r"$l$", r"$b$", r"$DM$", r"$\dot{l}$", r"$\dot{b}$", r"$v_r$", r"$\ln p$"]
    fig = tri.corner(obsfoo.transpose(), labels=labels)
    fn = prefix + "b.png"
    print "triangle_plot_chain(): writing " + fn
    fig.savefig(fn)
    intfoo = 1. * foo[:,2:] # copy
    for i in range(nx):
        intfoo[i,:4] = SixPosition(foo[i,:6]).get_integrals_of_motion()
    labels = [r"E", r"$L_x$", r"$L_y$", r"$L_z$", r"$\ln p$"]
    fig = tri.corner(intfoo.transpose(), labels=labels)
    fn = prefix + "c.png"
    print "triangle_plot_chain(): writing " + fn
    fig.savefig(fn)
    return None

def figure_01():
    """
    Make figure 1.

    bugs:
    - HACK: NOT YET WRITTEN
    """
    print "hello world"
    sixpos = SixPosition([-12., 5., 22., 115., 20., 160.])
    lb, dm, pm, rv = sixpos.get_observables()
    lb_ivar = np.diag([1e9 * np.cos(np.deg2rad(lb[1])) ** 2, 1e9]) # deg^{-2}
    dm_ivar = 1. / (0.30 ** 2) # mag^{-2}
    pm_ivar = np.diag([0., 0.]) # mas^{-2} yr^2
    rv_ivar = 1. # km^{-2} s^2
    star = ObservedStar(lb, lb_ivar, dm, dm_ivar, pm, pm_ivar, rv, rv_ivar)
    N = 4096
    chain, lnprob = star.get_prior_samples(N)
    nx, ny, nd = chain.shape
    chain = chain.reshape((nx * ny, nd))
    lnprob = lnprob.reshape((nx * ny))
    lbs = np.array([SixPosition(sp).get_observables()[0] for sp in chain])
    good = np.abs(lbs[:,1]) < 30. # deg
    chain = chain[good]
    lnprob = lnprob[good]
    triangle_plot_chain(chain, lnprob, "figure_01")
    chain, lnprob = star.get_posterior_samples(N)
    nx, ny, nd = chain.shape
    chain = chain.reshape((nx * ny, nd))
    lnprob = lnprob.reshape((nx * ny))
    triangle_plot_chain(chain, lnprob, "figure_02")
    return None

if __name__ == "__main__":
    assert unit_tests()
    figure_01()
