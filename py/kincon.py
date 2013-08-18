"""
This file is part of the Kinematic Consensus project.
Copyright 2013 David W. Hogg (NYU)
"""

import numpy as np
import emcee

class SixPosition:

    def __init__(self, sixvector):
        """
        input:
        - `sixvector`: 6-element np array Galactocentric phase-space position (kpc, km s^{-1})

        output:
        - `SixPosition` object
        """
        assert len(sixvector) == 6
        self.pos = np.array(sixvector)
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
        return np.array([-8., 0., 0., 0., 225., 0.])

    def get_helio_pos(self):
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
        return self.get_helio_pos()[:3]

    def get_helio_3vel(self):
        """
        output:
        - heliocentric 3-vector velocity (km s^{-1})
        """
        return self.get_helio_pos()[3:]

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

    def get_distance(self):
        """
        output:
        - heliocentric distance (kpc)
        """
        return np.sqrt(np.sum(self.get_helio_3pos() ** 2))

    def get_observables(self):
        """
        output:
        - measurement inputs to `ObservedStar` object: `(lb, dm, pm, rv)` (units deg, mag, mas yr^{-1}, km s^{-1})

        bugs:
        - Doesn't compute `pm` because I SUCK.
        """
        lb = self.get_lb()
        d = self.get_distance()
        dm = 5. * np.log10(100. * d) # magic number 100 for kpc -> (10 pc)
        dhat = self.get_helio_3pos() / d
        v = self.get_helio_3vel()
        rv = np.dot(v, dhat)
        pm = np.array([0., 0.]) # HACK
        return lb, dm, pm, rv

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
        - HACK: NOT YET WRITTEN
        """
        return None

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
        if (d2 < self.prior_dmin ** 2) or (d2 > self.prior_dmax ** 2):
            return -np.inf
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

    def ln_posterior(self, sixpos):
        lnp = self.ln_prior(sixpos)
        if np.isfinite(lnp):
            return lnp + self.ln_likelihood(sixpos)
        return -np.inf

    def get_posterior_samples(self):
        """
        Run emcee to generate posterior samples of true position given measured position.

        bugs:
        - HACK: NOT YET WRITTEN
        """
        # setup emcee
        # initialize walkers
        # burn in
        # sample
        return None

def unit_tests():
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
    print "unit_tests(): all tests passed"
    return True

def figure_01():
    """
    Make figure 1.
    """
    print "hello world"
    sixpos = SixPosition([-32., 45., 12., 115., 95., 160.])
    lb, dm, pm, rv = sixpos.get_observables()
    lb_ivar = np.diag([1e9, 1e9]) # deg^{-2}
    dm_ivar = 1. / (0.15 ** 2) # mag^{-2}
    pm_ivar = np.diag([0., 0.]) # mas^{-2} yr^2
    rv_ivar = 1. # km^{-2} s^2
    star = ObservedStar(lb, lb_ivar, dm, dm_ivar, pm, pm_ivar, rv, rv_ivar)
    print star.ln_posterior(sixpos)
    return None

if __name__ == "__main__":
    unit_tests()
    figure_01()
