"""
This file is part of the Kinematic Consensus project.
Copyright 2013 David W. Hogg (NYU)
"""

import numpy as np

class SixPosition:

    # the position of the Sun in 6-vector phase space
    # units [kpc, kpc, kpc, km s^{-1}, km s^{-1}, km s^{-1}]
    # needs to be checked!
    # needs to be consistent with get_lb() function
    sun = np.array([-8., 0., 0., 0., 225., 0.])

    def __init__(self, sixvector):
        """
        input:
        - `sixvector`: 6-element np array Galactocentric phase-space position, same units as `SixPosition.sun`

        output:
        - `SixPosition` object
        """
        self.pos = sixvector
        return None

    def get_helio_pos(self):
        """
        output:
        - heliocentric 6-vector phase-space position
        """
        return self.pos - SixPosition.sun

    def get_helio_3pos(self):
        """
        output:
        - heliocentric 3-vector spatial position
        """
        return get_helio_pos()[:3]

    def get_helio_3vel(self):
        """
        output:
        - heliocentric 3-vector velocity
        """
        return get_helio_pos()[3:]

    def get_lb(self):
        """
        output:
        - Galactic celestial coordinates (deg)
        """
        x = self.get_helio_3pos()
        return np.array([np.rad2deg(np.arctan(x[1], x[0])),
                         np.rad2deg(np.arcsin(x[2] / (x[0] ** 2 + x[1] ** 2 + x[2] ** 2)))])

    def get_distance(self):
        """
        output:
        - heliocentric distance
        """
        return np.sqrt(np.sum(self.get_helio_3pos() ** 2))

    def get_observables(self):
        """
        output:
        - measurement inputs to `Star` object: `(lb, dm, pm, rv)`

        bugs:
        - Doesn't compute `pm` because I SUCK.
        """
        lb = self.get_lb()
        d = self.get_distance()
        dm = 5. * np.log10(100. * d) # magic number 100 for kpc -> (10 pc)
        pm = np.array([0., 0.]) # HACK
        rv = np.dot(self.get_helio_3vel(), self.get_helio_3pos()) / d
        return rd, dm, pm, rv

class Star:

    def __init__(self, rd, rd_ivar, dm, dm_ivar, pm, pm_ivar, rv, rv_ivar):
        """
        inputs:
        - `lb`: measured celestial position (l, b) (deg)
        - `lb_ivar`: inverse variance tensor (2x2) for `lb` measurement (deg^{-2})
        - `dm`: inferred distance modulus (mag)
        - `dm_ivar`: inverse variance for `dm` measurement (mag^{-2})
        - `pm`: measured proper motion vector (alphadot, deltadot) (mas yr^{-1})
        - `pm_ivar`: inverse variance tensor (2x2) for `pm` measurement (mas^{-2} yr^2)
        - `rv`: measured radial velocity (km s^{-1})
        - `rv_ivar`: inverse variance for `rv` measurement (km^{-2} s^2)

        output:
        - initialized `Star` object

        comments:
        - If a quantity (eg `pm`) is unmeasured, set corresponding inverse variance (eg `pm_ivar`) to zero.
        - ONLY works for distance modulus measurements, DOESN'T work for parallax measurements (yet).
        """
        self.lb = lb
        self.lb_ivar = lb_ivar
        self.dm = dm
        self.dm_ivar = dm_ivar
        self.pm = pm
        self.pm_ivar = pm_ivar
        self.rv = rv
        self.rv_ivar = rv_ivar
        return None

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
        rd, dm, pm, rv = sixpos.get_observables()
        return -0.5 * (np.dot(np.dot(rd - self.rd, self.rd_ivar), rd - self.rd)
                       + self.dm_ivar * (dm - self.dm) ** 2
                       + np.dot(np.dot(pm - self.pm, self.pm_ivar), pm - self.pm)
                       + self.rv_ivar * (rv - self.rv) ** 2
