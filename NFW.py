#!/usr/bin/env python

##################################################################
#
# This module is the class of Navarro-Frenk-White Model.
#
##################################################################

import numpy as np
from math import *
import cosmolopy.distance as cosdist
import cosmolopy.density as cosdens
import cosmolopy.constants as cosconst
import scipy.optimize as optimize
import scipy.integrate as integrate
import scipy.stats as stats
import scipy.interpolate as interpolate

# ---
# Set a default cosmology
# ---
cosmo           =       {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.7}
cosmo           =       cosdist.set_omega_k_0(cosmo)

# ---
# Define the vir_overden
# ---
def calc_vir_overden(zd, cosmo = cosmo):
    """
    This is the overdensity wrt to mean density of the Universe at redshift zd when the halo is viriliazed.
    This is a fitting formula given in equation C19 in Nakamura and Suto (1997).

    Parameters:
        -`zd`: float. The halo redshift.
        -`cosmo`: dict. The cosmology parameter for this halo. It has to be compatible with
        the format of the cosmolopy module.
    """
    return 18.0*pi*pi*( 1.0 + 0.4093 * (cosdens.omega_M_z(zd,**cosmo)**-1 - 1.0)**0.9052 )

# ---
# Define configuration convertor
# ---
def MXXXtoMYYY(redshift =   0.3,
               MXXX     =   6E14,
               CXXX     =   3.0,
               wrt      =   "crit",
               new_wrt  =   "crit",
               XXX      =   500.0,
               YYY      =   500.0,
               cosmo    =   cosmo):
    """
    It converts the (MXXX,CXXX) into (MYYY,CYYY) for a halo at a given redshift assuming spherical sysmetry.
    (1) XXX and YYY are float and negative value means virial estimation.
    (2) it returns (the MXXX,CXXX,RXXX,MYYY,CYYY,RYYY,rhos,rs)
        where MXXX, MYYY are in the Msun, RXXX and RYYY are in Mpc.
        rhos is in Msun/Mpc^3 and rs is in Mpc.

    ---

    a. It first solves the NFW model parameters - rhos, rs.
        rhos =  FactorO(redshift,XXX) * (CXXX**3 / 3) / (log(1+CXXX) - CXXX/(1+CXXX))
        where FactorO(z,XXX) = OverDen(z) * rho_m(z)        if XXX  = VIR
        XXX        * rho_crit(z)     if XXX != VIR
    b. It solves for CYYY:
        FactorO(redshift,YYY) * (CYYY**3 / 3) / (log(1+CYYY) - CYYY/(1+CYYY))
        = FactorO(redshift,XXX) * (CXXX**3 / 3) / (log(1+CXXX) - CXXX/(1+CXXX))

    c. Solve for rs:
        rs**3 = MXXX / ( 4*pi*rhos*(log(1+CXXX) - CXXX/(1+CXXX)) )

    d. MYYY = 4*pi*rhos*rs**3 * [log(1+CYYY) - CYYY/(1+CYYY)]

    e. RYYY = CYYY * rs and RXXX = CXXX * rs

    ---

    Parameters:
        -`redshift`: float, the redshift of the halo.
        -`MXXX`: float, the mass of the halo in the unit of Msun.
        -`CXXX`: float, the concentration of the halo associating to the overdensity of the input mass.
        -`wrt`: string. It has to be either 'crit' or 'mean'. It will be overwritten as 'vir' if XXX < 0.0.
        -`new_wrt`: string. Same as above, but it will be overwritten as 'vir' if YYY < 0.0.
        -`XXX`: float, the overdensity against the rho_crit for the given input halo mass.
                Negative if it is with respect to the virial mass.
        -`YYY`: float, the overdensity against the rho_crit for the desired output halo mass.
                Negative if it is with respect to the virial mass.
        -`cosmo`: dict. The cosmology parameter for this halo. It has to be compatible with
                the format of the cosmolopy module.

    Return:
        -`MXXX`: float, the input halo mass in the unit of Msun.
        -`CXXX`: float, the input concentration.
        -`RXXX`: float, the radius for the given input halo mass in the unit of Mpc.
        -`MYYY`: float, the output halo mass in the unit of Msun.
        -`CYYY`: float, the output concentration.
        -`RYYY`: float, the radius for the output halo mass in the unit of Mpc.
        -`rhos`: float, the normalization of the NFW model (Msun/Mpc^3)
        -`rs`: float, the core radius of the NFW model (Mpc).

    """
    # sanitiy check
    if  not ( (redshift >= 0.0) and (MXXX > 0.0) and (CXXX > 0.0) ):
        raise ValueError("The input halo params are wrong, (redshift, MXXX, CXXX):", redshift, MXXX, CXXX, ".")

    # sanity check on wrt
    if  wrt not in ["crit", "mean"]:
        raise NameError("The input wrt", wrt, "has to be crit or mean.")
    if  new_wrt not in ["crit", "mean"]:
        raise NameError("The input new_wrt", new_wrt, "has to be crit or mean.")

    # set up wrt
    if  XXX < 0.0: wrt     = "vir"
    if  YYY < 0.0: new_wrt = "vir"

    # Define the function form for the convenience
    def FactorCC(CXXX):
        return (CXXX**3 / 3.0) / (log(1.0+CXXX) - CXXX / (1.0+CXXX))

    # Define the function FactorO
    def FactorO(redshift, XXX, wrt):
        if     wrt   ==    "crit":
            return XXX * \
                   cosdens.cosmo_densities(**cosmo)[0] * cosdist.e_z(redshift, **cosmo)**2
        elif   wrt   ==    "mean":
            return XXX * \
                   cosdens.cosmo_densities(**cosmo)[0] * cosdist.e_z(redshift, **cosmo)**2 * cosdens.omega_M_z(redshift, **cosmo)
        elif   wrt   ==    "vir":
            return calc_vir_overden(zd = redshift, cosmo = cosmo) * \
                   cosdens.cosmo_densities(**cosmo)[0] * cosdist.e_z(redshift, **cosmo)**2 * cosdens.omega_M_z(redshift, **cosmo)

    # Solve rhos Msun/Mpc^3
    rhos    =   FactorO(redshift = redshift, XXX = XXX, wrt = wrt) * FactorCC(CXXX = CXXX)

    #Define the function we solve for CYYY:
    def Solve4CYYY_Func(CYYY):
        return FactorO(redshift = redshift, XXX = YYY, wrt = new_wrt) * FactorCC(CXXX = CYYY) - \
               FactorO(redshift = redshift, XXX = XXX, wrt =     wrt) * FactorCC(CXXX = CXXX)

    # Solve for CYYY
    CYYY = optimize.newton(Solve4CYYY_Func, CXXX, fprime=None, args=(), tol=1.48e-08, maxiter=50)

    #Solve for rs [Mpc]
    rs   =   ( MXXX / ( 4.0 * pi * rhos * ( log(1.0+CXXX) - CXXX/(1.0+CXXX) ) ) )**(1.0/3.0)

    #Solve for MYYY [Msun]
    MYYY =   4.0 * pi * rhos * rs**3 * ( log(1.0+CYYY) - CYYY/(1.0+CYYY) )

    #Solve for RXXX and RYYY [Mpc]
    RXXX = CXXX * rs
    RYYY = CYYY * rs

    return np.array([ MXXX, CXXX, RXXX, MYYY, CYYY, RYYY, rhos, rs ], dtype=float)


# ---
# Deprojection Factor
# ---
def DeprojectFactor(concen):
    """
    This method calculates the deproject factor witin the radius defining C = r/rs
    for a given NFW model parameters.
    Namely,
    DeprojectFactor =   int_{0}^{Concen}{rhoNFW(r) * 4*pi*r^2 dr} / int_{0}^{Concen}{Kappa * 2*pi*r dr}
    This factor only depends on the concentration only, normalization doesn't matter.

    Parameters:
        -`concen`: nd array. The concentration parameters.
    Return:
        -`Dpj`: nd array. The deprojection factor.
    """
    # Define the internal function to describe the profile of Kappa.
    def Factor1(X):
        if X < 1.0:
            insideF = (1.0 - X)/(1.0 + X)
            return 1.0 / (1.0 - X**2) * ( -1.0 + 2.0/sqrt(1.0-X**2) * atanh( sqrt(insideF) ) )
        elif X == 1.0:
            return 1.0/3.0
        else:
            insideF = (X - 1.0)/(1.0 + X)
            return 1.0 / (X**2 - 1.0) * (  1.0 - 2.0/sqrt(X**2-1.0) * atan( sqrt(insideF) ) )

    # sanitize
    concen  =   np.array(concen, ndmin=1)

    # calc the numerator, namely the normalized mass within the radus defining C.
    numerator   =   np.log(1.0 + concen) + 1.0/(1.0+concen) - 1.0 # dont forget to minus 1.
    # calculate the denominator
    denominator =   np.array([
                    integrate.quad( lambda x: Factor1(x) * x, 0.0, cc)[0]
                    for cc in concen
                    ], ndmin=1)

    return numerator / denominator


# ---
# Offset distribution
# ---
def Poffset(rmpc_offset, mis_type = "zitrin+12"):
    """
    This function describes the mis-centering of galaxy clusters.

    It returns the probability of the offset of the center proxy as a function of Mpc.

    Parameters:
        -`rmpc_offset`: float or 1d array. The radial offset in the unit of Mpc.
        -`mis_type`: string. The functional form of offset distribution in different literature.

    Return:
        -`Poffset`: float or 1d array. The offset distribution.
    """
    # sanitize
    if      mis_type    not in ["zitrin+12",]:
        raise NameError("mis_type:", mis_type, "is not in the list.")
    # sanitize
    rmpc_offset    =   np.array(rmpc_offset, ndmin=1)

    # Zitrin+12, 1208.1766v3
    if    mis_type  ==  "zitrin+12":
        # define P_zitrin
        def P_zitrin(x):
            """
            In their Figure 5, it is lognormal distribution.

            u := <log10(Delta)> = -1.895
            and
            sigma := sigma of log10(Delta) = 0.501

            In linear space,

            dP/dx = dP/dy * 1.0 / (log(10.0) * 10.0**y),
            where y = log10(x)
            """
            y       =   np.log10(x)
            dPdx    =   stats.norm.pdf(x = y, loc = -1.895, scale = 0.501) * 1.0 / (log(10.0) * 10.0**y)
            return dPdx

        return P_zitrin(x = rmpc_offset)

    return

####################################
#
# NFW model class
#
####################################

class Halo(object):
    """
    This is the class for NFW model (Navarro+1997).
    The basic idea is the following. Each halo can be described by NFW model and an instance
    of this NFW class. For given redshift, cosmology and halo mass, then the halo properties
    can be well derived. Furthermore, the profile-like properties can be further calculated
    by assuming a concentration.

    Therefore, (zd, cosmo, mass, concen, overdensity, wrt) -- NFW --> halo properties.
    """

    # ---
    # initialize the set ups
    # ---
    def __init__(self, zd          =       0.3,
                       mass        =       6E14,
                       concen      =       3.0,
                       overden     =       500.0,
                       wrt         =       "crit",
                       cosmo       =       cosmo):
        """
        This is the initialization of a NFW instance.

        Parameters:
            -`zd`: float. The halo redshift.
            -`mass`: float. The halo mass in the unit of Msun.
                     Note, it is NOT in the unit of Msun/h.
            -`concen`: float. Concentration parameter.
            -`overden`: float. The overdensity wrt the critical or mean density of the Universe.
                        If overden < 0, then it stands for the `virial` mass (so that the `wrt`
                        does not have impact).
            -`wrt`: string. The type which the overdensity with respect to. It is either
                    `crit` or `mean`. If `overden` < 0.0, then the `wrt` has no effect.
            -`cosmo`: dict. The cosmology parameter for this halo. It has to be compatible with
            the format of the cosmolopy module.
        """
        # sanity check
        if  not ( (zd >=  0.0) and (mass > 0.0) and (concen > 0.0) ):
            raise ValueError("The input halo params are wrong, (zd, mass, concen):", zd, mass, concen, ".")

        if  wrt not in ["crit", "mean"]:
            raise NameError("The input wrt", wrt, "has to be crit or mean.")

        # initiate
        self.zd         =       float(zd)
        self.mass       =       float(mass)
        self.concen     =       float(concen)
        self.overden    =       float(overden)
        self.wrt        =       str(wrt)
        self.cosmo      =       cosdist.set_omega_k_0(cosmo)                # set cosmology
        self.vir_overden=       calc_vir_overden(zd = self.zd, cosmo = self.cosmo)     # calculate virialized overden

        # cosmology properties
        self.da         =       cosdist.angular_diameter_distance(self.zd, **self.cosmo)
        self.ez         =       cosdist.e_z(self.zd, **self.cosmo)
        self.rho_crit   =       cosdens.cosmo_densities(**self.cosmo)[0] * self.ez**2    # Msun/Mpc3
        self.rho_mean   =       self.rho_crit * cosdens.omega_M_z(self.zd, **self.cosmo) # Msun/Mpc3
        self.arcmin2mpc =       1.0 / 60.0 * pi / 180.0  * self.da                       # Mpc/arcmin

        # set up the compatibility between overden and wrt
        if     self.overden  <  0.0:  self.wrt    =       "vir"

        # set up halo radius / rhos[Msun/Mpc^3] / rs[Mpc]
        if   self.wrt   ==      "mean":
            self.radmpc           =   ( self.mass / (4.0 * pi / 3.0 * self.overden     * self.rho_mean) )**(1.0/3.0)
            self.radarcmin        =   self.radmpc / self.arcmin2mpc
            self._factorO         =   self.overden * self.rho_mean
            self.rhos             =   self.overden * self.rho_mean * (self.concen**3 / 3.0) / ( log(1.0 + self.concen) - self.concen/(1.0 + self.concen) )
            self.rs                =   ( self.mass / ( 4.0 * pi * self.rhos * ( log(1.0 + self.concen) - self.concen/(1.0 + self.concen) ) ) )**(1.0/3.0)

        elif self.wrt   ==      "crit":
            self.radmpc           =   ( self.mass / (4.0 * pi / 3.0 * self.overden     * self.rho_crit) )**(1.0/3.0)
            self.radarcmin        =   self.radmpc / self.arcmin2mpc
            self._factorO         =   self.overden * self.rho_crit
            self.rhos             =   self.overden * self.rho_crit * (self.concen**3 / 3.0) / ( log(1.0 + self.concen) - self.concen/(1.0 + self.concen) )
            self.rs                =   ( self.mass / ( 4.0 * pi * self.rhos * ( log(1.0 + self.concen) - self.concen/(1.0 + self.concen) ) ) )**(1.0/3.0)

        elif self.wrt   ==      "vir":
            self.radmpc           =   ( self.mass / (4.0 * pi / 3.0 * self.vir_overden * self.rho_mean) )**(1.0/3.0)
            self.radarcmin        =   self.radmpc / self.arcmin2mpc
            self._factorO         =   self.vir_overden * self.rho_mean
            self.rhos             =   self.vir_overden * self.rho_mean * (self.concen**3 / 3.0) / ( log(1.0 + self.concen) - self.concen/(1.0 + self.concen) )
            self.rs                =   ( self.mass / ( 4.0 * pi * self.rhos * ( log(1.0 + self.concen) - self.concen/(1.0 + self.concen) ) ) )**(1.0/3.0)

        # return
        return

    # ---
    # Diagnostic information
    # ---
    def i_am(self):
        """
        Just print the diagnostic information
        """
        # print the information on screen.
        print
        print "#", "mass is in the unit of Msun."
        print "#", "length is in the unit of Mpc."
        print "#", "density is in the unit of Msun/Mpc^3."
        for name_of_attribute in ['zd',
                                  'mass',
                                  'concen',
                                  'overden',
                                  'wrt',
                                  'vir_overden',
                                  'rhos',
                                  'rs',
                                  'radmpc',
                                  'radarcmin',
                                  'arcmin2mpc',
                                  'ez',
                                  'rho_crit',
                                  'rho_mean',
                                  'cosmo',
                                  ]:
            print "#", name_of_attribute, ":", getattr(self, name_of_attribute)
        print

        # return
        return


    # ---
    # Calculate the different mass configuration
    # ---
    def calc_halo_config(self, new_overden, new_wrt):
        """
        This method converts the (mass, concen) into another (new_mass, new_concen)
        with `new_overden` and `new_wrt` assuming spherical sysmetry.

        Parameters:
            -`new_overden`: float. The overdensity with respect to `new_wrt`.
                            Negative stands for the virial mass estimates.
            -`new_wrt`: string. It has to be either 'crit' or 'mean'.
                        It will overwritten if `new_overden` < 0.0

        Return:
            -`new_mass`: float. The mass whose has `new_overden` with respect to `new_wrt`.
                         It is in the unit of Msun.
            -`new_concen`: float. Same above, but it is the concentration parameter (i.e., new_radmpc/rs).
            -`new_radmpc`: float. Same above, but it is the radius in the unit of Mpc.
        """
        # sanity check
        if  new_wrt not in ["crit", "mean"]:
            raise NameError("The input new_wrt", new_wrt, "has to be crit or mean.")

        # derive the new configuration.
        if    self.overden  <   0.0:
            new_mass, new_concen, new_radmpc        =       \
            MXXXtoMYYY(redshift = self.zd,
                       MXXX     = self.mass,
                       CXXX     = self.concen,
                       wrt      = "crit",
                       new_wrt  = new_wrt,
                       XXX      = -1,
                       YYY      = new_overden,
                       cosmo    = self.cosmo)[[3,4,5]]
        else:
            new_mass, new_concen, new_radmpc        =       \
            MXXXtoMYYY(redshift = self.zd,
                       MXXX     = self.mass,
                       CXXX     = self.concen,
                       wrt      = self.wrt,
                       new_wrt  = new_wrt,
                       XXX      = self.overden,
                       YYY      = new_overden,
                       cosmo    = self.cosmo)[[3,4,5]]

        # return
        return new_mass, new_concen, new_radmpc


    # ---
    # Density profile
    # ---
    def Density3DPrfl(self, rmpc):
        """
        This method calculates density profile given the radius r.

        Parameters:
            -`rmpc`: float or nd array. The radius in the unit of Mpc.
        Return:
            -`density`: nd array. The returned density profile in the unit of Msun/Mpc^3.
        """
        # sanitize
        rmpc    =   np.array(rmpc, ndmin=1)
        # return
        return self.rhos / ( rmpc/self.rs * (1.0 + rmpc/self.rs)**2.0 )

    # ---
    # Projected surface density profile
    # ---
    def Density2DPrfl(self, rmpc):
        """
        This method calculates the projected surface mass density profile.

        ---

        Sigma(X) =   2.0 * rhos * rs * Factor1(X)
        where
        Factor1(X) = 1.0 / (1.0 - X**2) * ( -1.0 + 2.0/sqrt(1.0-X**2) * atanh( sqrt(insideF) ) ) for X < 1.0
                   = 1.0/3.0 for X == 1
                   = 1.0 / (X**2 - 1.0) * (  1.0 - 2.0/sqrt(X**2-1.0) * atan(  sqrt(insideF) ) ) for X > 1.0

        insideF = (1.0 - X)/(1.0 + X)
        X = r / rs

        ---

        Parameters:
            -`rmpc`: float or nd array. The radius in the unit of Mpc.
        Return:
            -`2ddensity`: nd array. The returned density profile in the unit of Msun/Mpc^2.

        """
        # sanitize
        rmpc    =   np.array(rmpc, ndmin=1)

        # calculate the normalization.
        Norm    =   2.0 * self.rhos * self.rs

        # define internal functions
        def factor_smaller_than_one(X):
            X       = np.array(X, ndmin=1)
            insideF = (1.0 - X)/(1.0 + X)
            return 1.0 / (1.0 - X**2) * ( -1.0 + 2.0/np.sqrt(1.0-X**2) * np.arctanh( np.sqrt(insideF) ) )

        def factor_larger_than_one(X):
            X       = np.array(X, ndmin=1)
            insideF = (X - 1.0)/(1.0 + X)
            return 1.0 / (X**2 - 1.0) * (  1.0 - 2.0/np.sqrt(X**2-1.0) * np.arctan(  np.sqrt(insideF) ) )

        # calculate the profile
        profile_smaller_than_one     =      factor_smaller_than_one(X = rmpc/self.rs)
        profile_equal_to_one         =      1.0/3.0 * np.ones(len(rmpc))
        profile_larger_than_one      =      factor_larger_than_one(X = rmpc/self.rs)

        # calculate the final profile
        profile =   np.copy( profile_equal_to_one )
        profile[ (rmpc/self.rs > 1.0) ] =   np.copy( profile_larger_than_one[  (rmpc/self.rs > 1.0) ] )
        profile[ (rmpc/self.rs < 1.0) ] =   np.copy( profile_smaller_than_one[ (rmpc/self.rs < 1.0) ] )

        # return
        return Norm * profile


    # ---
    # Beta - lensing efficiency
    # ---
    def Beta(self, zs = 1.0):
        """
        This function calculates the lensing efficiency defined as the following
        beta = Dds / Ds

        Parameters:
            -`zs`: nd array. source redshift.

        Return:
            -`lensing_efficiency`: nd array. the lensing efficiency.
        """
        # sanitize
        zs = np.array(zs, ndmin = 1, dtype = np.float)

        # calc Dds
        Dds =   cosdist.angular_diameter_distance(zs, self.zd, **self.cosmo)

        # sanitize the cases of zs <= zd
        Dds[ (zs <= self.zd) ] = 0.0

        # calc Ds
        Ds  =   cosdist.angular_diameter_distance(zs, 0.0, **self.cosmo)

        # return
        return Dds/Ds


    # ---
    # Sigma_crit - critical surface density, which is used as the normalization of kappa in lensing.
    # ---
    def Sigma_crit(self, zs = 1.0, beta = None):
        """
        This function calculates the critical surface density [Msun/Mpc^2] defined as the following
        Sigma_crit = (c**2 / 4.0 * pi* G) * (Dd * lensing efficiency)**-1
        You can specify the lensing efficiency, beta, yourself.

        Parameters:
            -`zs`: nd array. The redshift of the source.
            -`beta`: nd array. The lensing efficiency.
            If it is None, then calculate the lensing efficiency from zd and zs.

        Return:
            -`Sigma_crit`: nd array, the critical surface mass density.

        """
        # sanitize
        zs          =   np.array(zs, ndmin=1)

        # calc beta
        if beta is None:
            beta    =   self.Beta(zs = zs)
        else:
            beta    =   np.array(beta, ndmin=1)

        # return
        return cosconst.c_light_Mpc_s**2 / ( 4.0 * pi * cosconst.G_const_Mpc_Msun_s ) / \
        ( cosdist.angular_diameter_distance(self.zd, 0.0, **self.cosmo) * beta )


    # ---
    # Dimensionless projected surface mass density - KappaAtR
    # ---
    def KappaAtR(self, rmpc, zs = 1.0, beta = None):
        """
        This function calculates the projected mass density at the scaled radius rmpc in the unit of Mpc.

        It follows the formular at Umetsu 2010 (astro-ph).
        You can specify the lensing efficiency or zs, yourself.

        Parameters:
            -`rmpc`: float or 1d numpy array, the radius in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.
        Return:
            -`KappaAtR`: float or 1d numpy array, the projected surface mass density at rmpc
        """
        # sanitize
        rmpc            =   np.array(rmpc, ndmin=1)
        # the normalization
        crit_den        =   self.Sigma_crit(zs = zs, beta = beta)
        Norm            =   2.0 * self.rhos * self.rs / crit_den

        # define internal functions
        def factor_smaller_than_one(X):
            X       = np.array(X, ndmin=1)
            insideF = (1.0 - X)/(1.0 + X)
            return 1.0 / (1.0 - X**2) * ( -1.0 + 2.0/np.sqrt(1.0-X**2) * np.arctanh( np.sqrt(insideF) ) )

        def factor_larger_than_one(X):
            X       = np.array(X, ndmin=1)
            insideF = (X - 1.0)/(1.0 + X)
            return 1.0 / (X**2 - 1.0) * (  1.0 - 2.0/np.sqrt(X**2-1.0) * np.arctan(  np.sqrt(insideF) ) )
        # calculate the profile
        profile_smaller_than_one     =      factor_smaller_than_one(X = rmpc/self.rs)
        profile_equal_to_one         =      1.0/3.0 * np.ones(len(rmpc))
        profile_larger_than_one      =      factor_larger_than_one(X = rmpc/self.rs)
        # calculate the final profile
        profile =   np.copy( profile_equal_to_one )
        profile[ (rmpc/self.rs > 1.0) ] =   np.copy( profile_larger_than_one[  (rmpc/self.rs > 1.0) ] )
        profile[ (rmpc/self.rs < 1.0) ] =   np.copy( profile_smaller_than_one[ (rmpc/self.rs < 1.0) ] )

        # return
        return Norm * profile


    # ---
    # Dimensionless average surface mass density - KappaBar
    # ---
    def KappaBar(self, rmpc, zs = 1.0, beta = None):
        """
        This function calculates the mean projected mass density
        inside the radius in the unit of Mpc.

        It follows the formular at Umetsu 2010 (astro-ph).
        You can specify the lensing efficiency or zs, yourself.

        Parameters:
            -`rmpc`: float or 1d numpy array, the radius in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.
        Return:
            -`KappaBar`: float or 1d numpy array, the average projected surface mass density within rmpc.
        """
        # sanitize
        rmpc            =   np.array(rmpc, ndmin=1)
        # the normalization
        crit_den        =   self.Sigma_crit(zs = zs, beta = beta)
        Norm            =   2.0 * self.rhos * self.rs / crit_den

        # define internal functions
        def factor_smaller_than_one(X):
            X       = np.array(X, ndmin=1)
            insideF = (1.0 - X)/(1.0 + X)
            return np.log(X/2.0) + 2.0 / np.sqrt(1.0 - X**2) * np.arctanh( np.sqrt(insideF) )

        def factor_larger_than_one(X):
            X       = np.array(X, ndmin=1)
            insideF = (X - 1.0)/(1.0 + X)
            return np.log(X/2.0) + 2.0 / np.sqrt(X**2 - 1.0) * np.arctan( np.sqrt(insideF) )

        # calculate the profile
        profile_smaller_than_one     =      factor_smaller_than_one(X = rmpc/self.rs)
        profile_equal_to_one         =      np.log( np.array(rmpc/self.rs, ndmin=1) / 2.0 ) + 1.0
        profile_larger_than_one      =      factor_larger_than_one(X = rmpc/self.rs)

        # calculate the final profile
        profile =   np.copy( profile_equal_to_one )
        profile[ (rmpc/self.rs > 1.0) ] =   np.copy( profile_larger_than_one[  (rmpc/self.rs > 1.0) ] )
        profile[ (rmpc/self.rs < 1.0) ] =   np.copy( profile_smaller_than_one[ (rmpc/self.rs < 1.0) ] )

        # return
        return 2.0 * Norm * profile / ( np.array(rmpc, ndmin=1) / self.rs )**2


    # ---
    # Tangential shear - gamma
    # ---
    def TangentialShear(self, rmpc, zs = 1.0, beta = None):
        """
        This function calculates the tangential shear at the radius in the unit of Mpc.

        It follows the formular at Umetsu 2010 (astro-ph).
        You can specify the lensing efficiency or zs, yourself.

        Parameters:
            -`rmpc`: float or 1d numpy array, the radius in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.

        Return:
            -`TangentialShear`: float or 1d numpy array, the tangential shear at rmpc.

        """
        return ( self.KappaBar(rmpc, zs = zs, beta = beta) - self.KappaAtR(rmpc, zs = zs, beta = beta) )

    # ---
    # Reduced shear - g := gamma / (1 - kappa)
    # ---
    def ReducedShear(self, rmpc, zs = 1.0, beta = None):
        """
        This function calculates the reduced shear at the radius in the unit of Mpc.

        It follows the formular at Umetsu 2010 (astro-ph).
        You can specify the lensing efficiency or zs, yourself.

        Parameters:
            -`rmpc`: float or 1d numpy array, the radius in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.

        Return:
            -`ReducedShear`: float or 1d numpy array, the tangential shear at rmpc.

        """
        return self.TangentialShear(rmpc, zs = zs, beta = beta) / (1.0 - self.KappaAtR(rmpc, zs = zs, beta = beta) )


    # ---
    # det of Jacobian - the exact form
    # ---
    def DetJ(self, rmpc, zs = 1.0, beta = None):
        """
        This function calculates the det(Jacobian) at the radius in the unit of Mpc.

        It returns det(J) = (1.0 - Kappa)**2 - tangential**2
        You can specify the lensing efficiency or zs, yourself.

        Parameters:
            -`rmpc`: float or 1d numpy array, the radius in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.

        Return:
            -`DetJ`: float or 1d numpy array, the determinant of Joacobian at rmpc.

        """
        return ( 1.0 - self.KappaAtR(rmpc, zs = zs, beta = beta) )**2 - self.TangentialShear(rmpc, zs = zs, beta = beta)**2

    # ---
    # det of Jacobian - the weak lensing regime
    # ---
    def DetJ_weak(self, rmpc, zs = 1.0, beta = None):
        """
        This function calculates the det(Jacobian) at the radius in the unit of Mpc in the weak lensing regime.

        It returns det(J) = 1.0 - 2.0 * Kappa
        You can specify the lensing efficiency or zs, yourself.

        Parameters:
            -`rmpc`: float or 1d numpy array, the radius in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.

        Return:
            -`DetJ_weak`: float or 1d numpy array, the determinant of Joacobian at rmpc (weak lensing approach).

        """
        return 1.0 - 2.0 * self.KappaAtR(rmpc, zs = zs, beta = beta)


    # ---
    # magnification factor - mu - which is 1 / detJ
    # ---
    def mu(self, rmpc, zs = 1.0, beta = None):
        """
        This function calculates the magnification := 1.0 /det(Jacobian)
        at the radius in the unit of Mpc.

        It returns 1.0 / det(J) = 1.0/ ( (1.0 - Kappa)**2 - tangential**2 )
        You can specify the lensing efficiency or zs, yourself.

        Parameters:
            -`rmpc`: float or 1d numpy array, the radius in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.

        Return:
            -`mu`: float or 1d numpy array, the magnification factor at rmpc.

        """
        return 1.0 / self.DetJ(rmpc, zs = zs, beta = beta)


    # ---
    # magnification factor in the weak lensing regime - mu_weak
    # ---
    def mu_weak(self, rmpc, zs = 1.0, beta = None):
        """
        This function calculates the magnification factor at the radius in the unit of Mpc in the weak lensing regime.

        It returns 1.0 + 2 * Kappa
        You can specify the lensing efficiency or zs, yourself.

        Parameters:
            -`rmpc`: float or 1d numpy array, the radius in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.

        Return:
            -`mu_weak`: float or 1d numpy array, the magnification factor (weak lensing approximated) at rmpc.

        """
        return 1.0 + 2.0 * self.KappaAtR(rmpc, zs = zs, beta = beta)


    # ---
    # Miscentered Dimensionless projected surface mass density - _misKappaAtR
    # ---
    def _misKappaAtR(self, rmpc, rmpc_offset = 0.3, zs = 1.0, beta = None, ntheta = 2**10 + 2):
        """
        This function calculates the projected mass density at the scaled radius rmpc in the unit of Mpc. This is the Kappa at that radius __before__ convolving the Poffset.

        It follows the formular in Jes Ford paper.

        Parameters:
            -`rmpc`: float or 1d numpy array, the radius in the unit of Mpc.
            -`rmpc_offset`: float or 1d array, the radius offset in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.
            -`ntheta`: int. It is the number of bins for numerical integration.
        Return:
            -`KappaAtR`: float or 1d numpy array, the projected surface mass density at rmpc
        """
        # sanitize
        rmpc        =   np.array(rmpc, ndmin=1)
        rmpc_offset =   np.array(rmpc_offset, ndmin=1)
        ntheta      =   int(ntheta)
        nrmpc       =   len(rmpc)
        nrmpc_offset=   len(rmpc_offset)
        # theta
        theta_edges =   np.linspace(0.0, 2.0 * pi, ntheta)
        theta_bins  =   0.5 * ( theta_edges[1:] + theta_edges[:-1] )
        theta_steps =   ( theta_edges[1:] - theta_edges[:-1] )
        # meshgrid in the shape of (nrmpc, nrmpc_offset, ntheta - 1)
        mesh_rmpc, mesh_rmpc_offset, mesh_theta_bins = np.meshgrid(rmpc, rmpc_offset, theta_bins , indexing = "ij")
        _        ,                _, mesh_theta_steps= np.meshgrid(rmpc, rmpc_offset, theta_steps, indexing = "ij")
        # define effect radius in the shape of ( nrmpc, nrmpc_offset, ntheta - 1 )
        rmpc_eff    =   np.sqrt( mesh_rmpc**2 + mesh_rmpc_offset**2 - 2.0 * mesh_rmpc * mesh_rmpc_offset * np.cos(mesh_theta_bins) )
        # calculate misKappa(R,Roffset,theta)
        misKappa_R_Roffset_theta    =   self.KappaAtR(rmpc = rmpc_eff.flatten(), zs = zs, beta = beta).reshape( (nrmpc, nrmpc_offset, -1) )
        # calculate misKappa(R,Roffset)
        misKappaAtR                 =   np.sum( misKappa_R_Roffset_theta * mesh_theta_steps, axis = -1) / (2.0 * pi)
        #misKappaAtR                 =   integrate.romb( misKappa_R_Roffset_theta * mesh_theta_steps, axis = -1) / (2.0 * pi)
        # return - in the shape of ( nrmpc, nrmpc_offset)
        return misKappaAtR



    # ---
    # Miscentered Dimensionless projected surface mass density - misKappaAtR
    # ---
    def misKappaAtR(self, rmpc, zs = 1.0, beta = None, rmpc_offset_lo = 1E-4, rmpc_offset_hi = 1E2, nrmpc_offset = 2**7 + 2, ntheta = 2**7 + 2, mis_type = "zitrin+12"):
        """
        This function calculates the projected mass density at the scaled radius rmpc in the unit of Mpc.

        It follows the formular in Jes Ford paper.

        Parameters:
            -`rmpc`: float or 1d numpy array, the radius in the unit of Mpc.
            -`rmpc_offset_lo`: float, the lower bound of integrating radius offset in the unit of Mpc.
            -`rmpc_offset_hi`: float, the higher bound of integrating radius offset in the unit of Mpc.
            -`nrmpc_offset`: int, the number of binning for integrating radius offset in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.
            -`ntheta`: int. It is the number of bins for numerical integration.
            -`mis_type`: string. It is the miscentering function given by different literature. see Poffset(rmpc, mis_type).

        Return:
            -`KappaAtR`: float or 1d numpy array, the projected surface mass density at rmpc
        """
        # sanitize
        rmpc              =   np.array(rmpc, ndmin=1)
        rmpc_offset_edges =   10.0**np.linspace( log10(rmpc_offset_lo), log10(rmpc_offset_hi), nrmpc_offset )
        rmpc_offset_bins  =   0.5 * ( rmpc_offset_edges[1:] + rmpc_offset_edges[:-1] )
        rmpc_offset_steps =   ( rmpc_offset_edges[1:] - rmpc_offset_edges[:-1] )
        ntheta            =   int(ntheta)
        nrmpc             =   len(rmpc)
        nrmpc_offset      =   len(rmpc_offset_bins)
        # Poffset
        Poff              =   Poffset(rmpc_offset = rmpc_offset_bins, mis_type = mis_type)
        # _misKappaAtR
        misKappa_at_R_Roffset   =   self._misKappaAtR(rmpc = rmpc, rmpc_offset = rmpc_offset_bins, zs = zs, beta = beta, ntheta = ntheta)
        # integrating along rmpc_offset
        misKappa_at_R           =   np.sum( misKappa_at_R_Roffset * Poff * rmpc_offset_steps, axis = 1)
        #misKappa_at_R           =   integrate.romb( misKappa_at_R_Roffset * Poff * rmpc_offset_steps, axis = 1)
        # return - in the shape of ( nrmpc, nrmpc_offset)
        return misKappa_at_R



    # ---
    # Miscentered Dimensionless projected surface mass density - misKappaAtR
    # ---
    def misKappaBar(self, rmpc, zs = 1.0, beta = None, nrmpc = 2**8 + 2, rmpc_offset_lo = 1E-4, rmpc_offset_hi = 1E2, nrmpc_offset = 2**7 + 2, ntheta = 2**7 + 2, mis_type = "zitrin+12"):
        """
        This function calculates the average projected mass density within the scaled radius rmpc in the unit of Mpc.

        It follows the formular in Jes Ford paper.

        Parameters:
            -`rmpc`: float or 1d numpy array, the radius in the unit of Mpc.
            -`nrmpc`: int. The number of the bins used for integration.
            -`rmpc_offset_lo`: float, the lower bound of integrating radius offset in the unit of Mpc.
            -`rmpc_offset_hi`: float, the higher bound of integrating radius offset in the unit of Mpc.
            -`nrmpc_offset`: int, the number of binning for integrating radius offset in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.
            -`ntheta`: int. It is the number of bins for numerical integration.
            -`mis_type`: string. It is the miscentering function given by different literature. see Poffset(rmpc, mis_type).

        Return:
            -`KappaAtR`: float or 1d numpy array, the projected surface mass density at rmpc
        """
        # sanitize
        rmpc              =   np.array(rmpc, ndmin=1)
        nrmpc             =   int(nrmpc)
        # rmpc binning for integration - this is hardcoded, and the result should be converged.
        xx_edges        =   10.0**np.linspace( -6.0, 2.0, 2**10 + 2 )
        xx_bins         =   0.5 * ( xx_edges[1:] + xx_edges[:-1] )
        xx_steps        =   ( xx_edges[1:] - xx_edges[:-1] )
        # derive the misKappaAtR - in the length of 2**int + 1
        misKappa_at_R     =   self.misKappaAtR(rmpc = xx_bins, zs = zs, beta = beta, rmpc_offset_lo = rmpc_offset_lo, rmpc_offset_hi = rmpc_offset_hi, nrmpc_offset = nrmpc_offset, ntheta = ntheta, mis_type = mis_type)
        # interpolate - using slinear
        interpolate_f     =  interpolate.interp1d(xx_bins, misKappa_at_R, kind = "slinear")
        ## define func
        #def integrate_me(x):
        #    return self.misKappaAtR(rmpc = x, zs = zs, beta = beta, rmpc_offset_lo = rmpc_offset_lo, rmpc_offset_hi = rmpc_offset_hi, nrmpc_offset = nrmpc_offset, ntheta = ntheta, mis_type = mis_type) * 2.0 * pi * x


        # define a function to do romb integration
        def integrate_me2(x):
            # hardcode the inner boundary as 10**-5 Mpc - a sub galaxy scale should be more than enough...
            new_xx_edges    =   10.0**np.linspace(-5.0, log10(x), nrmpc)
            new_xx_bins     =   0.5 * ( new_xx_edges[1:] + new_xx_edges[:-1] )
            new_xx_steps    =   ( new_xx_edges[1:] - new_xx_edges[:-1] )
            return integrate.romb( ( interpolate_f(new_xx_bins) * 2 * pi * new_xx_bins * new_xx_steps) )

        # calculate miscentered kappa bar - ok, I loop them, it can be faster if I dont.
        misKappa_bar    =   np.array([ integrate_me2(x) for x in rmpc ]) / (rmpc**2 * pi)
        # return
        return misKappa_bar



    # ---
    # Tangential shear - gamma
    # ---
    def misTangentialShear(self, rmpc, zs = 1.0, beta = None, nrmpc = 2**8 + 2, rmpc_offset_lo = 1E-4, rmpc_offset_hi = 1E2, nrmpc_offset = 2**7 + 2, ntheta = 2**7 + 2, mis_type = "zitrin+12"):
        """
        This function calculates the tangential shear at the radius in the unit of Mpc - including the miscentering.

        It follows the formular at Umetsu 2010 (astro-ph) and Ford+14.
        You can specify the lensing efficiency or zs, yourself.

        Parameters:
            -`rmpc`: float or 1d numpy array, the radius in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.
            -`nrmpc`: int. The number of the bins used for integration.
            -`rmpc_offset_lo`: float, the lower bound of integrating radius offset in the unit of Mpc.
            -`rmpc_offset_hi`: float, the higher bound of integrating radius offset in the unit of Mpc.
            -`nrmpc_offset`: int, the number of binning for integrating radius offset in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.
            -`ntheta`: int. It is the number of bins for numerical integration.
            -`mis_type`: string. It is the miscentering function given by different literature. see Poffset(rmpc, mis_type).


        Return:
            -`TangentialShear`: float or 1d numpy array, the tangential shear at rmpc.

        """
        return ( self.misKappaBar(rmpc, zs = zs, beta = beta, nrmpc = nrmpc, rmpc_offset_lo = rmpc_offset_lo, rmpc_offset_hi = rmpc_offset_hi, nrmpc_offset = nrmpc_offset, ntheta = ntheta, mis_type = mis_type) - \
                self.misKappaAtR(rmpc, zs = zs, beta = beta, rmpc_offset_lo = rmpc_offset_lo, rmpc_offset_hi = rmpc_offset_hi, nrmpc_offset = nrmpc_offset, ntheta = ntheta, mis_type = mis_type) )

    # ---
    # Reduced shear - g := gamma / (1 - kappa)
    # ---
    def misReducedShear(self, rmpc, zs = 1.0, beta = None, nrmpc = 2**8 + 2, rmpc_offset_lo = 1E-4, rmpc_offset_hi = 1E2, nrmpc_offset = 2**7 + 2, ntheta = 2**7 + 2, mis_type = "zitrin+12"):
        """
        This function calculates the reduced shear at the radius in the unit of Mpc - including miscentering.

        It follows the formular at Umetsu 2010 (astro-ph) and Ford+14.
        You can specify the lensing efficiency or zs, yourself.

        Parameters:
            -`rmpc`: float or 1d numpy array, the radius in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.
            -`nrmpc`: int. The number of the bins used for integration.
            -`rmpc_offset_lo`: float, the lower bound of integrating radius offset in the unit of Mpc.
            -`rmpc_offset_hi`: float, the higher bound of integrating radius offset in the unit of Mpc.
            -`nrmpc_offset`: int, the number of binning for integrating radius offset in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.
            -`ntheta`: int. It is the number of bins for numerical integration.
            -`mis_type`: string. It is the miscentering function given by different literature. see Poffset(rmpc, mis_type).

        Return:
            -`ReducedShear`: float or 1d numpy array, the tangential shear at rmpc.

        """
        return self.misTangentialShear(rmpc, zs = zs, beta = beta, nrmpc = nrmpc, rmpc_offset_lo = rmpc_offset_lo, rmpc_offset_hi = rmpc_offset_hi, nrmpc_offset = nrmpc_offset, ntheta = ntheta, mis_type = mis_type) / \
                (1.0 - self.misKappaAtR(rmpc, zs = zs, beta = beta, rmpc_offset_lo = rmpc_offset_lo, rmpc_offset_hi = rmpc_offset_hi, nrmpc_offset = nrmpc_offset, ntheta = ntheta, mis_type = mis_type) )


    # ---
    # det of Jacobian - the exact form
    # ---
    def misDetJ(self, rmpc, zs = 1.0, beta = None, nrmpc = 2**8 + 2, rmpc_offset_lo = 1E-4, rmpc_offset_hi = 1E2, nrmpc_offset = 2**7 + 2, ntheta = 2**7 + 2, mis_type = "zitrin+12"):
        """
        This function calculates the det(Jacobian) at the radius in the unit of Mpc - including miscentering.

        It returns det(J) = (1.0 - Kappa)**2 - tangential**2
        You can specify the lensing efficiency or zs, yourself.

        Parameters:
            -`rmpc`: float or 1d numpy array, the radius in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.
            -`nrmpc`: int. The number of the bins used for integration.
            -`rmpc_offset_lo`: float, the lower bound of integrating radius offset in the unit of Mpc.
            -`rmpc_offset_hi`: float, the higher bound of integrating radius offset in the unit of Mpc.
            -`nrmpc_offset`: int, the number of binning for integrating radius offset in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.
            -`ntheta`: int. It is the number of bins for numerical integration.
            -`mis_type`: string. It is the miscentering function given by different literature. see Poffset(rmpc, mis_type).

        Return:
            -`DetJ`: float or 1d numpy array, the determinant of Joacobian at rmpc.

        """
        return ( 1.0 - self.misKappaAtR(rmpc, zs = zs, beta = beta, rmpc_offset_lo = rmpc_offset_lo, rmpc_offset_hi = rmpc_offset_hi, nrmpc_offset = nrmpc_offset, ntheta = ntheta, mis_type = mis_type) )**2 - \
                self.misTangentialShear(rmpc, zs = zs, beta = beta, nrmpc = nrmpc, rmpc_offset_lo = rmpc_offset_lo, rmpc_offset_hi = rmpc_offset_hi, nrmpc_offset = nrmpc_offset, ntheta = ntheta, mis_type = mis_type)**2

    # ---
    # det of Jacobian - the weak lensing regime
    # ---
    def misDetJ_weak(self, rmpc, zs = 1.0, beta = None, nrmpc = 2**8 + 2, rmpc_offset_lo = 1E-4, rmpc_offset_hi = 1E2, nrmpc_offset = 2**7 + 2, ntheta = 2**7 + 2, mis_type = "zitrin+12"):
        """
        This function calculates the det(Jacobian) at the radius in the unit of Mpc in the weak lensing regime - including miscentering.

        It returns det(J) = 1.0 - 2.0 * Kappa
        You can specify the lensing efficiency or zs, yourself.

        Parameters:
            -`rmpc`: float or 1d numpy array, the radius in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.
            -`nrmpc`: int. The number of the bins used for integration.
            -`rmpc_offset_lo`: float, the lower bound of integrating radius offset in the unit of Mpc.
            -`rmpc_offset_hi`: float, the higher bound of integrating radius offset in the unit of Mpc.
            -`nrmpc_offset`: int, the number of binning for integrating radius offset in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.
            -`ntheta`: int. It is the number of bins for numerical integration.
            -`mis_type`: string. It is the miscentering function given by different literature. see Poffset(rmpc, mis_type).

        Return:
            -`DetJ_weak`: float or 1d numpy array, the determinant of Joacobian at rmpc (weak lensing approach).

        """
        return 1.0 - 2.0 * self.misKappaAtR(rmpc, zs = zs, beta = beta, rmpc_offset_lo = rmpc_offset_lo, rmpc_offset_hi = rmpc_offset_hi, nrmpc_offset = nrmpc_offset, ntheta = ntheta, mis_type = mis_type)


    # ---
    # magnification factor - mu - which is 1 / detJ
    # ---
    def mismu(self, rmpc, zs = 1.0, beta = None, nrmpc = 2**8 + 2, rmpc_offset_lo = 1E-4, rmpc_offset_hi = 1E2, nrmpc_offset = 2**7 + 2, ntheta = 2**7 + 2, mis_type = "zitrin+12"):
        """
        This function calculates the magnification := 1.0 /det(Jacobian)
        at the radius in the unit of Mpc - including miscentering.

        It returns 1.0 / det(J) = 1.0/ ( (1.0 - Kappa)**2 - tangential**2 )
        You can specify the lensing efficiency or zs, yourself.

        Parameters:
            -`rmpc`: float or 1d numpy array, the radius in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.
            -`nrmpc`: int. The number of the bins used for integration.
            -`rmpc_offset_lo`: float, the lower bound of integrating radius offset in the unit of Mpc.
            -`rmpc_offset_hi`: float, the higher bound of integrating radius offset in the unit of Mpc.
            -`nrmpc_offset`: int, the number of binning for integrating radius offset in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.
            -`ntheta`: int. It is the number of bins for numerical integration.
            -`mis_type`: string. It is the miscentering function given by different literature. see Poffset(rmpc, mis_type).

        Return:
            -`mu`: float or 1d numpy array, the magnification factor at rmpc.

        """
        return 1.0 / self.misDetJ(rmpc, zs = zs, beta = beta, nrmpc = nrmpc, rmpc_offset_lo = rmpc_offset_lo, rmpc_offset_hi = rmpc_offset_hi, nrmpc_offset = nrmpc_offset, ntheta = ntheta, mis_type = mis_type)


    # ---
    # magnification factor in the weak lensing regime - mu_weak
    # ---
    def mismu_weak(self, rmpc, zs = 1.0, beta = None, nrmpc = 2**8 + 2, rmpc_offset_lo = 1E-4, rmpc_offset_hi = 1E2, nrmpc_offset = 2**7 + 2, ntheta = 2**7 + 2, mis_type = "zitrin+12"):
        """
        This function calculates the magnification factor at the radius in the unit of Mpc in the weak lensing regime - including miscentering.

        It returns 1.0 + 2 * Kappa
        You can specify the lensing efficiency or zs, yourself.

        Parameters:
            -`rmpc`: float or 1d numpy array, the radius in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.
            -`nrmpc`: int. The number of the bins used for integration.
            -`rmpc_offset_lo`: float, the lower bound of integrating radius offset in the unit of Mpc.
            -`rmpc_offset_hi`: float, the higher bound of integrating radius offset in the unit of Mpc.
            -`nrmpc_offset`: int, the number of binning for integrating radius offset in the unit of Mpc.
            -`zs`: float, the redshift of the source.
            -`beta`: float, the lensing efficiency.
                     If it is None, then calculate the beta from zd and zs.
            -`ntheta`: int. It is the number of bins for numerical integration.
            -`mis_type`: string. It is the miscentering function given by different literature. see Poffset(rmpc, mis_type).

        Return:
            -`mu_weak`: float or 1d numpy array, the magnification factor (weak lensing approximated) at rmpc.

        """
        return 1.0 + 2.0 * self.misKappaAtR(rmpc, zs = zs, beta = beta, rmpc_offset_lo = rmpc_offset_lo, rmpc_offset_hi = rmpc_offset_hi, nrmpc_offset = nrmpc_offset, ntheta = ntheta, mis_type = mis_type)








# ---
# testing
# ---
if    __name__ == "__main__":

    A = Halo(mass = 5E14)
    rmpc            =   10.0**np.linspace(-1.5,1.0,100)
    #rmpc_offset     =   10.0**np.linspace(-3.0,2.0,100)
    #rmpc_offset_bins =   0.5*(rmpc_offset[1:] + rmpc_offset[:-1])
    #rmpc_offset_steps=   (rmpc_offset[1:] - rmpc_offset[:-1])
    centered_kappa  =   A.KappaAtR(rmpc)
    mis_kappa       =   A.misKappaAtR(rmpc)
    import matplotlib.pyplot as pyplt
    pyplt.plot(rmpc, centered_kappa, "k-", rmpc, mis_kappa, "r-")
    pyplt.xscale("log")
    pyplt.ylabel("$\kappa(r)$")
    pyplt.xlabel("$r / \mathrm{Mpc}$")
    pyplt.show()

    pyplt.plot(rmpc, A.TangentialShear(rmpc), "k-", rmpc, A.misTangentialShear(rmpc), "r-")
    pyplt.xscale("log")
    pyplt.ylabel("$\gamma(r)$")
    pyplt.xlabel("$r / \mathrm{Mpc}$")
    pyplt.show()

    pyplt.plot(rmpc, A.mu(rmpc), "k-", rmpc, A.mismu(rmpc), "r-")
    pyplt.xscale("log")
    pyplt.ylabel("$\mu(r)$")
    pyplt.xlabel("$r / \mathrm{Mpc}$")
    pyplt.show()
