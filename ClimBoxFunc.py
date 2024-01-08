#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Climate Box Model Functions
Created on Sun Oct 23 10:52:14 2022
@author: michaelbeck
"""


def Gt_to_PgC(co2_gt_value):
    return co2_gt_value / 3.66

def PgC_to_ppm(pgc_value):
    """
    Converts petagrams of carbon into CO2 parts per million (ppm)

    Parameters
    ----------
    pgc_value : Numeric
        Peta-grams of carbon

    Returns
    -------
    Float
        Carbon Dioxide in parts per million (ppm)

    """
    return pgc_value / 2.13

def PgC_to_Gt(pgc_value):
    """
    Converts peta grams of carbon into gigatones of CO2

    Parameters
    ----------
    pgc_value : Numeric
        peta-grams of carbon

    Returns
    -------
    Float
        Carbon Dioxide in giga tonnes
    """
    return pgc_value * 3.66

def PgC_buffer_factor(n1_pgc_value):
    """
    

    Parameters
    ----------
    pgc_value : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    n1_ppm  = PgC_to_ppm(n1_pgc_value)
    
    return 3.69 + 1.86e-2*n1_ppm - 1.8e-6*n1_ppm**2

def n6_flux(pgc_now,ppm_strt,flux_strt,f_fert):
    """
    Calculates current snapshot's biosphere flux based on previous observations

    Parameters
    ----------
    flux_strt : Numerical
        Pre-industrial Flux in PgC/year
    ppm_strt : Numerical
        Pre-industrial C02 ppm 
    pgc_now : Numerical 
        CO2 in Peta-grams of carbon for current snapshot
    f_fert : Float
        Free parameter used to match present observed atmosphere (approx 0.3-0.48)

    Returns
    -------
    Numeric
        Current snapshot's biosphere C02 flux (XXX units)

    """
    from numpy import log

    # Conver Pgrams of Carbon to ppm
    ppm_now = PgC_to_ppm(pgc_now)
    
    return flux_strt * (1 + f_fert * (log(ppm_now/ppm_strt)))

def radiative_forcing(pgc_now,ppm_strt):
    from numpy import log

    ppm_now = PgC_to_ppm(pgc_now)
    return 5.35 * log(ppm_now/ppm_strt)

def temp_anomaly(pgc_now,ppm_strt):
    k_ClimSen = 0.27
    return k_ClimSen * radiative_forcing(pgc_now,ppm_strt)



    