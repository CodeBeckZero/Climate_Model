#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case 2: Linear decrease of CO2 from current levels to carbon neutral by 2100

Created on Tue Nov  1 21:31:51 2022
@author: michaelbeck
"""
import numpy as np
import ClimBoxFunc
from scipy import integrate
import matplotlib.pyplot as plt
from ClimBoxModel import box_model, co2_data, init_conds, t, ppm_slope, ppm_yint    
import scipy.interpolate as interp

# Forcasting

## Initial Variables

t_forecast = np.arange(2021,2101,1)         # Forecast Range
t_new = np.concatenate([t,t_forecast])      # Combining Range (1750-2100)

# Setting CO2 Land Use linear decrease from current levels
c2_co2_lu = np.linspace(co2_data[3][-1],0,80)              
## Combining data with original CO2 LU data                      
c2_lu_data = np.concatenate((co2_data[3], c2_co2_lu))    

# Setting CO2 FF linear decrease from current levels
c2_co2_ff = np.linspace(co2_data[1][-1],0,80)            
## Combining data with original CO2 FF data             
c2_ff_data = np.concatenate((co2_data[1],c2_co2_ff))   

# Totalling C02 Emissions (FF + LU) (1750-2100)
c2_tot_data = c2_lu_data + c2_ff_data

# Creating time functions for FF, LU, TOT for ODE Model
c2_co2_ff = interp.interp1d(t_new,c2_ff_data, kind='cubic', fill_value='extrapolate')
c2_co2_lu = interp.interp1d(t_new,c2_lu_data, kind='cubic', fill_value='extrapolate')
c2_co2_tot = interp.interp1d(t_new,c2_tot_data, kind='cubic', fill_value='extrapolate')

# Solving Model
c2_sol = integrate.odeint(box_model, init_conds, t_new, tfirst = True, 
                        args=(c2_co2_lu, c2_co2_ff,init_conds,0.42))
 
c2n1, c2n2, c2n3, c2n4, c2n5, c2n6, c2n7 = c2_sol.T ## Unpacking Solutions

# Calculations for Plots
c2_ff_gt = ClimBoxFunc.PgC_to_Gt(c2_co2_ff(t_new)) ## Fossil Fuels from PgC to GT
c2_lu_gt = ClimBoxFunc.PgC_to_Gt(c2_co2_lu(t_new)) ## Land Use from PgC to GT
cl_tot_gt = c2_ff_gt + c2_lu_gt ## Computing Total CO2
c2_ppm = ClimBoxFunc.PgC_to_ppm(c2n1) ## Atmosphere PgC to CO2 ppm
ppm_max = max(c2_ppm) ## Find Max PPM   
tpos = np.where(c2_ppm == ppm_max) ## Find Position to find year
tmax =  t_new[tpos][0] ## Year @ PPM max
atemp_max = ppm_slope * ppm_max + ppm_yint ## Calculate Anomalous Temperature from line of best fit


# Ploting Case 2
fig2, (ax8,ax9) = plt.subplots(2,1,constrained_layout=True)

## Model Input (FF,LU)
ax8.set_title(f'Model Input: $CO_{2}$ Input')
ax8.plot(t_new,c2_ff_gt , label="Fossil Fuel", color='b')
ax8.plot(t_new,c2_lu_gt , label="Land use", color='g')
ax8.plot(t_new,cl_tot_gt, label="Total", color='r')
ax8.set_ylabel(f'$CO_{2}$ Emissions (Gt per year)')
ax8.set_xticklabels("") 
ax8.axvspan(2021,2100, color ='#a2d9ce', alpha=0.3)
ax8.grid()
ax8.legend()

## Model Output PPM
ax9.set_title(f'Model Output: $CO_{2}$ Output')
ax9.plot(t_new,c2_ppm, label="Model", color='#0C090A')
ax9.set_ylabel(f'$CO_{2}$ Concentration (ppm)')
ax9.axvspan(2021,2100, color ='#a2d9ce', alpha=0.3)
ax9.annotate(f'Max:[{tmax},{ppm_max:.1f} ppm]\nAnomaly Temp: {atemp_max:.1f}\u2103',
             xy=(tmax, ppm_max), xytext=(tmax-200, ppm_max-35),
            arrowprops=dict(color='red',arrowstyle="->"),)
ax9.grid()
ax9.legend()
ax9.set_xlabel("Year")


fig2.savefig("case2.png")
