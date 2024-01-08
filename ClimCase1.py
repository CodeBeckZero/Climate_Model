
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case 1: C02 Emissions Stay at current levels until 2100
Created on Tue Nov  1 21:30:58 2022
@author: michaelbeck
"""
import numpy as np
import ClimBoxFunc
from scipy import integrate
import matplotlib.pyplot as plt
from ClimBoxModel import box_model, co2_data, init_conds, t, ppm_slope, ppm_yint  
import scipy.interpolate as interp


# Initial Variables

t_forecast = np.arange(2021,2101,1)         # Forecast Range
t_new = np.concatenate([t,t_forecast])      # Combining Range (1750-2100)

# Setting CO2 Land Use Constant, based 2020 Value
c1_co2_lu = np.zeros(len(t_forecast))               ## Initializing
c1_co2_lu.fill(co2_data[3][-1])                     ## Filling with last Value         
c1_lu_data = np.concatenate((co2_data[3], c1_co2_lu))    ## Combining Data 

# Setting CO2 Fossil Fuel Use Constant, based 2020 Value
c1_co2_ff = np.empty(len(t_forecast))               ## Initializing
c1_co2_ff.fill(co2_data[1][-1])                     ## Filling with last Value 
c1_ff_data = np.concatenate((co2_data[1],c1_co2_ff))    ## Combining Data 

# Totalling C02 Emissions (FF + LU) (1750-2100)
c1_tot_data = c1_lu_data + c1_ff_data

# Creating time functions for FF, LU, TOT for ODE Model
c1_co2_ff = interp.interp1d(t_new,c1_ff_data, kind='cubic', fill_value='extrapolate')
c1_co2_lu = interp.interp1d(t_new,c1_lu_data, kind='cubic', fill_value='extrapolate')
c1_co2_tot = interp.interp1d(t_new,c1_tot_data, kind='cubic', fill_value='extrapolate')

# Solving Model
c1_sol = integrate.odeint(box_model, init_conds, t_new, tfirst = True, 
                        args=(c1_co2_lu, c1_co2_ff,init_conds,0.42))
 
c1n1, c1n2, c1n3, c1n4, c1n5, c1n6, c1n7 = c1_sol.T ## Unpacking Solutions


# Calculations for Plots
c1_ff_gt = ClimBoxFunc.PgC_to_Gt(c1_co2_ff(t_new)) ## Fossil Fuels from PgC to GT
c1_lu_gt = ClimBoxFunc.PgC_to_Gt(c1_co2_lu(t_new)) ## Land Use from PgC to GT
cl_tot_gt = c1_ff_gt + c1_lu_gt ## Computing Total CO2
c1_ppm = ClimBoxFunc.PgC_to_ppm(c1n1) ## Atmosphere PgC to CO2 ppm
ppm_max = max(c1_ppm) ## Find Max PPM   
tpos = np.where(c1_ppm == ppm_max) ## Find Position to find year
tmax =  t_new[tpos][0] ## Year @ PPM max
atemp_max = ppm_slope * ppm_max + ppm_yint ## Calculate Anomalous Temperature from line of best fit


# Ploting Case 1
fig1, (ax6,ax7) = plt.subplots(2,1,constrained_layout=True)

## Model Input (FF,LU)
ax6.set_title(f'Model Input: $CO_{2}$ Input')
ax6.plot(t_new,c1_ff_gt , label="Fossil Fuel", color='b')
ax6.plot(t_new,c1_lu_gt , label="Land use", color='g')
ax6.plot(t_new,cl_tot_gt, label="Total", color='r')
ax6.set_ylabel(f'$CO_{2}$ Emissions (Gt per year)')
ax6.set_xticklabels("") 
ax6.axvspan(2021,2100, color ='#a2d9ce', alpha=0.3)
ax6.grid()
ax6.legend()

## Model Output PPM
ax7.set_title(f'Model Output: $CO_{2}$ Output')
ax7.plot(t_new,c1_ppm, label="Model", color='#0C090A')
ax7.set_ylabel(f'$CO_{2}$ Concentration (ppm)')
ax7.axvspan(2021,2100, color ='#a2d9ce', alpha=0.3)
ax7.annotate(f'Max:[{tmax},{ppm_max:.1f} ppm]\nAnomaly Temp: {atemp_max:.1f}\u2103',
             xy=(tmax, ppm_max), xytext=(tmax-200, ppm_max-35),
            arrowprops=dict(color='red',arrowstyle="->"),)

ax7.grid()
ax7.legend()
ax7.set_xlabel("Year")

fig1.savefig("case1.png")
