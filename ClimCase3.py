#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case 3: FF Emissions grow linearly until 2050, decline to 0 by 2100.
        FU emissions decline linearly to 0 by 2100

Created on Tue Nov  1 21:32:13 2022
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

# Setting CO2 Land Use to decline to 0 by 2100 from current levels
c3_co2_lu = np.linspace(co2_data[3][-1],0,80)      
## Combining data with original CO2 LU data                
c3_lu_data = np.concatenate((co2_data[3], c3_co2_lu))    

# Setting up CO2 Fossil Fuel Use 
## Linear Interpolation from 1970-2020, to predict Maximum in 2050
a, b = np.polyfit(co2_data[0][-50:],co2_data[1][-50:],1) ### Creating Linear Equation
max_2050= a*2050+b ### Predicting Value
### Setting up time domain for cubic interpolation 
t_ff = [1970,2020,2050,2100]
### Setting up FF for cubic interpolation 
c3_ff = [co2_data[1][-50],co2_data[1][-1], max_2050, 0]  
### FF cubic interpolation between 1970-2100         
c3_func_ff = interp.interp1d(t_ff ,c3_ff, kind='cubic', fill_value='extrapolate')             
### Combining original FF data with Cubic Interpolation on year-by-year scale       
c3_ff_data = np.concatenate((co2_data[1], c3_func_ff(t_forecast)))

# Totalling C02 Emissions (FF + LU) (1750-2100)
c3_tot_data = c3_lu_data + c3_ff_data

# Creating time functions for FF, LU, TOT for ODE Model
c3_co2_ff = interp.interp1d(t_new,c3_ff_data, kind='cubic', fill_value='extrapolate')
c3_co2_lu = interp.interp1d(t_new,c3_lu_data, kind='cubic', fill_value='extrapolate')
c3_co2_tot = interp.interp1d(t_new,c3_tot_data, kind='cubic', fill_value='extrapolate')

# Solving Model
c3_sol = integrate.odeint(box_model, init_conds, t_new, tfirst = True, 
                        args=(c3_co2_lu, c3_co2_ff,init_conds,0.42))
 
c3n1, c3n2, c3n3, c3n4, c3n5, c3n6, c3n7 = c3_sol.T #### Unpacking Solutions

# Calculations for Plots
c3_ff_gt = ClimBoxFunc.PgC_to_Gt(c3_co2_ff(t_new)) #### Fossil Fuels from PgC to GT
c3_lu_gt = ClimBoxFunc.PgC_to_Gt(c3_co2_lu(t_new)) #### Land Use from PgC to GT
c3_tot_gt = c3_ff_gt + c3_lu_gt #### Computing Total CO2
c3_ppm = ClimBoxFunc.PgC_to_ppm(c3n1) #### Atmosphere PgC to CO2 ppm
ppm_max = max(c3_ppm) ## Find Max PPM   
tpos = np.where(c3_ppm == ppm_max) ## Find Position to find year
tmax =  t_new[tpos][0] ## Year @ PPM max
atemp_max = ppm_slope * ppm_max + ppm_yint ## Calculate Anomalous Temperature from line of best fit

# Ploting Case 3
fig3, (ax10,ax11) = plt.subplots(2,1,constrained_layout=True)

#### Model Input (FF,LU)
ax10.set_title(f'Model Input: $CO_{2}$ Input')
ax10.plot(t_new,c3_ff_gt , label="Fossil Fuel", color='b')
ax10.plot(t_new,c3_lu_gt , label="Land use", color='g')
ax10.plot(t_new,c3_tot_gt, label="Total", color='r')
ax10.set_ylabel(f'$CO_{2}$ Emissions (Gt per year)')
ax10.set_xticklabels("") 
ax10.axvspan(2021,2100, color ='#a2d9ce', alpha=0.3)
ax10.grid()
ax10.legend()

#### Model Output PPM
ax11.set_title(f'Model Output: $CO_{2}$ Output')
ax11.plot(t_new,c3_ppm, label="Model", color='#0C090A')
ax11.set_ylabel(f'$CO_{2}$ Concentration (ppm)')
ax11.axvspan(2021,2100, color ='#a2d9ce', alpha=0.3)
ax11.annotate(f'Max:[{tmax},{ppm_max:.1f} ppm]\nAnomaly Temp: {atemp_max:.1f}\u2103',
             xy=(tmax, ppm_max), xytext=(tmax-200, ppm_max-35),
            arrowprops=dict(color='red',arrowstyle="->"),)
ax11.grid()
ax11.legend()
ax11.set_xlabel("Year")

fig3.savefig("case3.png")
