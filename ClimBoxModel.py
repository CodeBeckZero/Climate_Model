#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 10:32:30 2022
@author: michaelbeck
"""

import numpy as np
import ClimBoxFunc
from scipy import integrate
import matplotlib.pyplot as plt
import ClimDataClean 
import scipy.interpolate as interp


def box_model(t,N, func_co2_lu, func_co2_ff,
              N0=[615,842,9744, 26280, 90000000, 731, 1238],beta=0.42):
    from ClimBoxFunc import n6_flux, PgC_buffer_factor
    
    n1,n2,n3,n4,n5,n6,n7 = N
    n1_0, n2_0, n3_0, n4_0, n5_0, n6_0, n7_0 = N0

    # Carbon Cycle Box Model Setup   
    #-----------------------------------------------------------------------------
    ## Box 1: Atmosphere
    #-----------------------------------------------------------------------------
    ### Initial Conditions
    n1_0 = n1_0 
    n1_0_ppm = 289      #### Global PPM in 1750 (pre-industrial)
    ### Tansfer Coefficients to other boxes
    k_12 = 60/615
    #-----------------------------------------------------------------------------
    ## Box 2: Surface Ocean
    #-----------------------------------------------------------------------------
    ### Initial Conditions
    n2_0 = n2_0 ### PgC of carbon content on sea floor in 1750
    ### Tansfer Coefficients to other boxes
    k_21 = 60/842
    k_23 = 9/842
    k_24 = 43/842
    ### Transfer functions to other boxes
    bfr =  PgC_buffer_factor(n1)    #### Buffer Factor         
    #-----------------------------------------------------------------------------
    ## Box 3: Intermediate Ocean
    #-----------------------------------------------------------------------------
    ### Initial Conditions
    n3_0 = n3_0
    ### Tansfer Coefficients to other boxes
    k_32 = 52/9744
    k_34 = 162/9744
    #-----------------------------------------------------------------------------
    ## Box 4: Deep Ocean
    #-----------------------------------------------------------------------------
    ### Initial Conditions
    n4_0 = n4_0
    ### Tansfer Coefficients to other boxes
    k_43 = 205/26280
    k_45 = 0.2/26280
    #-----------------------------------------------------------------------------
    # Box 5: Sediments
    #-----------------------------------------------------------------------------
    ### Initial Conditions
    n5_0 = n5_0
    ### Tansfer Coefficients to other boxes
    k_51 = 0.2/90e6
    #-----------------------------------------------------------------------------
    # Box 6: Bioshpere
    #-----------------------------------------------------------------------------
    ### Initial Conditions
    n6_0 = n6_0
    n6_flx_0 = 62 
    ### Tansfer Coefficients to other boxes
    k_67 = 62/731
    ### Other Constants
    npk_f = beta  # Fertilization Factor for Biosphere Flux Function
    ### Transfer functions to other boxes
    f = n6_flux(n1, n1_0_ppm , n6_flx_0, npk_f) ### Biosphere Flux
    #-----------------------------------------------------------------------------
    # Box 7: Soil
    #-----------------------------------------------------------------------------
    ### Initial Conditions
    n7_0 = n7_0
    ### Tansfer Coefficients to other boxes
    k_71 = 62/1238           
    #-----------------------------------------------------------------------------        
    ## Rate Equation
    
    ### Box 1 Equation
    d_n1_dt = -k_12*n1 + k_21*(n2_0 + (bfr * (n2 - n2_0))) + func_co2_ff(t) - f\
        + func_co2_lu(t) + k_51*n5 + k_71*n7
        
    ### Box 2 Equation           
    d_n2_dt = k_12*n1 - k_21*(n2_0 + (bfr * (n2 - n2_0))) - k_23*n2 + k_32*n3 - k_24*n2
    
    ### Box 3 Equation           
    d_n3_dt = k_23*n2 - k_32*n3 - k_34*n3 + k_43*n4
    
    ### Box 4 Equation
    d_n4_dt = k_34*n3 - k_43*n4 + k_24*n2 - k_45*n4
    
    ### Box 5 Equation
    d_n5_dt = k_45*n4 - k_51*n5
    
    ### Box 6 Equation
    d_n6_dt = f - k_67*n6 - 2*func_co2_lu(t)
    
    ### Box 7 Equation
    d_n7_dt = k_67*n6 - k_71*n7 + func_co2_lu(t) 
    #----------------------------------------------------------------------------- 
    return [d_n1_dt, d_n2_dt, d_n3_dt, d_n4_dt, d_n5_dt, d_n6_dt, d_n7_dt]

def model_ppm_error(year_ppm_data, model, time):
    # Initializing Variables
    error = 0 # Sum of residual error (abs(model - observation))
    n = 0 # Number of ppm data points compared to model
    
   
        
    for idy, year in enumerate(year_ppm_data[0]): 
        
                
        idz = np.nonzero(time == year)[0][0] ### Find index of year where ppm observiation exist
        error += abs(model[idz] - year_ppm_data[1][idy]) ### Add abs(model - observation)) to last error
        n += 1 # Update number of ppm data points compared
       
    return error / n



# Import climate Data, cleaning for dataset from 1750-2020
co2_data = ClimDataClean.clim_data_cleanup("raw_1750_2020_ff.csv" ,"raw_1850_2022_co2.csv")
    
# Create F(t) for CO2 Fossil Fuel and Land Use Data
ft_co2_ff = interp.interp1d(co2_data[0],co2_data[1],kind='cubic', fill_value='extrapolate')
ft_co2_lu = interp.interp1d(co2_data[0],co2_data[3],kind='cubic', fill_value='extrapolate')
ft_co2_tot = interp.interp1d(co2_data[0],co2_data[2],kind='cubic', fill_value='extrapolate')  
# Initial Conditions of Model
init_conds=[615,842,9744, 26280, 90000000, 731, 1238]
 
# Timespan of Model
t = np.arange(1750,2021,1)

# Solving Differential Equations
sol1 = integrate.odeint(box_model, init_conds, t,tfirst = True, 
                        args=(ft_co2_lu,ft_co2_ff,init_conds,0.42)) 
n1,n2,n3,n4,n5,n6,n7 = sol1.T ## Unpacking Solutions




# Calculations for Plotting
co2_ff_gt = ClimBoxFunc.PgC_to_Gt(ft_co2_ff(t)) ## Fossil Fuels from PgC to GT
co2_lu_gt = ClimBoxFunc.PgC_to_Gt(ft_co2_lu(t)) ## Land Use from PgC to GT
co2_tot_gt = co2_ff_gt + co2_lu_gt ## Computing Total CO2
model_ppm = ClimBoxFunc.PgC_to_ppm(n1) ## Atmosphere PgC to CO2 ppm
n1_0_ppm = ClimBoxFunc.PgC_to_ppm(init_conds[0]) # For Temperature Anomaly Calculation

# Importing Observation Data [[year],[ppm],[atemp]]
year_ppm_atemp = ClimDataClean.year_ppm_atemp_cleanup('raw_ppm_1750_2020.csv','raw_year_temp.csv')


ppm_slope, ppm_yint = np.polyfit(year_ppm_atemp[1],year_ppm_atemp[2],1)

# Setting Up % Change Bar Graph for Boxes
bar_labels = ['N1',"N2","N3", "N4", "N5","N6","N7"]

bar_colors = ["#ffee65","#7eb0d5","#8bd3c7","#beb9db","#bd7ebe","#b2e061","#6d4b4b"]
 
deltas = [round(abs(n1[0]-n1[-1])/n1[0] * 100,1),
          round(abs(n2[0]-n2[-1])/n2[0] * 100,1),
          round(abs(n3[0]-n3[-1])/n3[0] * 100,1),
          round(abs(n4[0]-n4[-1])/n4[0] * 100,1),
          round(abs(n5[0]-n5[-1])/n5[0] * 100,1),
          round(abs(n6[0]-n6[-1])/n6[0] * 100,1),
          round(abs(n7[0]-n7[-1])/n7[0] * 100,1),]
    
n_bars = np.arange(len(deltas))

# Plotting Results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,constrained_layout=True)

ax1.set_title(f'Input: $CO_{2}$')
ax1.plot(t,co2_ff_gt , label="Fossil Fuel", color='b')
ax1.plot(t,co2_lu_gt , label="Land use", color='g')
ax1.plot(t,co2_tot_gt, label="Total", color='r')
ax1.set_ylabel(f'$CO_{2}$ Emissions (Gt per year)')
ax1.set_xlabel("Year") 
ax1.grid()
ax1.legend()

ax2.set_title('Components: PgC % Change (1750-2020)')
bars = ax2.bar(n_bars,deltas,color = bar_colors)
ax2.bar_label(bars)
ax2.set_xlabel('Box')
ax2.set_ylabel('% Change')
ax2.set_xticks(n_bars,bar_labels)
ax2.legend()
ax2.grid()

ax3.set_title(f'Output: $CO_{2}$')
ax3.plot(t,model_ppm, label="Model", color='#0C090A')
ax3.scatter(year_ppm_atemp[0],year_ppm_atemp[1], label="Measured", marker =".", color='c')
ax3.set_ylabel(f'$CO_{2}$ Concentration (ppm)')
ax3.grid()
ax3.legend()
ax3.set_xlabel("Year")

ax4.set_title('Output: Temperature Anomaly ')
ax4.scatter(ClimBoxFunc.PgC_to_ppm(n1), ClimBoxFunc.temp_anomaly(n1,ClimBoxFunc.PgC_to_ppm(n1[0])),
            label="Model", color='#0C090A')
ax4.scatter(year_ppm_atemp[1],year_ppm_atemp[2], label="Measured", marker =".", color='c')
ax4.plot(year_ppm_atemp[1],ppm_slope * year_ppm_atemp[1]+ ppm_yint,
         label=f"best fit: {ppm_slope:.3f} X ppm {ppm_yint:.3f}", color='b')
ax4.set_ylabel(r'Global Temp. Anomaly $^{\circ}$C')
ax4.set_xlabel(f'$CO_{2}$ Concentration (ppm)')
ax4.grid()
ax4.legend()

fig.savefig("CO2_1750_2020.png")


# Trying Different Betas

## Defining Beta Range
beta_rng = np.arange(0.33,0.46,0.02)

## Initiating Variables
beta_models_error = np.zeros(len(beta_rng)) ## Mean Residual Error
beta_models_ppm = []  

fig1, ax5 = plt.subplots()


for idx, coef in enumerate(beta_rng):
    
    temp = integrate.odeint(box_model, init_conds, t,tfirst = True,
                            args=(ft_co2_lu,ft_co2_ff,init_conds,coef)).T
    
    beta_models_ppm.append(ClimBoxFunc.PgC_to_ppm(temp[0]))
    
    beta_models_error[idx] = model_ppm_error(year_ppm_atemp, beta_models_ppm[idx],t)
    
    ax5.plot(t,ClimBoxFunc.PgC_to_ppm(temp[0]), 
             label=f'Model($beta$={coef:.2f}: Error={beta_models_error[idx]:.2f}ppm')
    

    
ax5.set_title(f'Output: $CO_{2}$ Output')
ax5.scatter(year_ppm_atemp[0],year_ppm_atemp[1], label="Measured", marker =".", color='c')
ax5.set_ylabel(f'$CO_{2}$ Concentration (ppm)')
ax5.grid()
ax5.legend()
ax5.set_xlabel("Year")    
fig1.savefig("ppm_beta.png")







