#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Climate Data Cleaning
Created on Fri Oct 28 14:27:36 2022
@author: michaelbeck
"""
import pandas as pd
import numpy as np

def clim_data_cleanup(csv_1750_2020:str,csv_1850_2020:str):
    """
    Function to Cleanup CO2 input data from ourworldindata.org 

    Parameters
    ----------
    csv_1750_2020 : str
        Name of specific CSV data file from ourworldindata.org
    csv_1850_2020 : str
        Name of specific CSV data file from ourworldindata.org

    Returns
    -------
    co2_data : np.array
        Array in format [[year: 1750:2022],[CO2_ff],[CO2_tot],[CO2_lu]]

    """
   
    # Defining dictionary used to map col_names and dtype to 1850-2020 CSV Data
    df1_dic = {"entity": str,
               "code":str,
               "year":int,
               "co2_tot":int,
               "co2_ff":int,
               "co2_lu":int}
    
    # Importing 1850-2020 CSV Climate Data
    raw_data_df1 = pd.read_csv(csv_1850_2020,
                               names=list(df1_dic.keys()),
                               dtype=df1_dic, header=0)
    
    # Isolating relevent data: World data for C02 Total, Fossil Fuels, Land Use
    world_raw_data_df1 = raw_data_df1.loc[raw_data_df1['entity'] == "World"]\
        [['year','co2_tot','co2_ff','co2_lu']].copy().reset_index(drop=True)
        
    
    # Defining dictionary used to map col_names and dtype to 1750-2020 CSV Data
    df2_dic = {"entity":str,
               "code":str,
               "year":int,
               "co2_ff":int}
    
    # Importing 1750-2020 CSV Climate Data
    raw_data_df2 = pd.read_csv(csv_1750_2020,
                               names=list(df2_dic.keys()),
                               dtype=df2_dic, header=0)
    
    
    # Isolating relevent data: World data before 1850, only C02 Fossil Fuels
    world_raw_data_df2 = raw_data_df2.loc[(raw_data_df2['entity'] == "World")\
                                          & (raw_data_df2['year'] < 1850)]\
        [['year','co2_ff']].copy().reset_index(drop=True)
    
    
    # Combine dataframes together
    co2_df = pd.concat([world_raw_data_df2,world_raw_data_df1],
                       axis=0,ignore_index=True).copy()               
    # Setting Year as DF Index
    co2_df.set_index('year', inplace = True)
    
    # Converting into PgC (Units of Box Model)
    co2_df["co2_lu"] = co2_df.loc[:,"co2_lu"] / 3.667 / 1e9
    co2_df["co2_ff"] = co2_df.loc[:,"co2_ff"] / 3.667 / 1e9
    co2_df["co2_tot"] = co2_df.loc[:,"co2_tot"] / 3.667 / 1e9 
    
    # Interpolation of CO2 Land Use between 1750 and 1850 (Filling up NaN)
    co2_df.at[1750,"co2_lu"] = 0 ## Setting 1750 Value to 0
    co2_df["co2_lu"].interpolate(method='linear',inplace=True) # Linear Interpolation
    
    # Computing CO2 Total based on 1750-1849 FF and Linear Interpolation for LU data 
    co2_df.loc[1750:1850].eval('co2_tot = co2_ff + co2_lu', inplace=True)
    co2_df.to_csv("project_co2.csv") # Export for Manual Check
    
    # Formating as array [[Year],[FF],[TOT],[LU]]
    co2_data = np.transpose(co2_df.to_records(index=True))  
    co2_data = np.array(list(zip(*co2_data)))

    return co2_data

def year_ppm_cleanup(year_ppm_csv:str,retrn_type = 'np.arr'):
    """
    Function to cleanup PPM

    Parameters
    ----------
    year_ppm_csv : str
        Name of specific CSV data file from ourworldindata.org
    retrn_type : str, optional
        Return type, either np.arr, pd.df or "csv". The default is 'np.arr'.

    Returns
    -------
    either np.arr or pd.df or nothing
         [[year],[ppm]] data
    """
    # Defining dictionary used to map col_names and dtype to Climate Change CSV Data
    df1_dic = {"entity": str,
               "year":int,
               "co2_ppm":float,
               "ch4_ppm":float,
               "n2o_ppm":float,
               "feb":float,
               'sept':float,
               'us_glacier_mass': float,
               'ice_CSIRO': float,
               'ice_IAP': float,
               'ice_MRIJMA': float,
               'ice_NOAA': float,
               'snow_cover': float,
               'sea_surf_temp_mean': float,
               'sea_surf_temp_lb': float,
               'sea_surf_temp_ub': float,
               'sea_IAP': float,
               'sea_NOAA': float,
               'sea_MRIJMA': float,
               'artic_sea_ice': float}
    
    # Importing Climate Change CSV Data
    raw_data_df1 = pd.read_csv(year_ppm_csv,
                               names=list(df1_dic.keys()),
                               dtype=df1_dic, header=0)
    
    # Isolating relevent data: World data before 1850, only C02 Fossil Fuels
    world_raw_data_df1 = raw_data_df1.loc[(raw_data_df1['entity'] == "World")\
                                          & (raw_data_df1['year'] > 1750)
                                          & (raw_data_df1['co2_ppm'].notnull() == True)]\
        [['year','co2_ppm']].copy().reset_index(drop=True)
    
    # Setting Year as DF Index
    world_raw_data_df1.set_index('year', inplace = True)    
    
    
    if retrn_type == 'np.arr':   
        # Formating as array [[Year],[ppm]]
        co2_ppm_data = np.transpose(world_raw_data_df1.to_records(index=True))  
        co2_ppm_data = np.array(list(zip(*co2_ppm_data)))
        
        return co2_ppm_data
    
    elif retrn_type == 'pd.df':
        
        return world_raw_data_df1
    
    elif retrn_type == 'csv':
        world_raw_data_df1.to_csv("year_co2.csv")
        print("\nGenerated: year_co2.csv")
        
        return

def year_atemp_cleanup(year_atemp_csv:str,retrn_type:str = 'np.arr'):
    """
    
    Function to cleanup anomaly temperature raw input

    Parameters
    ----------
    year_atemp_csv : str
        Name of specific CSV data file from NASA
    retrn_type : str, optional
         Return type, either np.arr, pd.df or "csv". The default is 'np.arr'.

    Returns
    -------
    either np.arr or pd.df or nothing
         [[year],[a_temp]] data
   """
    
    # Defining dictionary used to map col_names and dtype to Climate Change CSV Data
    df1_dic = {"year":float,
               "temp_anomaly":float}
    
    # Importing Temperature CSV Data
    raw_data_df1 = pd.read_csv(year_atemp_csv,
                               names=list(df1_dic.keys()),
                               dtype=df1_dic, header=1)
    
    raw_data_df1.year = raw_data_df1.year.astype(int)
    df_atemp = raw_data_df1.groupby('year').mean().copy()
       
    if retrn_type == 'np.arr':   
        # Formating as array [[Year],[ppm]]
        year_atemp_data = np.transpose(df_atemp.to_records(index=True))  
        year_atemp_data = np.array(list(zip(*year_atemp_data)))
        
        return year_atemp_data
    
    elif retrn_type == 'pd.df':
        
        return df_atemp
    
    elif retrn_type == 'csv':
        df_atemp.to_csv("year_atemp.csv")
        print("\nGenerated: year_atemp.csv")
        return 
def year_ppm_atemp_cleanup(year_ppm_csv:str,year_atemp_csv:str,retrn_type:str = 'np.arr'):
    """
    Combines two specific [[year],[ppm]] and [[year],[atemp]] dataframes together. 

    Parameters
    ----------
    year_ppm_csv : str
        Name of specific CSV data file from ourworldindata.org
    year_atemp_csv : str
        Name of specific CSV data file from NASA
    retrn_type : str, optional
         Return type, either np.arr, pd.df or "csv". The default is 'np.arr'.

    Returns
    -------
    either np.arr or pd.df or nothing
         [[ppm],[a_temp]] data

    """
    
    # Importing CSV Files into dataframes 
    df_ppm = year_ppm_cleanup(year_ppm_csv,"pd.df")
    df_atemp = year_atemp_cleanup(year_atemp_csv,"pd.df")
    
    # Combining Dataframes on matching year
    df_ppm_atemp = pd.merge(df_ppm, df_atemp, left_index=True, right_index=True)
    df_ppm_atemp  =  df_ppm_atemp[:-2] ## Removing data frames 2021, 2022 to match other CSV inputs 
    
    
    if retrn_type == 'np.arr':   
        # Formating as array [[ppm,[Anomaly Temperature]]
        year_atemp_data = np.transpose(df_ppm_atemp.to_records(index=True))
        year_atemp_data = np.array(list(zip(*year_atemp_data)))
        
        return year_atemp_data
    
    elif retrn_type == 'pd.df':
        
        return df_ppm_atemp
    
    elif retrn_type == 'csv': # For Manual Error Checking
        df_ppm_atemp.to_csv("year_atemp.csv")
        print("\nGenerated: year_atemp.csv") 
        
        return
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    