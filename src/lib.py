import pandas as pd 
import xarray as xr 
import os

#currently unnecessary
#import numpy as np 
#import matplotlib.pyplot as plt 
#import scipy
#import scipy.stats as stats
#import seaborn as sns

xr.set_options(display_style='html')
#https://stackoverflow.com/questions/5833623/mplconfigdir-error-in-matplotlib
#os.environ['MPLCONFIGDIR'] = 'temp/'
from netCDF4 import Dataset

import netCDF4

#Set directories
# TODO setup to use with input --dir
home_dir = '../data/Spatial-and-Annual-Averages/'

gnu_dir = home_dir+'GNU-runs/'
intel_dir = home_dir+'Intel-runs/'
test_dir = home_dir+'Test-runs/'

ensemble_dir = '../data/Ensemble_project/'

#output directory
local_dir = '../output/'


### Start empty DataFrame here

def get_data(dir = ensemble_dir, in_file = 'ensemble_all.csv'):
  '''
  if no dir given, take file from ensemble_dir else
  write new file if ensemble_all doesn't exist in dir
  '''
  #verify file name, if exists: 
  if os.path.exists(dir+in_file):
    print("found file, reading now")
    df = pd.read_csv(dir+in_file, low_memory=False)
    df = df.drop(axis=1,columns=['Unnamed: 0'])
    #df[df.TS < 280] = np.nan
    df['Simulation'] = [str(my_str).zfill(3) for my_str in df['Simulation']]
    
  else: 
    print("no file found, creating new")
    df = pd.DataFrame()

    ### Append all data into this DataFrame
    for my_dir, my_label in zip([gnu_dir,intel_dir,test_dir],['GNU','Intel','Test']):
      for filename in os.listdir(my_dir):
        ## check extension
        if filename.endswith("nc"): 
            #print(filename)
            simulation = filename.split('.')[-2]
            #print(simulation)
            ds = xr.open_dataset(my_dir+filename)
            newdf = ds.to_dataframe()
            newdf = newdf.reset_index(level=[0,1])
            newdf['varNames'] = newdf['varNames'].str.decode("utf-8")


            newdf = newdf.pivot_table(index = ['nyear'],columns=['varNames'],values=['spatialAvgs'])
            newdf.columns = newdf.columns.droplevel(0)
            # Label the simulations with 'GNU','Intel', or 'Test'
            newdf['Label'] = my_label
            newdf['Simulation'] = str(simulation)
            newdf = newdf.reset_index()
            df = df.append(newdf,ignore_index = True)
    df.Simulation = df.Simulation.astype(str)
    df.to_csv(local_dir+in_file)
  
  return df

def add_droplist(label, index, droplist = None):
  if not droplist:
    droplist = []
  index = str(index)
  tup = (label, index)
  droplist.append(tup)
  return droplist

def df_droplist(df, droplist):
  #drop from droplist
  if droplist:
    for each in droplist:
      print("dropping ", each)
      label = each[0]
      index = each[1]
      df.drop(index = df[(df.Label == label) & (df.Simulation.str.contains(index))].index,inplace = True)
  return df



