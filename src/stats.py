import scipy.stats as stats
import numpy as np
import pandas as pd
### For PCA packages
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

### Define function for 2-samp KS test summary stats
def stats_table(df, ctrl_label, compare_label, compare_simu, test_list, use_year = 90, my_var = 'TS', remove_var_list = None):
  ### Test all test runs
  if ctrl_label not in np.unique(df.Label) or compare_label not in np.unique(df.Label):
      print('ERROR: Input ensemble label does not exist.')
  else:
    summary_stat = pd.DataFrame()
    one_line = {
        'Dist_A':'',
        'Dist_B':'',
        'KS-stats':0,
        'KS-pvalue':0
    }
    nyear = len(np.unique(df.nyear))
    df_x = df[(df.Label == ctrl_label) & (df.nyear >= nyear - use_year)]
    run_list = np.unique(df_x.Simulation.astype(float))
    for my_test in test_list:
      new_line = one_line.copy()
      ### df_y is the DataFrame for test run
      df_y = df[(df.Simulation == my_test) & (df.nyear >= nyear - use_year)]
      x = np.array(df_x[my_var])
      y = np.array(df_y[my_var])

      '''
      For KL divergence
      
      bins = np.linspace(np.nanmin(np.hstack((x,y))),np.nanmax(np.hstack((x,y))),40)



      p = plt.hist(x,bins=bins)[0]
      q = plt.hist(y,bins=bins)[0]
      p = p/p.sum(axis=0, keepdims=True) 
      q = q/q.sum(axis=0, keepdims=True) 
      #Sp = stats.entropy(np.array(x))
      #print('KL-entropy of A: ',Sp)
      #kl_res = KL(p,q)
      #print('KL-divergence: ',kl_res)
      '''

      #print('This is test run: ',my_test)
      res = stats.ks_2samp(x,y)
      #print('KS-statistics: ',res)
      
      new_line['Dist_A'] = my_test
      new_line['Dist_B'] = str(len(run_list))+' Intel'
      new_line['KS-stats'] = round(res[0],2)
      new_line['KS-pvalue'] = round(res[1],2)
      #new_line['KL-entropy'] = round(Sp,2)
      #new_line['KL-div'] = round(kl_res,2)

      summary_stat = summary_stat.append(new_line,ignore_index = True)

    ### Add GNU 000 for reference
    if compare_label:
      if compare_simu:
        df_y = df[(df.Label == compare_label) & (df.Simulation == compare_simu)]
        compare_name = compare_label#+'.'+compare_simu
      else: ### No compare_simu input: use all GNU ensemble
        df_y = df[(df.Label == compare_label)]
        compare_name = compare_label
      x = np.array(df_x[my_var])
      y = np.array(df_y[my_var])
      '''
      bins = np.linspace(np.nanmin(np.hstack((x,y))),np.nanmax(np.hstack((x,y))),40)
      
      p = plt.hist(x,bins=bins)[0]
      q = plt.hist(y,bins=bins)[0]
      p = p/p.sum(axis=0, keepdims=True) 
      q = q/q.sum(axis=0, keepdims=True) 
      #z = np.array(test_df[my_var])



      # Note slight difference in the final result compared to Dawny33
      print(KL(p, q)) # 0.775278939433
      kl_res = KL(p,q)
      Sp = stats.entropy(np.array(x))
      print('KL-entropy of A: ',Sp)
      print('KL-divergence: ',kl_res)
      print('This is GNU.000')
      #new_line['KL-entropy'] = round(Sp,2)
      #new_line['KL-div'] = round(kl_res,2)
      '''
      res = stats.ks_2samp(x,y)
      #print('KS-statistics: ',res)
      
      ctrl_name = str(len(run_list))+' '+ctrl_label
      new_line = one_line.copy()
      ### It is worth considering whether dist A and B should be compare first and then ctrl ensemble
      new_line['Dist_A'] = compare_name
      new_line['Dist_B'] = ctrl_name
      new_line['KS-stats'] = round(res[0],2)
      new_line['KS-pvalue'] = round(res[1],2)

      summary_stat = summary_stat.append(new_line,ignore_index = True)
  return summary_stat#[['Dist_A','Dist_B','KS-stats','KS-pvalue']].transpose()


### Define function for 2-sample KS test visualization
def ks_2samp_plotting(df, ctrl_label, compare_label, my_var = 'TS', my_year = 'all', ci = 0.05, matched = False):
  '''
  ### input values
  ``my_var'' is the variable name to be considered. 
  ``my_year'' is set to 'all' if all years are aggregated, and set to a particular year from the beginning to use simulations at only one year. Should check the stationarity of those years before defining a number. 
  ``ci'' is the confidence interval, default value is 0.05.
  ``matched'' is set to True if the two ensembles are paired, default is False. 
  '''
  nyear = len(np.unique(df.nyear))
  if my_year == 'all':
    year_list = range(nyear)
  else:
    year_list = list(my_year)
  if matched:
    match_list = [str(simu).zfill(3) for simu in range(25)]
    match_list.remove('000')
    df_x = df[(df.Label == ctrl_label) & (df.Simulation.isin(match_list)) & (df.nyear.isin(year_list))]
    df_y = df[(df.Label == compare_label) & (df.Simulation.isin(match_list)) & (df.nyear.isin(year_list))]
  else:
    df_x = df[(df.Label == ctrl_label) & (df.nyear.isin(year_list))]
    df_y = df[(df.Label == compare_label) & (df.nyear.isin(year_list))]


  

  fig = plt.figure(3,figsize = (8,6))
  
  res = stats.ks_2samp(np.array(df_y[my_var]),np.array(df_x[my_var]))

  bins = np.linspace(np.nanmin([df_x[my_var].min(),df_y[my_var].min()]),np.nanmax([df_x[my_var].max(),df_y[my_var].max()]),20)
  ax.hist(df_x[my_var], bins=bins, density=True, histtype='step', cumulative=True,
          label=ctrl_label,lw=2)
  ax.hist(df_y[my_var], bins=bins, density=True, histtype='step', cumulative=True,
          label=compare_label,lw=2)
  ax.set_title('nyear: {my_year}, ks-pvalue: {pval}'.format(my_year= my_year,pval = round(res[1],2)))
  ax.legend(loc = 2,frameon=False, fontsize = fs-2)
  ax.set_xlabel(my_var,fontsize = fs-2)
  ax.set_ylabel('CDF',fontsize = fs-2)
  ax.set_xticks(fontsize = fs-2)
  ax.set_yticks(fontsize = fs-2)
  plt.locator_params(nbins=5)
  return ax

### The final version used
def new_get_PCs(single_intel_run, remove_var_list,sample_year = False, use_year = 100): 
  # Version from Ziwei
  var_df = pd.read_csv('../output/CESM_variable_type.csv')
  var_df = var_df.rename(columns = {'type':'my_type'})
  column_list = list(single_intel_run.drop(axis=1,columns=['nyear','Label','Simulation']).columns)

  if remove_var_list:
    removed_column_list = column_list
    for my_col in remove_var_list:
      removed_column_list.remove(my_col)
    feat_cols = removed_column_list
    single_intel_run = single_intel_run.drop(remove_var_list, axis=1)
  else:
    feat_cols = column_list
  
  single_intel_run_no_lab = single_intel_run.drop(['nyear','Label', 'Simulation'], axis=1)
  

  single_intel_run_no_lab.head()

  # Standardizing the features
  single_intel_run_standardized = StandardScaler().fit_transform(single_intel_run_no_lab)
  single_intel_run_stand_df = pd.DataFrame(single_intel_run_standardized,columns=feat_cols)
  single_intel_run_stand_df.head()
  my_shape = np.shape(single_intel_run_stand_df)
  n_components = np.nanmin((my_shape[0],my_shape[1]))#min

  pca_single_intel_run = PCA(n_components=n_components,svd_solver = 'full')

  if sample_year:
    all_PCs = pca_single_intel_run.fit_transform(single_intel_run_stand_df)
    #ts = np.dot(all_PCs,single_intel_run_standardized)
  else:
    all_PCs = pca_single_intel_run.fit_transform(single_intel_run_stand_df.T)
  

  all_EOFs = pca_single_intel_run.components_  
  n_samples = pca_single_intel_run.n_samples_
  n_components = pca_single_intel_run.n_components_
  n_features = pca_single_intel_run.n_features_
  explained_variance = pca_single_intel_run.explained_variance_ratio_
  return(all_PCs,all_EOFs,n_samples,n_components,n_features,explained_variance)

### Define function for PCA package
def pca_analysis(df, label, remove_var_list, use_year = 100,use_runs = 'all',sample_year = False):
  '''
  ``df'' is the full DataFrame, label is the tag of ensemble to be used in deriving PCA. 
  ``use_year'' is the number of years to be used from the end. use_year = 90 means exclude first 10 years if we have a full simulation of 100 years. 
  ``use_runs'' is a string 'all' or the number of simulations to be used in the given ensemble. Detault is 'all'. 
  ``remove_var_list'' is the list of variables to be removed from the PCA analysis. These variables includes, but are not limited to, the ones that are constant, and the ones that have unphysical spurious peaks.
  ``sample_year'' is a switch to indicate if the sample dimension is time. By default is True.  
  '''
  
  #intel_runs_only = intel_runs_only[(intel_runs_only['Simulation'] != '000') & (intel_runs_only['Simulation'] != '115')]
  nyear = len(np.unique(df[df.Label == label].nyear))
  simu_list = (pd.unique(df[df.Label == label].Simulation.astype(float)))
  nruns = len(simu_list)
  df = df[df.Label == label]

  ### Removed variable list
  var_df = pd.read_csv('../output/CESM_variable_type.csv')
  var_df = var_df.rename(columns = {'type':'my_type'})
  column_list = list(df.drop(axis=1,columns=['nyear','Label','Simulation']).columns)
  if remove_var_list:
    removed_column_list = column_list
    for my_col in remove_var_list:
      removed_column_list.remove(my_col)
    feat_cols = removed_column_list
  else:
    feat_cols = column_list

  if use_year > nyear:
    print('ERROR: use_year exceed maximum of years.')
  elif use_runs == 'all':
    newdf = df[(df.Label == label) & (df.nyear >= nyear - use_year)]
    all_PCs,all_EOFs,n_samples,n_components,n_features,explained_variance  = new_get_PCs(newdf,remove_var_list)
  elif np.isfinite(use_runs):
    ### use_runs is a number; produce 3D array of PCs, with PCA applied to each simulation
    if use_runs > nruns:
      print('ERROR: use_runs exceed maximum of simulations.')
    else:
      rand = np.random.randint(0,nruns,use_runs)
      rand_list = [str(simu_list[isimu]) for isimu in range(len(rand))]
      ### Initialize new arrays
      if sample_year:
        intel_run_PC_array = np.zeros((nruns, use_year, use_year))
        intel_run_EOF_array = np.zeros((nruns, use_year,len(feat_cols)))
        explained_var_array = np.zeros((nruns, use_year))
      else:
        intel_run_PC_array = np.zeros((nruns, len(feat_cols), use_year))
        intel_run_EOF_array = np.zeros((nruns, use_year, use_year))
        explained_var_array = np.zeros((nruns, use_year))
      
      for isimu,sim_index in enumerate(simu_list):
        newdf = df[(df.Label == label) & (df.nyear >= nyear - use_year) & (df.Simulation.astype(float) == sim_index)] 
        intel_run_PC_array[isimu,:,:], intel_run_EOF_array[isimu,:,:],n_samples,n_components,n_features,explained_var_array[isimu,:]= new_get_PCs(single_intel_run=newdf,remove_var_list=remove_var_list)
      all_PCs = intel_run_PC_array
      all_EOFs = intel_run_EOF_array
      explained_variance = explained_var_array

  return(all_PCs,all_EOFs,n_samples,n_components,n_features,explained_variance)

### Sort variables by PCA
def sorted_variables_pca(df, all_PCs, remove_var_list, nmodes = 9):
  '''
  This function sorts the variables by the contribution to the first several modes (no repeat). 
  ``nmodes'' specify how many PCA modes are considered. Default is 9. 
  '''
  sorted_column_list_pc = []
  removed_column_list = list(df.drop(axis=1,columns=['nyear','Label','Simulation']).columns)
  for my_col in remove_var_list:
    removed_column_list.remove(my_col)
  nvar_each_mode = int(len(removed_column_list)/(nmodes-1))
  for imode in range(nmodes):
    sorted_list= sorted(zip(abs(all_PCs[:,imode]),removed_column_list))[::-1]
    var_idx = 0
    column_list_idx = 0
    print('Mode'+str(imode))
    if imode < nmodes-1:
      var_idx = 0
      while var_idx < nvar_each_mode:
        if sorted_list[column_list_idx][1] not in sorted_column_list_pc:
          sorted_column_list_pc.append(sorted_list[column_list_idx][1])
          var_idx +=1
        column_list_idx+=1
    else:
      while column_list_idx < len(removed_column_list):
        if sorted_list[column_list_idx][1] not in sorted_column_list_pc:
          sorted_column_list_pc.append(sorted_list[column_list_idx][1])
          column_list_idx +=1
        else:
          column_list_idx+=1
  return sorted_column_list_pc, nvar_each_mode


### Autocorrelation function -- credit to Curtis
def autocorr(x,maxlag):
  mean_x = np.mean(x)
  deviations_x = x - mean_x
  variance_x = np.var(x,ddof=0)
  autocorrelation = np.zeros((maxlag,))
  for lag in range(maxlag):
    deviations_shifted = np.roll(deviations_x,-lag)
    if variance_x != 0:
      autocorrelation[lag] = np.mean(deviations_x[0:len(deviations_x - lag)]*deviations_shifted[0:len(deviations_shifted - lag)])/variance_x
    else:
      autocorrelation_lag = 0
  return autocorrelation