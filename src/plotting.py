### Plotting functions
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
import aux_functions as ax
from matplotlib.colors import from_levels_and_colors
from matplotlib.colorbar import ColorbarBase
#from skimage.segmentation import mark_boundaries

import stats as st


### Single plot QQ-plot for one variable
def qqplot(ctrl_label, compare_label,  my_var, matched = True, test_list = None, dperc = 1):
    '''
    ``ctrl_label'' and ``compare_label'' are the names of the control ensemble. 
    and the ensemble with some physical or computational change amended. 
    ``var_name'' is the variable name to be evaluated, matching the format 
    of field naming convention in the data provided. 
    ``matched'' is a switch to determine whether or not to subset to paired simulations 
    in control and compare ensembles. 
    ``test_list'' is a list of single test runs, each with one physical or 
    computational parameter changed. By default, these lines won't be shown. 
    ``dperc'' is the step of percentiles from 0th to 100th perncentiles. 
    By default, ``dperc'' is 1(%). 
    '''
    ### Check input variable, label type
    if my_var not in np.unique(col_df.columns):
      print('ERROR: Input variable does not exist.')
    elif ctrl_label not in np.unique(df.Label) or compare_label not in np.unique(df.Label):
      print('ERROR: Input ensemble label does not exist.')
    else:
      ### Label and variable input proper
      newdf = df.dropna(subset=[my_var]).copy()
      fig,ax = plt.subplots(figsize = (8,8))
      my_perc = np.arange(0,100,dperc)

      if matched:
        #match_list = [str(simu).zfill(3) for simu in range(25)]
        match_list = np.intersect1d(newdf[(newdf.Label == ctrl_label)]['Simulation'],newdf[(newdf.Label == compare_label)]['Simulation'])
        if len(match_list) > 0:
          print('Successfully found paired simulations.')
          ctrl_perc = np.nanpercentile(newdf[(newdf.Label == ctrl_label) & (newdf.Simulation.isin(match_list))][my_var],my_perc)
          compare_perc = np.nanpercentile(newdf[(newdf.Label == compare_label) & (newdf.Simulation.isin(match_list))][my_var],my_perc)
        else:
          print('ERROR: No paired simulations found. ')
      else:
        ctrl_perc = np.nanpercentile(newdf[newdf.Label == ctrl_label][my_var],my_perc)
        compare_perc = np.nanpercentile(newdf[newdf.Label == compare_label][my_var],my_perc)

      #ax = plt.subplot()
      vmin = np.round(np.nanmin([ctrl_perc,compare_perc]),decimals=2)
      vmax = np.round(np.nanmax([ctrl_perc,compare_perc]),decimals=2)

      ax.plot([vmin,vmax],[vmin,vmax],'--k')
      ax.plot(ctrl_perc,compare_perc,'--b',label = compare_label)
      ax.set_xlabel(ctrl_label,fontsize = 12)
      if np.size(test_list) > 0:
        ### If test_list input is not None
        ax.set_ylabel('{compare_label}/Test'.format(compare_label = compare_label),fontsize = 12)
        #color_list = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf']      
        ncolor = len(test_list)
        cm = pylab.get_cmap('gist_rainbow')
        color_list = [cm(1.*i/ncolor) for i in range(ncolor)]

        for my_test,my_color in zip(test_list,color_list):
          test_perc = np.nanpercentile(newdf[(newdf.Simulation == my_test)][my_var],my_perc)
          ax.plot(ctrl_perc,test_perc,c = my_color,label = my_test,lw = 2)
      else:
        ax.set_ylabel(compare_label,fontsize = 12)
      
      
      ax.set_xticks(np.round(np.linspace(vmin, vmax, 5), 2))
      ax.set_yticks(np.round(np.linspace(vmin, vmax, 5), 2))


      ax.set_title(my_var)
      ax.legend()
      fig.patch.set_facecolor('xkcd:white')

    return ax


### Example script
#match_list = [str(simu).zfill(3) for simu in range(25)]
#ctrl_label = 'Intel'
#compare_label = 'GNU'
#test_list = np.unique(df[df.Label == 'Test']['Simulation'])
#qqplot(ctrl_label,compare_label,'TS',test_list = test_list)

### Heatmap for a given set of test runs or ensemble against control ensemble. Color coded by value column. 


def heatmap_zscore(df, ctrl_label, compare_label, remove_var_list = None, test_list = None, fs = 12):
  ### First check if input label names are proper: Maybe we should define this as an attribute of a class? 
  if ctrl_label not in np.unique(df.Label) or compare_label not in np.unique(df.Label):
      print('ERROR: Input ensemble label does not exist.')
  else:
    ### load variable DataFrame produced by get_var_type.py
    var_df = pd.read_csv('../output/CESM_variable_type.csv')
    var_df = var_df.rename(columns = {'type':'my_type'})
    removed_column_list = list(df.drop(axis=1,columns=['nyear','Label','Simulation']).columns)
    for my_col in remove_var_list:
      removed_column_list.remove(my_col)
    new_var_df = var_df[var_df.variable.isin(removed_column_list)].drop(columns=['Unnamed: 0','index'])


    single_intel_run = df[(df.Label == ctrl_label)]
    single_intel_run_no_lab = single_intel_run.drop(['nyear','Label', 'Simulation'], axis=1)
    single_intel_run_no_lab = single_intel_run_no_lab.drop(remove_var_list, axis=1)

    single_gnu_run = df[(df.Label == compare_label)]
    single_gnu_run_no_lab = single_gnu_run.drop(['nyear','Label', 'Simulation'], axis=1)
    single_gnu_run_no_lab = single_gnu_run_no_lab.drop(remove_var_list, axis=1)

    single_run_norm = np.array(abs(single_gnu_run_no_lab.mean() - single_intel_run_no_lab.mean())/single_intel_run_no_lab.std()).reshape(-1,)

    print(var_df.columns)
    new_var_df[compare_label] = list(single_run_norm)
    print((single_run_norm))
    if test_list:
      for my_test in test_list:
        single_run = df[(df.Simulation == my_test)]
        single_run_no_lab = single_run.drop(['nyear','Label', 'Simulation'], axis=1)
        single_run_no_lab = single_run_no_lab.drop(remove_var_list, axis=1)
        single_run_norm = np.array(abs(single_run_no_lab.mean() - single_intel_run_no_lab.mean())/single_intel_run_no_lab.std()).reshape(-1,)

        new_var_df[my_test] = list(single_run_norm)


    ### Draw heatmap:
    my_type_list = np.unique(new_var_df['my_type'])
    mapping = {  my_type_list[i] : i+1 for i in range(0, len(my_type_list) ) }
    #mapping = {'Flux':1,'Cloud':2,'Aerosol':3,'Precipitation':4,'Height':5,'Pressure':6,'Temperature':7,'Transport':8,'Fraction':9}
    new_var_df['my_color'] = new_var_df['my_type'].map(mapping)
    new_var_df['sort_order'] = new_var_df['my_color'] - 1

    my_type_list = [my_type for my_type in mapping]
    #nine_type_list = ['Flux','Cloud','Aerosol','Precipitation','Height','Temperature','Pressure','Transport','Fraction']

    df_reorder = new_var_df.sort_values(by = 'my_color')
    #np.unique(df_reorder.my_type)

    new_var_df = new_var_df.sort_values(by=['sort_order','my_type'])
    new_var_df = new_var_df.reset_index()
    sorted_column_list = new_var_df.variable
    sorted_column_list = list(sorted_column_list)
    sorted_type_list = new_var_df.my_type
    

    #data = np.random.randn(6, 6)
    nvar = len(removed_column_list)
    nmodes = len(test_list)

    data = np.zeros((nvar,nmodes))
    z_array = np.zeros((nvar,nmodes))

    for itest,my_test in enumerate(test_list):
      #data_df
      df_reorder = new_var_df.copy()#.sort_values(by = my_test,ascending=True)
      #data[:,itest] = df_reorder['my_color']
      z_array[:,itest] = df_reorder[my_test]
      #my_sig = 'pc_sig'+str(imode+1)
    data_df = new_var_df.copy()


    ### Color coding by KS p-value
    fig, ax = plt.subplots(figsize=(8,8))

    bounds = [0., 0.05,0.1, 0.25,0.5,0.75,1,2,5,10]
    clist = discrete_color_list('YlGnBu',len(bounds)-1)
    cmap= colors.ListedColormap(clist)
    norm = colors.BoundaryNorm(bounds, cmap.N, clip=True)


    ### Construct new_alpha matrix that repeat the boundary rows and columns for plotting
    new_alpha = z_array.copy()
    new_alpha = np.vstack((new_alpha[0,:],new_alpha,new_alpha[-1,:]))
    new_alpha = np.hstack((new_alpha[:,0].reshape(len(new_alpha),1),new_alpha,new_alpha[:,-1].reshape(len(new_alpha),1)))
    my_shape = np.shape(new_alpha)

    qrates = [str(my_cbar_tick) for my_cbar_tick in bounds]
    norm = mpl.colors.BoundaryNorm(bounds,9)
    fmt = mpl.ticker.FuncFormatter(lambda x, pos: qrates[:][norm(x)])



    im = ax.imshow(z_array, cmap= cmap, norm=norm)#, extent=[0.5,9.5,len(removed_column_list)+0.5,-0.5])

    ax.set_xlim(-0.5,len(test_list) - 0.5)
    ax.set_ylim(len(removed_column_list)-0.5,-0.5)

    ax.set_xticks(range(len(test_list)))
    ax.set_xticklabels(labels=test_list,rotation=45)
    ytick_locs = ([np.mean(np.array(data_df[data_df.my_type == my_type].index)) for my_type in (my_type_list)])
    ytick_sep_locs = ([data_df[data_df.my_type == my_type].index[0]-0.5 for my_type in my_type_list])
    ytick_sep_locs = list(ytick_sep_locs)
    ytick_sep_locs.append(ytick_sep_locs[-1]+1)
    ytick_labels = ([str(my_ytick) for my_ytick in my_type_list ])

    ax.vlines([0.5,1.5,2.5,3.5,4.5,5.5,6.5],-0.5,127.5,linewidth=1)
    ax.vlines(3.5,-0.5,127.5,linewidth=2,color = 'k')
    if len(test_list)>8:
      ax.vlines(7.5,-0.5,127.5,linewidth=2,color = 'k')
    ax.vlines(3.5,-0.5,127.5,linewidth=2,color = 'k')
    ax.hlines(ytick_sep_locs,-0.5,8.5,linewidth=1,color = 'k')

    ax.set_xlabel('All runs') 

    ax.set_ylabel('Variables ordered by type', fontsize=fs) 
    ax.set_yticks(ytick_locs)
    ax.set_yticklabels(ytick_labels)
    ax.set_yticks(ytick_sep_locs,minor = True)
    ax.tick_params(axis='y', which='major',length=0)
    ax.tick_params(axis='y', which='minor',length=5)

    ax.set_aspect(0.1)


    #cbar_kw=dict(ticks=np.arange(0,1), format=fmt)
    cbar = ax.figure.colorbar(im,format = '%.3f')
    cbar.ax.set_ylabel("Mean shift Z-score", rotation=-90, va="bottom",fontsize = 12)

    plt.savefig('../output/mean_shift_z_heatmap.png',dpi = 500)
  return ax

'''
### Example code for heatmap
from plotting import heatmap
import pandas as pd
df = pd.read_csv('../output/ensemble_all.csv')
test_list = ['rh-min-low', 'albice00', 'cpl-bug','nu', 'fma','expand',  'no-opt', 'rand-mt','GNU']
ctrl_label = 'Intel'
compare_label = 'GNU'
removed_column_list = df.drop(axis=1,columns=['nyear','Label','Simulation']).columns
heatmap(df, ctrl_label, compare_label, removed_column_list, test_list = test_list)
'''

def heatmap_2samp_ks(df, ctrl_label, compare_label, compare_simu, test_list, use_year = 90, fs = 12, remove_var_list = None, orderby = 'Type'):
  '''
  ``orderby'' is an input to specify the order of variables in the y-axis. By default the heatmap is ordered by type as specified in CESM_var_list.csv. The other option is to feed-in manually set list of variables. 
  '''
  ### Test all test runs
  if ctrl_label not in np.unique(df.Label) or compare_label not in np.unique(df.Label):
      print('ERROR: Input ensemble label does not exist.')
  else:
    ### Todo: Could be pre-defined. Default input var_df and removed_column_list: Can be turned into function or class.
    var_df = pd.read_csv('../output/CESM_variable_type.csv')
    var_df = var_df.rename(columns = {'type':'my_type'})
    removed_column_list = list(df.drop(axis=1,columns=['nyear','Label','Simulation']).columns)
    ### Construct the list of variables to be removed
    if remove_var_list:
      for my_col in remove_var_list:
        removed_column_list.remove(my_col)
    newdf = pd.DataFrame()
    for my_var in removed_column_list:
      stat_df = stats_table(df,ctrl_label, compare_label, compare_simu, test_list, my_var = my_var, remove_var_list = remove_var_list)
      stat_df['Variable'] = my_var
      newdf = newdf.append(stat_df)

    if compare_label:
      ### Add new column for compare ensemmble
      new_test_list = test_list + [compare_label]
    else:
      ### Use only test runs
      new_test_list = test_list

    

    ### Reformat data table
    data_df = newdf.pivot_table(values = 'KS-pvalue',index = 'Variable',columns='Dist_A')
    
    var_df_removed = var_df[var_df.variable.isin(removed_column_list)].drop(columns=['Unnamed: 0','index'])
    var_df_removed['Variable'] = var_df_removed['variable']
    data_df = pd.merge(data_df,var_df_removed,on = 'Variable')
    data_df = data_df.rename(columns = {'type':'my_type'})
    ### order rows by type
    if orderby == 'Type':
      #my_type_list = ['Flux','Cloud','Aerosol','Precipitation','Height','Temperature','Pressure','Transport','Fraction']
      my_type_list = np.unique(data_df['my_type'])
      mapping = {  my_type_list[i] : i for i in range(0, len(my_type_list) ) }
      #mapping = {'Aerosol':1,'Precipitation':2,'Cloud':3,'Temperature':4,'Pressure':5,'Flux':6,'Fraction':7,'Height':8,'Transport':9}
      data_df['sort_order'] = data_df['my_type'].map(mapping)
    else:
      if isinstance(orderby, list) & len(orderby) == len(data_df.Variable):
        ### Check if input is a list
        mapping = { orderby[i] : i for i in range(0, len(orderby) ) }
        data_df['sort_order'] = data_df['Variable'].map(mapping)
      else:
        print('Error: Input type not list. ')
        


    ### Prepare the data array for KS test
    nvar = len(removed_column_list)
    nmodes = len(new_test_list)

    data = np.zeros((nvar,nmodes))
    ks_array = np.zeros((nvar,nmodes))

    for itest,my_test in enumerate(new_test_list):
      #df_reorder = data_df.copy()
      df_reorder = data_df.sort_values(by = 'sort_order',ascending=True).reset_index()
      ks_array[:,itest] = df_reorder[my_test]
      #my_sig = 'pc_sig'+str(imode+1)


    fig, ax = plt.subplots(figsize=(10,10))
    df_reorder = df_reorder.rename(columns={'index':'old_index'})
    print(df_reorder)

    YlBu = np.array(['#f7f7f7','#ffffd9','#edf8b1','#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8','#0c2c84'])[::-1]
    cmap= colors.ListedColormap(YlBu)
    ### Manually set boundaries for colors
    bounds = [0., 0.001, 0.005, 0.01,0.05,0.1,0.2,0.5,0.9,1.]
    norm = colors.BoundaryNorm(bounds, cmap.N, clip=True)

    new_alpha = ks_array.copy()
    new_alpha = np.vstack((new_alpha[0,:],new_alpha,new_alpha[-1,:]))
    new_alpha = np.hstack((new_alpha[:,0].reshape(len(new_alpha),1),new_alpha,new_alpha[:,-1].reshape(len(new_alpha),1)))
    my_shape = np.shape(new_alpha)

    qrates = [str(my_cbar_tick) for my_cbar_tick in bounds]
    norm = mpl.colors.BoundaryNorm(bounds,9)
    fmt = mpl.ticker.FuncFormatter(lambda x, pos: qrates[:][norm(x)])

    left_coord = 0
    right_coord = len(new_test_list)
    top_coord = 0
    bottom_coord= len(removed_column_list)
    im = ax.imshow(ks_array, cmap= cmap, norm=norm, extent=[left_coord,right_coord,bottom_coord,top_coord],origin='upper')
    #ax.contourf(new_alpha < 0.05, 1, hatches=['//', ''], alpha=0,origin = 'image',extent=[-1.5,8.5,len(removed_column_list)+1.5,-1.5])
    ax.set_xlim(left_coord,right_coord)
    ax.set_ylim(bottom_coord,top_coord)

    ax.set_xticks(0.5*np.ones(len(new_test_list))+range(len(new_test_list)))
    ax.set_xticklabels(labels=new_test_list,rotation=45,fontsize = fs-1)


    ytick_locs = ([np.mean(np.array(df_reorder[df_reorder.my_type == my_type].index)) for my_type in (my_type_list)])
    ytick_sep_locs = ([df_reorder[df_reorder.my_type == my_type].index[0] for my_type in my_type_list])
    ytick_sep_locs = list(ytick_sep_locs)
    ytick_sep_locs.append(ytick_sep_locs[-1]+1)
    ytick_labels = ([str(my_ytick) for my_ytick in my_type_list ])
    print(ytick_locs,ytick_labels)

    ax.vlines(4,top_coord,bottom_coord,linewidth=2,color = 'k')
    ax.vlines(range(len(new_test_list)),top_coord,bottom_coord,linewidth=1,color = 'k')
    if compare_label:
      ax.vlines(right_coord - 1,top_coord,bottom_coord,linewidth=2,color = 'k')
    ax.hlines(ytick_sep_locs,left_coord,right_coord,linewidth=1,color = 'k')
    #ax.grid(color='k', linestyle='-', linewidth=1)


    ax.set_xlabel('All runs',fontsize = fs) 
    ### Need to fix ylabel

    ax.set_ylabel('Variables ordered by type',fontsize = fs-2) ## Zero means largest PC mode
    ax.set_yticks(ytick_locs)
    ax.set_yticklabels(ytick_labels,fontsize = fs-1)
    ax.set_yticks(ytick_sep_locs,minor = True)
    #ax.tick_params(axis='y', which='major',length=0)
    #ax.tick_params(axis='y', which='minor',length=5)

    ax.set_aspect(0.1)


    #cbar_kw=dict(ticks=np.arange(0,1), format=fmt)
    cbar = ax.figure.colorbar(im)
    cbar.ax.set_ylabel("K-S p-value", rotation=-90, va="bottom",fontsize = fs-2)
  plt.savefig('../output/heatmap_2samp_ks.png')
  return ax

def heatmap_pca(df, ctrl_label = 'Intel', use_year = 100, fs = 12, remove_var_list = None, flip_sign = True, mapping = {'Aerosol':1,'Precipitation':2,'Cloud':3,'Temperature':4,'Pressure':5,'Flux':6,'Fraction':7,'Height':8,'Transport':9}):
  '''
  This function produces heatmap of variables ordered by descending order of absolute singular value for each PCA mode. By default, the first 9 modes considered.

  Flip_sign is set to True if the sign convention is corrected to ensure negative and positive singular vectors are not canceled out during averaging. By default is True.  

  mapping is a dictionary that maps the Type of variable to the colors. 
  '''
  ### Test all test runs
  if ctrl_label not in np.unique(df.Label):
      print('ERROR: Input ensemble label does not exist.')
  else:
    ### Todo: Could be pre-defined. Default input var_df and removed_column_list. Same as above.
    var_df = pd.read_csv('../output/CESM_variable_type.csv')
    var_df = var_df.rename(columns = {'type':'my_type'})
    removed_column_list = list(df.drop(axis=1,columns=['nyear','Label','Simulation']).columns)
    ### Construct the list of variables to be removed
    if remove_var_list:
      for my_col in remove_var_list:
        removed_column_list.remove(my_col)


    
    all_PCs, all_EOFs, _,_,_,_ =pca_analysis(df,ctrl_label,remove_var_list,use_year = use_year,use_runs = 112)
    if flip_sign and all_PCs.ndim > 2:
      EOF_array = all_EOFs.copy()
      PC_array = all_PCs.copy()
      idx = 13
      imode_plot = 0
      counter = 0
      print(np.shape(PC_array))
      sim_list_length = np.shape(PC_array)[0]
      n_components = np.shape(PC_array)[2]
      for isimu in range(sim_list_length):    
        for imode in range(n_components):
          pc1 = PC_array[0,:,imode]
          pearsonr,pvalue = st.pearsonr(pc1,PC_array[isimu,:,imode])
          if np.sign(pearsonr) < 0 and imode ==imode_plot:
            counter +=1
          PC_array[isimu,:,imode] = PC_array[isimu,:,imode]*np.sign(pearsonr)
          EOF_array[isimu,imode,:] = EOF_array[isimu,imode,:]*np.sign(pearsonr)
      ### Considers flip sign of PCA modes if specified
      my_array = PC_array 
    else:
      my_array= all_PCs
   
    var_df_removed = var_df[var_df.variable.isin(removed_column_list)].drop(columns=['Unnamed: 0','index'])
    #var_df_removed['Variable'] = var_df_removed['variable']

    my_mode_list = ['pc_mode'+str(imode+1) for imode in range(9)]
    x = range(len(removed_column_list))

    ### Prepare for the heatmap
    newdf = var_df_removed.copy()  

    for imode,my_mode in enumerate(my_mode_list): 
      if my_array.ndim == 3:     
        std_mode = np.nanstd(my_array[:,:,imode],axis=0) 
        pc_mode = np.nanmean(my_array[:,:,imode],axis=0)
        up_quant = np.nanquantile(my_array[:,:,imode],0.75,axis=0)
        lw_quant = np.nanquantile(my_array[:,:,imode],0.25,axis=0)
      elif my_array.ndim == 2:
        #std_mode = np.nanstd(my_array[:,imode],axis=0) 
        pc_mode = my_array[:,imode]
        #up_quant = np.nanquantile(my_array[:,imode],0.75,axis=0)
        #lw_quant = np.nanquantile(my_array[:,imode],0.25,axis=0)
      newdf[my_mode] = abs(pc_mode)
      my_sig = 'pc_sig'+str(imode+1)
      newdf[my_sig] = (up_quant*lw_quant > 0)

    
    newdf['my_color'] = newdf['my_type'].map(mapping)
    my_type_list = [my_type for my_type in mapping]
      
    


    ### Prepare the data array for KS test
    nvar = len(removed_column_list)
    nmodes = len(my_mode_list)

    data = np.zeros((nvar,nmodes))
    alpha_array = np.zeros((nvar,nmodes))

    for imode,my_mode in enumerate(my_mode_list):
      newdf = newdf
      df_reorder = newdf.sort_values(by = my_mode,ascending=False)
      data[:,imode] = df_reorder['my_color']
      my_sig = 'pc_sig'+str(imode+1)

      alpha_array[:,imode] = df_reorder[my_sig]


    #df_reorder = df_reorder.rename(columns={'index':'old_index'})

    
    
    fig, ax = plt.subplots(figsize=(8,8))
    
    discrete_nine_colors = ['#fee0b6','#2166ac','#67a9cf','#b2182b','#984ea3','#fc8d59','#999999','#e0e0e0','#7fbf7b']
    cmap= colors.ListedColormap(discrete_nine_colors)
    bounds = np.arange(0.,10.,1)
    norm = colors.BoundaryNorm(bounds, cmap.N, clip=True)


    new_alpha = alpha_array.copy()
    new_alpha = np.vstack((new_alpha[0,:],new_alpha,new_alpha[-1,:]))
    new_alpha = np.hstack((new_alpha[:,0].reshape(len(new_alpha),1),new_alpha,new_alpha[:,-1].reshape(len(new_alpha),1)))
    my_shape = np.shape(new_alpha)

    qrates = np.array(my_type_list)
    norm = mpl.colors.BoundaryNorm(np.linspace(1, 10, 10), 9)
    fmt = mpl.ticker.FuncFormatter(lambda x, pos: qrates[:][norm(x)])



    im = ax.imshow(data, cmap=cmap,  norm = norm)#, extent=[0.5,9.5,len(removed_column_list)+0.5,-0.5])
    ax.contourf(new_alpha, 1, hatches=['//', ''], alpha=0,origin = 'image',extent=[-1.5,9.5,len(removed_column_list)+1.5,-1.5])
    ax.set_xlim(-0.5,8.5)
    ax.set_ylim(len(removed_column_list)-0.5,-0.5)

    ax.set_xticks(range(9))
    ax.set_xticklabels(labels=[str(imode+1) for imode in range(9)])

    ax.set_xlabel('PC modes', fontsize=fs) 

    ax.set_ylabel('Descending order of PC mode coefficients', fontsize=fs) ## Zero means largest PC mode
    ax.set_aspect(0.1)


    cbar_kw=dict(ticks=np.arange(0.5, 10.5,1), format=fmt,norm=norm)
    cbar = ax.figure.colorbar(im,  **cbar_kw)
    cbar.ax.set_ylabel("Type", rotation=-90, va="bottom",fontsize = fs)


    plt.savefig('../output/heatmap_pca_nine_modes.png')
  return ax

### autocorr_plotting
def autocorr_plotting(df, ctrl_label = 'Intel', var_list ='all', n_lags = 20):
  '''
  df is the full input DataFrame. 
  ctrl_label is the ensemble name used as control set. 
  var_list is the list of variables to be used. If the input is a string 'all', then it uses all non-label columns as the input; otherwise use the list feeded. 
  n_lag is the maximum units of lags considered. In this case is 20 years. 
  '''
  
  ctrl_df = df[df.Label == ctrl_label]
  sim_list = np.unique(ctrl_df.Simulation)
  if var_list == 'all':
    col_df = newdf.drop(axis=1,columns=['nyear','Label','Simulation'])
    var_list = col_df.columns
  autocorr_array = np.zeros((n_lags,len(var_list),len(sim_list)))
  autocorr_avg = np.zeros((n_lags,len(var_list)))


  for n1,sim in enumerate(sim_list):
    for n2,var in enumerate(var_list):
      curr_var = np.array(ctrl_df[(ctrl_df.Simulation == sim)][var])    
      autocorrelation_tmp = st.autocorr(curr_var,n_lags)
      autocorr_array_Intel[:,n2,n1] = autocorrelation_tmp
  

  for n1 in range(len(var_list)):
    for n2 in range(n_lags):
      autocorr_avg[n2,n1] = np.mean(autocorr_array[n2,n1,:])

  fig, ax = plt.subplot()
  ax.plot(autocorrelations_Intel_ave[:,:])
  return ax

### correlation matrix
def correlation_matrix(df, ctrl_label = 'Intel', test_label = None, sort_order = None, remove_var_list = None, fs = 14):
  import seaborn as sns
  from stats import pca_analysis, sorted_variables_pca
  '''
  df is the full DataFrame. 
  ctrl_label is the ensemble name to be plotted, default is 'Intel'. Could be 'Test' but then has to be fed a test_label. 
  '''
  if ctrl_label == 'Intel':
    single_run = df[(df.Label == 'Intel')]
    my_title = 'Intel ensemble'
  elif ctrl_label == 'GNU':
    single_run = df[(df.Label == 'GNU') ]
    my_title = 'GNU ensemble'
  elif ctrl_label == 'Test' and test_label:
    my_test = test_label
    single_run = df[(df.Simulation == my_test)]
    my_title = my_test
  else:
    print('Error: unmatched input label. ')

  single_run_no_lab = single_run.drop(['nyear','Label', 'Simulation'], axis=1)
  single_run_no_lab = single_run_no_lab.drop(remove_var_list, axis=1)

  if sort_order == 'Type':
    var_df = pd.read_csv('../data/Ensemble_project/CESM_variable_type.csv')

    var_df['my_type'] = var_df['type']
    new_var_df = var_df.copy()
    new_var_df['sort_order'] = 8
    new_var_df['sort_order'].loc[new_var_df.my_type == 'Flux'] = 0
    move_clear_sky = False ### Set to True if want to move clear sky variable to one cluster
    if move_clear_sky:
      new_var_df['sort_order'].loc[(new_var_df.my_type == 'Flux') & (new_var_df.variable.str.contains('C'))] = 0.5
    new_var_df['sort_order'].loc[new_var_df.my_type == 'Cloud'] = 1
    new_var_df['sort_order'].loc[new_var_df.my_type == 'Aerosol'] = 2
    new_var_df['sort_order'].loc[new_var_df.my_type == 'Precipitation'] = 3
    new_var_df['sort_order'].loc[new_var_df.my_type == 'Height'] = 4
    new_var_df['sort_order'].loc[new_var_df.my_type == 'Temperature'] = 5
    new_var_df['sort_order'].loc[new_var_df.my_type == 'Pressure'] = 6
    new_var_df['sort_order'].loc[new_var_df.my_type == 'Transport'] = 7
    new_var_df = new_var_df[~new_var_df.variable.isin(remove_var_list)]

    new_var_df = new_var_df.sort_values(by=['sort_order','my_type'])
    new_var_df = new_var_df.reset_index()
    sorted_column_list = list(new_var_df.variable)
    sorted_type_list = new_var_df.my_type

    ### if sort by type:
    hline_list = [0]
    for idx in range(len(sorted_type_list)):
      if idx > 0 and sorted_type_list[idx] != sorted_type_list[idx-1]:
        hline_list.append(idx)
  elif sort_order == 'PCA':
        
    df = pd.read_csv('../output/ensemble_all_updated.csv').drop(axis=1,columns=['Unnamed: 0'])
    intel_df = df[df.Label == ctrl_label]
    remove_pca_var_list =  ['H2O2_SRF','DTWR_H2O2','OCNFRAC','ICEFRAC','LANDFRAC','PHIS','SOLIN','EMISCLD','SNOWHLND','SNOWHICE'] 

    all_PCs, all_EOFs, _,_,_,_ =pca_analysis(df,'Intel',remove_pca_var_list,use_year = 100)

    sorted_column_list,nvar_each_mode  = sorted_variables_pca(df, all_PCs, remove_pca_var_list)
    ###hline_list if we sort by PCA
    
    hline_list = np.arange(nvar_each_mode,len(single_run_no_lab) - len(remove_pca_var_list),nvar_each_mode)

    
  newdf = single_run_no_lab[sorted_column_list]
  corrMatrix = newdf.corr()
  from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
  from mpl_toolkits.axes_grid1.colorbar import colorbar

  if ctrl_label != 'Intel':
    intel_corrMatrix = np.load('../output/correlation_matrix_intel001.npy')
    
  fig = plt.figure(1,figsize=(16,14)) #16,14
  ### This should have been fixed, so can be removed
  if my_title == '0nu':
    my_title = 'nu'
  fig.suptitle(my_title)
  #cax = fig.add_axes([0.94,0.2,0.05,0.6])
  #grid_kws = {"width_ratios": (.9, .05), "hspace": 0.2}
  #f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)

  triangle = False
  if triangle:
    mask = np.triu(np.ones_like(corrMatrix, dtype=np.bool))
    if my_ens == 'Intel':
      ax = sns.heatmap(corrMatrix, mask = mask, annot=False,cmap = 'RdBu_r',vmin = -1, vmax=1.0, square = True,cbar=False)
    else:
      # Lower triangle
      ax = sns.heatmap(corrMatrix - intel_corrMatrix, mask = mask, annot=False,cmap = 'RdBu_r',vmin = -0.5, vmax=0.5, square = True,cbar=False)
      # Upper triangle
      #ax = sns.heatmap(corrMatrix, mask = ~mask, annot=False,cmap = 'RdBu_r',vmin = -1, vmax=1.0, square = True,cbar=False)

  else:
    if ctrl_label == 'Intel':
      ax = sns.heatmap(corrMatrix, annot=False,cmap = 'RdBu_r',vmin = -1, vmax=1.0, square = True,cbar=False)
    else:
      ax = sns.heatmap(corrMatrix - intel_corrMatrix, annot=False,cmap = 'RdBu_r',vmin = -0.5, vmax=0.5, square = True,cbar=False)


  
  ax.hlines(hline_list, *ax.get_xlim())
  ax.vlines(hline_list, *ax.get_ylim())
  #ax.set_ylim(127,-1)
  #ax.tick_params(axis = 'x',top=True,labeltop = True,rotation = 90)
  yaxis = ax.get_yaxis()
  my_odd_locs = yaxis.get_ticklocs()
  my_odd_ticks = [str(sorted_column_list[idx]) for idx in np.arange(0,len(sorted_column_list),2)]
  my_even_locs = yaxis.get_ticklocs()+1
  my_even_ticks = [str(sorted_column_list[idx]) for idx in np.arange(1,len(sorted_column_list),2)]
  ax.set_yticks(my_even_locs[:]) ### Used to have -1
  ax.set_yticklabels(my_even_ticks)


  ### Use different top and bottom axis
  secax = ax.secondary_xaxis('top')
  secax.set_xticks(my_even_locs[:])### Used to have -1
  secax.set_xticklabels(my_even_ticks)
  secax.tick_params(axis = 'x',top=True,labeltop = True,rotation = 90)

  #secax = ax.secondary_xaxis('bottom')
  #secax.set_xticks(my_even_locs[:-1])
  #secax.set_xticklabels(my_even_ticks)
  #secax.tick_params(axis = 'x',bottom=True,labelbottom = True,rotation = 90,pad = 80)


  ### Use different top and bottom axis
  secay = ax.secondary_yaxis('right')
  secay.set_yticks(my_odd_locs)
  secay.set_yticklabels(my_odd_ticks)
  secay.tick_params(axis = 'y',right=True,labelright = True,rotation = 0)

  ### Get colorbar position

  if ctrl_label == 'Intel':
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes('bottom', size = '4%', pad = '15%')
    cbar = colorbar(ax.get_children()[0], cax = cax, orientation = 'horizontal',ticks=np.linspace(-1,1,5))
    cbar.ax.set_xlabel('Corr. Coef.', rotation=0,fontsize=fs)
  else:
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes('bottom', size = '4%', pad = '15%')
    cbar = colorbar(ax.get_children()[0], cax = cax, orientation = 'horizontal',ticks=np.linspace(-1,1,5))
    cbar.set_label_text('Corr. Coef. Diff.',fontsize = 12)

  if ctrl_label == 'Intel' and sort_order == 'Type':
    np.save('../output/correlation_matrix_intel001.npy',corrMatrix)

  
  plt.savefig(f'../output/correlationmatrix_{ctrl_label}_{sort_order.lower()}.png')
  
  
  return ax
      