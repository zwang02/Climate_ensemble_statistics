### Plotting functions
### Plotting functions
import pylab

def qqplot(ctrl_label, compare_ens_df,  my_var, matched = True, test_list = None, dperc = 1):
    '''
    ``ctrl_label'' and ``compare_label'' are the names of the control ensemble 
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