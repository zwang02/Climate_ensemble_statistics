import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Add histogram data
my_var = 'BURDEN2'
intel_df = df[(df.Label == 'Intel')]
my_test_input = 'rh-min-low'
nsteps = 10
fig = go.Figure()

gnu_df = df[df.Label == 'GNU']



def make_interact_dist(my_var,nsteps,my_test = my_test_input):
  init_year = 0
  test_df = df[df.Simulation == my_test]
  mean_init_intel = intel_df[intel_df.nyear == init_year].mean()[my_var]
  std_init_intel = intel_df[intel_df.nyear == init_year].std()[my_var]
  mean_init_gnu = gnu_df[gnu_df.nyear == init_year].mean()[my_var]
  std_init_gnu = gnu_df[gnu_df.nyear == init_year].std()[my_var]
  fig = make_subplots(rows=1, cols=1,shared_xaxes=True)#, subplot_titles = ['Mean %f' % (mean_init_intel) +', Std %f' % (std_init_intel),'Mean %f' % (mean_init_gnu) +', Std %f' % (std_init_gnu)]


  #xmin = np.nanmin((intel_df.min()[my_var],gnu_df.min()[my_var]))
  #xmax = np.nanmax((intel_df.max()[my_var],gnu_df.max()[my_var]))

  xmin = np.nanmin((intel_df.min()[my_var],gnu_df.min()[my_var],test_df.min()[my_var]))
  xmax = np.nanmax((intel_df.max()[my_var],gnu_df.max()[my_var],test_df.max()[my_var]))
  nbins = 40
  bin_size = (xmax - xmin)/nbins
  xbins = np.linspace(xmin,xmax,nbins+1)

  print(xmin,xmax,np.percentile((xmin,xmax),80))
  intel_ymax = []
  gnu_ymax = []
  intel_mean = []
  intel_std = []
  gnu_mean = []
  gnu_std = []
  for step in np.arange(0, nsteps, 1):
    my_year = step
    
    x1 = np.array(intel_df[intel_df.nyear == my_year][my_var])
    x2 = np.array(gnu_df[gnu_df.nyear == my_year][my_var])
    #x3 = np.array(test_df[test_df.nyear == my_year][my_var])    


    # Create distplot with custom bin_size
    #fig = ff.create_distplot(hist_data, group_labels)#, bin_size=[.05, 0.05]
    counts_intel, bins = np.histogram(x1,bins = xbins,density =True)
    counts_gnu, bins = np.histogram(x2,bins = xbins,density =True)
 
    counts_intel = counts_intel/np.sum(counts_intel)
    counts_gnu = counts_gnu/np.sum(counts_gnu)
    intel_ymax.append(counts_intel)
    gnu_ymax.append(counts_intel)
    intel_mean.append(x1.mean())
    intel_std.append(x1.std())
    gnu_mean.append(x2.mean())
    gnu_std.append(x2.std())
  ### Decide ymax
  ymax_intel = np.nanmax(intel_ymax)
  ymax_gnu = np.nanmax(gnu_ymax)   
  ymax = np.nanmax([ymax_intel,ymax_gnu]) 
  print(intel_std)

  for step in np.arange(0, nsteps, 1):
    my_year = step
    
    x1 = np.array(intel_df[intel_df.nyear == my_year][my_var])
    x2 = np.array(gnu_df[gnu_df.nyear == my_year][my_var])
    x3 = np.array(test_df[(df.Simulation == my_test) & (df.nyear == my_year)][my_var])[0]
    fig.add_trace(go.Histogram(visible=False,x=x1, histnorm='probability',name = 'Intel',xbins=dict(
                      start=xmin,
                      end=xmax,
                      size=bin_size), 
                      autobinx=False
                     ),row=1, col=1)
    
    fig.add_trace(go.Histogram(visible=False,x=x2, histnorm='probability',name = 'GNU',xbins=dict(
                      start=xmin,
                      end=xmax,
                      size=bin_size), 
                      autobinx=False
                     ),row=1, col=1)
    #fig.add_annotation(
    #          visible = False,
    #          x=np.percentile((xmin,xmax),20),
    #          y=ymax_intel*0.95,
    #          text="Mean: %f" % ((x1.mean())),
    #          showarrow= False
    #  )
    #fig.add_annotation(
    #          visible = False,
    #          x=np.percentile((xmin,xmax),20),
    #          y=ymax_intel*0.8,
    #          text="Std: %f" % ((x1.std())),
    #          showarrow= False
    #  )

    fig.add_trace(go.Scatter(visible=False,x=[x3, x3], y=[0,ymax], 
      mode="lines",name = my_test, line=dict(color="black",dash='dash') ),row=1,col=1)
    fig.add_trace(go.Scatter(visible=False,x=[x1.mean(), x1.mean()], y=[0,ymax], 
      mode="lines",name = 'Mean Intel',line=dict(color="blue",dash='dash') ),row=1,col= 1)
    fig.add_trace(go.Scatter(visible=False,x=[x2.mean(), x2.mean()], y=[0,ymax], 
      mode="lines",name = 'Mean GNU',line=dict(color="red",dash='dash') ),row=1,col= 1)
        
    fig.update_traces(opacity=0.75)
    

    #fig['layout']['annotations'][0].update(text='Mean %f' % (x1.mean()) +', Std %f' % (x1.std()))
    #fig['layout']['annotations'][1].update(text='Mean %f' % (x2.mean()) +', Std %f' % (x2.std()))
      




  fig.data[0].visible = True
  fig.data[1].visible = True
  fig.data[2].visible = True
  fig.data[3].visible = True
  fig.data[4].visible = True

  #fig.layout.annotations[0].visible = False
  #fig.layout.annotations[1].visible = False

  # Create and add slider
  years = []
  
  for i in range(0, len(fig.data), 5):
      year = dict(
          method="update",
          label = 'Year {}'.format(int(i/5)+1),
          args=[{"visible": [False] * len(fig.data)},
                {"title": "Distribution across ensemble for year " + str(int(i/5)+1)},
                {"annotation": ['Mean %f' % my_mean for my_mean in (intel_mean)]}
                ],  # layout attribute
      )
      year["args"][0]["visible"][i] =True  # Toggle i'th trace to "visible"
      year["args"][0]["visible"][i+1] =True  # Toggle i'th trace to "visible"
      year["args"][0]["visible"][i+2] =True  # Toggle i'th trace to "visible"
      year["args"][0]["visible"][i+3] =True  # Toggle i'th trace to "visible"
      year["args"][0]["visible"][i+4] =True  # Toggle i'th trace to "visible"
      #for new_idx in range(len(fig.layout.annotations)):
      #  fig.layout.annotations[new_idx].visible = False

      years.append(year)

  sliders = [dict(
      active=0,
      currentvalue={"prefix": "This is "},
      pad={"t": nsteps},
      steps=years
  )]

  fig.update_layout(
      sliders=sliders
  )

  fig.update_xaxes(range=[xmin, xmax])
  fig.update_yaxes(range=[0, ymax*1.1],row=1,col=1)
  #fig.update_yaxes(range=[0, ymax_gnu*1.2],row=2,col=1)


  fig.show()

  return fig
my_fig = make_interact_dist(my_var = 'SNOWHLND',nsteps = 10,my_test = 'rh-min-low')
#my_fig.write_html(local_dir+"/interactive_distribution.html")

