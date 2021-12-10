import pandas as pd

#cleaning function from Curtis
def approx_z_cleaning(input_df,my_var,threshold):
  #std_temp = np.std(input_variable)
  #mean_temp = np.mean(input_variable)
  low_percentile = input_df[my_var].quantile(0.025)
  high_percentile = input_df[my_var].quantile(0.975)
  variable_mid = input_df[(input_df[my_var] > low_percentile) & (input_df[my_var] < high_percentile)][my_var]
  mean_mid = variable_mid.mean()
  std_mid = variable_mid.std()
  if std_mid != 0:
    reduced_var = (input_df[my_var] - mean_mid)/std_mid
    bad_data_flags = reduced_var[np.abs(reduced_var) > threshold].index
  return bad_data_flags