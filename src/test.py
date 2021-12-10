# In total 114 Intel simulations, throw Intel.115 as it is compromised. Throw Intel.006 due to wrong SNOWHICE. 
import lib as libnc
import time

dropl = libnc.add_droplist('Intel', '115')
dropl = libnc.add_droplist('Intel', '000', dropl)

#create filename based on UNIX time
filename = 'ensemble_all'+str(int(time.time()))

#test new get_data file
print("testing writing to new file: ")
df = libnc.get_data(dir = '../output/', in_file = filename)
print("dropping from droplist")
df = libnc.df_droplist(df, dropl)
print("sucessfully tested create_new file: ", filename)

#test read from written data file
df2 = libnc.get_data()
print("successfully tested read from file in get_data")

print("tests over")