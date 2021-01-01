import pandas as pd
import os
os.system('sudo updatedb')
os.system('locate *.git/config > ./allgitconfig.txt')
data = pd.read_csv('allgitconfig.txt',names=['p'])

k = data.iloc[0]
for k in data.values:
    try:
        with open(k[0],'r') as f:
            cfg = f.readlines()
            print(k,cfg[6])
    except:
        pass
