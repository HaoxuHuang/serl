import os
import sys
import pickle

fnames = os.listdir('.')
pkl_names = []
for x in fnames:
    if x.endswith('.pkl'):
        pkl_names.append(x)

ds = []
for x in pkl_names:
    with open(x,'rb') as f:
        ds+=pickle.load(f)


print(ds[0])

for x in ds:
    x['rewards']=float(x['rewards'])

print(ds[0])

with open('data_store_100_20250617', 'wb') as f:
    pickle.dump(ds, f)