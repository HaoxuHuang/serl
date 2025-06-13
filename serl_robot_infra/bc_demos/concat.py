import pickle
import os
import numpy as np

dirs = os.listdir('.')
cur = []
for dir in dirs:
    if dir.startswith('basketball_5_demos_2025-06-13'):
        cur.append(dir)

cur=sorted(cur)
print(cur)    

pkls=[]
for d in cur:
    with open(d,'rb') as f:
        pkls.append(pickle.load(f)) 

# for i in range(2):
#     for t in pkls[i]:
#         if t['dones']:
#             t['rewards']=t['rewards']+18

for i in range(len(pkls)):
    for t in pkls[i]:
        t['observations']=t['observations']['state']
        t['next_observations']=t['next_observations']['state']
        # t['rewards']=np.array(t['rewards'],dtype=np.float32)

out=[]
for t in pkls:
    out=out+t

with open('basketball_5x6_demos_2025-06-13.pkl','wb') as f:
    pickle.dump(out,f)