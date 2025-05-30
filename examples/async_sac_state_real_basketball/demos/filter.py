import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

file_name = "/home/pjlab/serl/examples/async_sac_state_real_basketball/demos/basketball_5x6_demos_2025-05-30"


with open(file_name, "rb") as f:
    d = pickle.load(f)

print(len(d))

e = []
a = []
start = False

for t in d:
    v = np.max(np.abs(t['actions']))
    if v > 0.003:
        start = True
    if v <= 0.999 and v != -1 and start:
        e.append(t)
        a.append(v)
    if t['dones']:
        start = False
        a.append(-0.1)

with open(file_name + '_F', "wb") as f:
    pickle.dump(e, f)

print(len(e))

plt.plot(a)
plt.show()