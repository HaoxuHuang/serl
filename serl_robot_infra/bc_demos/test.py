import pickle
import os
import numpy as np
import matplotlib.pyplot as plt


with open('basketball_5x6_demos_2025-05-30.pkl','rb') as f:
    d=pickle.load(f)

action_max = []
b = [[] for _ in range(7)]
o = 0
dis = []
tot = 0

for i, t in enumerate(d):
    v = np.max(np.abs(t["actions"]))
    # v = min(v, 0.1)
    # if v > 0:
    if tot != 18:
        dis.append(np.linalg.norm(t['observations'] - d[o]['observations']))
        if True:
            action_max.append(v)
    if t["dones"]:
        action_max.append(-0.1)
        o = i + 1
        tot += 1
    if tot != 18:
        for i in range(7):
            b[i].append(t['actions'][i])

plt.figure(figsize=(10, 4))
# plt.subplot(1, 2 ,1)
plt.plot(action_max, label="max")
plt.plot(dis, label="dis")
# plt.title("max")
# plt.subplot(1, 2, 2)
# for i in range(7):
#     plt.plot(b[i], label=f"{i}")
# plt.title("[-3]")
plt.tight_layout()
plt.legend()
plt.show()