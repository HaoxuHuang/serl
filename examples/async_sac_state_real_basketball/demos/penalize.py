input_file = "/home/pjlab/serl/examples/async_sac_state_real_basketball/demos/data_store_130_20250617_headless"
output_file = "/home/pjlab/serl/examples/async_sac_state_real_basketball/demos/data_store_130_20250617_headless_penalized"
energy_penalty = 1e-2

import pickle
import numpy as np

with open(input_file, "rb") as f:
    d = pickle.load(f)

for t in d:
    t["rewards"] -= energy_penalty * np.linalg.norm(t["actions"])
    if t["observations"][14]<0.27:
        print("X OUT OF RANGE")
        t["rewards"] = -10
        t["dones"] = True
    if t["observations"][15]<-0.36 or t["observations"][15]>0.36:
        print("Y OUT OF RANGE")
        t["rewards"] = -10
        t["dones"] = True
    if t["observations"][16]<0 or t["observations"][16]>0.7:
        print("Z OUT OF RANGE")
        t["rewards"] = -10
        t["dones"] = True

    