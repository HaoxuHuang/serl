import os
import pickle
import numpy as np

# folder_path = "./basketball_demos_05-26"
# file_list = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
# print(file_list)

# data_list = []
# for file_name in file_list:
#     file_path = os.path.join(folder_path, file_name)
#     with open(file_path, "rb") as f:
#         data = pickle.load(f)
#         print(len(data))
#         data_list.append(data)
# data_list = np.concatenate(data_list, axis=0)

input_path = "./test.pkl"

with open(input_path, "rb") as f:
    data_list = pickle.load(f)
    print(len(data_list))

data = data_list
action_scale = []
for o in data:
    action = o["actions"]
    # if np.max(action) == 0:
    #     continue
    norm = np.linalg.norm(action)
    action_scale.append(norm)
    if o["dones"]:
        action_scale.append(-0.01)

import matplotlib.pyplot as plt

plt.plot(action_scale)
plt.xlabel("Step")
plt.ylabel("Action Norm")
plt.title("Action Norm per Step")
plt.show()
