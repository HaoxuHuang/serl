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

input_path = r"C:\Users\lcw\Desktop\basketball_demos_05-26\basketball_5_demos_2025-05-26_23-33-14.pkl"
output_path = "./test.pkl"

with open(input_path, "rb") as f:
    data_list = pickle.load(f)
    print(len(data_list))

trajectories = []
trajectory = []
for o in data_list:
    trajectory.append(o)
    if o["dones"]:
        trajectories.append(trajectory)
        trajectory = []

for i in range(len(trajectories)):
    new_data = []
    for o in trajectories[i]:
        if np.max(np.abs(o["actions"])) > 0 or o["dones"]:
            new_data.append(o)
    trajectories[i] = new_data

new_data_list = []
for data in trajectories:
    action_scale = []
    for o in data:
        action = o["actions"]
        norm = np.linalg.norm(action)
        action_scale.append(norm)

    import matplotlib.pyplot as plt

    plt.plot(action_scale)
    plt.xlabel("Step")
    plt.ylabel("Action Norm")
    plt.title("Action Norm per Step")
    plt.show()

    while True:
        user_input = input("New Start: ")
        try:
            x = int(user_input)
        except ValueError:
            print("Not a number")
            continue
        if x < -1 or x >= len(data):
            print(f"Out of range [-1, {len(data)})")
            continue
        break

    if x == -1:
        continue
    new_data_list.append(data[x:])

new_data_list = np.concatenate(new_data_list, axis=0)

with open(output_path, "wb") as f:
    pickle.dump(new_data_list, f)
