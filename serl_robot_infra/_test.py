import franka_env
import time
from franka_env.envs.basketball_env.franka_basketball import FrankaBasketball
from franka_env.envs.basketball_env.config import BasketballEnvConfig

debug=False
# debug=True
record_length=600
calib_pos=None
# calib_pos = [
#     [(167.3, 309.8), (0, 0)],  # center
#     [(126.2, 277.3), (-3, 0)],  # up
#     [(220.5, 346.4), (3, 0)],  # down
#     [(194.6, 241.5), (0, -3)],  # left
#     [(138.8, 373.6), (0, 3)],  # right
# ]

config = BasketballEnvConfig()


env = FrankaBasketball(config=config, trusted_region=((0,0),(480,640)), debug=debug, record_length=record_length, )
print('... OK')
env.init_cameras()


def reset_every_5_sec():
    while(True):
        if not env.camera_running:
            break
        env.reset()
        time.sleep(5)


import threading
th=threading.Thread(target=reset_every_5_sec)
th.start()
a=input()
env.close_cameras()
th.join()


import cv2
import imageio
import matplotlib.pyplot as plt
import os
os.makedirs('./temp', exist_ok=True)
for i, frame in enumerate(env.rec_hit):
    plt.imsave (f'./temp/hit_{i}.png', frame)
for i, frame in enumerate(env.rec):
    plt.imsave (f'./temp/{i}.png', frame)
for i, frame in enumerate(env.rec_detection):
    plt.imsave (f'./temp/detection_{i}.png', frame)
print('... Saved recording to ./temp')