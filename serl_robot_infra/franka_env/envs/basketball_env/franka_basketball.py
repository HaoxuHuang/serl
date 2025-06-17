import numpy as np
import gym
import time
import requests
import copy
import cv2
import threading
import queue
import matplotlib.pyplot as plt
import pygame
from datetime import datetime
from collections import OrderedDict
from typing import Dict
import logging

from franka_env.utils.rotations import euler_2_quat
from franka_env.envs.basketball_env.config import BasketballEnvConfig


class ImageDisplayer(threading.Thread):
    def __init__(self, queue, caption="Camera View"):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.caption = caption
 
    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            # initialize pygame on first image
            if not hasattr(self, "_pygame_inited"):
                pygame.init()
                h, w = img_array.shape[:2]
                self.screen = pygame.display.set_mode((w, h))
                pygame.display.set_caption(self.caption)
                self.clock = pygame.time.Clock()
                self._pygame_inited = True

            # handle pygame events (e.g. window close)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # pygame expects (width, height, channels) ordering
            surface = pygame.surfarray.make_surface(img_array.swapaxes(0, 1))

            # draw and update
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(10)


def print_green(x):
    return print("\033[92m{}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m{}\033[00m".format(x))


def print_red(x):
    return print("\033[91m{}\033[00m".format(x))


class FrankaBasketball(gym.Env):
    def __init__(self,
                 angle_penalty: float = 0.00001,
                 energy_penalty: float = 0.0001,
                 hz=50,
                 fake_env=False,
                 save_video=False,
                 config: BasketballEnvConfig = BasketballEnvConfig(),
                 max_episode_length=1000,
                 calibration_pos= [
                    [(179.4, 292.4), (0, 0)],  # center
                    [(140.8, 256.7), (-3, 0)],  # up
                    [(226.6, 333.8), (3, 0)],  # down
                    [(211.4, 225.4), (0, -3)],  # left
                    [(150.5, 353.5), (0, 3)],  # right
                 ],
                 use_camera=False,
                 **kwargs):
        self.camera_lock = threading.Lock()
        self.camera_running = False
        self.camera_loop_thread = None
        self.camera_reward = None
        self.grounded = False
        self.ball_pos = []
        self.second_derivative_range = kwargs.pop(
            "second_derivative_range", (9, 20, 100))
        self.trusted_region = kwargs.pop(
            "trusted_region", ((0, 0), (480, 640)))
        self.calibration_pos =calibration_pos
        self.debug = kwargs.pop("debug", False)
        self.record_length = kwargs.pop("record_length", 30)
        self.context_length = kwargs.pop("context_length", 9)
        self.target_position = np.array(kwargs.pop(
            "target_position", (0, 0)))
        self.rec = []
        self.rec_detection = []
        self.rec_hit = []
        self.angle_penalty = angle_penalty
        self.energy_penalty = energy_penalty
        # if self.debug:
        
        self.image_display = queue.Queue()
        self.image_displayer = ImageDisplayer(self.image_display)
        self.image_displayer.start()

        # safety limit
        self.joint_bounding_box_low = [-2.79, -1.66, -2.79, -2.8, -2.79, 0.1, -2.79]
        self.joint_bounding_box_high = [
            2.79, 1.18, 2.79, -1.05, 2.79, 3.65, 2.79]

        self.action_scale = config.ACTION_SCALE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.url = config.SERVER_URL
        self.config = config
        self.max_episode_length = max_episode_length

        # convert last 3 elements from euler to quat, from size (6,) to (7,)
        # self.resetpos = np.concatenate(
        #     [config.RESET_POSE[:3], euler_2_quat(config.RESET_POSE[3:])]
        # )

        self.currpos = np.zeros((7,), dtype=np.float32)  # tcp pose in xyz + quat
        self.currvel = np.zeros((6,), dtype=np.float32)
        self.q = np.zeros((7,), dtype=np.float32)
        self.dq = np.zeros((7,), dtype=np.float32)
        self.currforce = np.zeros((3,), dtype=np.float32)
        self.currtorque = np.zeros((3,), dtype=np.float32)

        self.curr_gripper_pos = 0
        self.gripper_binary_state = 0  # 0 for open, 1 for closed
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        self.joint_reset_cycle = 200  # reset the robot joint every 200 cycles

        if save_video:
            print("Saving videos!")
        self.save_video = save_video
        self.recording_frames = []

        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -0.4,
            np.ones((7,), dtype=np.float32) *  0.4,
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        "joint_pos": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "joint_vel": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                    }
                ),
            }
        )
        # self.observation_space = gym.spaces.Dict(
        #     {
        #         "tcp_pose": gym.spaces.Box(
        #             -np.inf, np.inf, shape=(7,)
        #         ),  # xyz + quat
        #         "joint_pos": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
        #         "joint_vel": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),     
        #     }
        # )
        self.cycle_count = 0

        if fake_env:
            return

        self.use_camera = use_camera
        if use_camera:
            self.init_cameras()

        print("Initialized Franka")

    def compute_reward(self, action, obs):
        """
        Reward from camera image.
        Alignment camera frequency 30Hz with policy frequency 50Hz.
        Individual process for camera. Communicate with shared memory or network or whatever.
        """
        if obs["state"]["tcp_pose"][0]<0.2:
            print_red("X OUT OF RANGE")
            return -10
        if obs["state"]["tcp_pose"][1]<-0.36 or obs["state"]["tcp_pose"][1]>0.36:
            print_red("Y OUT OF RANGE")
            return -10
        if obs["state"]["tcp_pose"][2]<0 or obs["state"]["tcp_pose"][2]>0.7:
            print_red("Z OUT OF RANGE")
            return -10
        # if obs["state"]["joint_pos"]
        pos_rew = 0.0
        with self.camera_lock:
            if self.camera_reward is not None:
                pos_rew = self.camera_reward
                self.camera_reward = None
        angle_rew = 0.0
        energy_rew = -self.energy_penalty * (np.linalg.norm(action) ** 2)
        return pos_rew + angle_rew + energy_rew

    def go_to_rest(self, joint_reset=False):
        # stop
        self._update_currpos()
        self._send_joint_command(self.q)
        time.sleep(0.5)

        print("JOINT RESET")
        requests.post(self.url + "jointreset")
        time.sleep(0.5)

        # reset
        input("Press enter when you finish picking up the ball and reset joints...")

    def reset(self, joint_reset=False, **kwargs):
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        self.go_to_rest(joint_reset=joint_reset)
        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()

        with self.camera_lock:
            self.grounded = False
            self.camera_reward = None
            self.recording_frames.clear()
            print('Camera reset.')
        return obs, {}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                video_writer = cv2.VideoWriter(
                    f'./videos/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4',
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10,
                    self.recording_frames[0].shape[:2][::-1],
                )
                for frame in self.recording_frames:
                    video_writer.write(frame)
                video_writer.release()
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def capture_ball_pos(self, frame):
        """ 
        Capture ball position from camera image.

        Detect the orange ball in the image and compute its position, assuming the ball is on the ground.
        """
        vis = []
        # vis.append(frame)
        # plt.imshow(frame)
        # plt.show()

        # 1) Smooth & convert to HSV
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # vis.append(hsv)
        # plt.imshow(hsv)
        # plt.show()

        # 2) Threshold for “orange” in HSV space
        #    (tweak these ranges if your lighting/background differs)
        lower_orange = np.array([10, 100, 150])
        upper_orange = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        mask = np.logical_and(mask, self.region_mask)
        # vis.append(self.region_mask)
        # plt.imshow(self.region_mask)
        # plt.show()

        # 3) Morphological open+close to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = np.array(mask * 255, dtype=np.uint8)
        mask = cv2.erode(mask, dst=None, kernel=kernel, iterations=1)
        mask = cv2.dilate(mask, dst=None, kernel=kernel, iterations=1)
        # vis.append(mask)
        # plt.imshow(mask)
        # plt.show()

        # plt.figure(figsize=(12,12))
        # plt.subplot(2,2,1)
        # plt.imshow(vis[0])
        # plt.subplot(2,2,2)
        # plt.imshow(vis[1])
        # plt.subplot(2,2,3)
        # plt.imshow(vis[2])
        # plt.subplot(2,2,4)
        # plt.imshow(vis[3])
        # plt.tight_layout()
        # plt.show()

        # self.rec.append(mask)

        # 4) Find contours in the mask
        cnts = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        center = None
        if cnts:
            # 5) Pick the largest contour by area
            c = max(cnts, key=cv2.contourArea)
            # 6) Compute the centroid via moments (or use minEnclosingCircle)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m01"] / M["m00"])
                cy = int(M["m10"] / M["m00"])
                # 7) Check if the centroid is within the trusted region
                if (self.trusted_region[0][0] <= cx <= self.trusted_region[1][0] and
                        self.trusted_region[0][1] <= cy <= self.trusted_region[1][1]):
                    # 8) If so, use it as the ball position
                    center = np.array([cx, cy])

        self.rec.append(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
        if  self.image_display.empty():
            self.image_display.put(self.rec[-1])
            self.image_display.put(mask)
        if len(self.rec) > self.record_length:
            self.rec.pop(0)
        self.rec_detection.append(mask)
        if len(self.rec_detection) > self.record_length:
            self.rec_detection.pop(0)
        self.recording_frames.append(mask)

        # 9) If no valid contour found, fallback to last known position (or [0,0])
        if center is None:
            self.ball_pos = []
            return None, "no ball detected"

        # 10) Append & keep only the most recent context_length positions
        self.ball_pos.append(center)
        if len(self.ball_pos) > self.context_length:
            self.ball_pos.pop(0)

        return center, "ball detected"

    def hit_ground(self):
        """
        Check if the ball hit the ground at the center of ball_pos list.

        ball_pos is a list of positions. Smooth it with a Gaussian filter. Compute the second derivative of the ball position. Apply a threshold and non-maximum suppression.
        """
        if len(self.ball_pos) < self.context_length:
            return None, "not enough data"

        arr = np.array(self.ball_pos)  # shape (context_length, 2)

        # Smooth each coordinate with Savitzky‐Golay
        from scipy.signal import savgol_filter
        sx = savgol_filter(arr[:, 0], window_length=5, polyorder=2)
        sy = savgol_filter(arr[:, 1], window_length=5, polyorder=2)
        ball_smoothed = np.stack([sx, sy], axis=1)  # shape (context_length,2)

        # First derivative (velocity), then second derivative (acc)
        vel = np.gradient(ball_smoothed, axis=0)
        acc = np.gradient(vel, axis=0)               # shape (context_length,2)

        # Take the center index (self.context_length//2)
        center_acc = acc[self.context_length // 2]
        mag = np.linalg.norm(center_acc)
        if self.debug:
            print(f'[{len(self.rec)}] Acceleration: ' + str(mag))

        # Threshold on magnitude
        if mag < self.second_derivative_range[0] or mag > self.second_derivative_range[2]:
            return None, "out of range"

        # # Non‐max suppression against neighbors (self.context_length//2-1 and self.context_length//2+1)
        if mag < self.second_derivative_range[1]:
            neighbor_mags = [np.linalg.norm(
                acc[self.context_length // 2 - 1]), np.linalg.norm(acc[self.context_length // 2 + 1])]
            if mag < max(neighbor_mags):
                return None, "not maximum"

        # Calibrate the position
        center = self.ball_pos[self.context_length // 2]
        center = np.array(list(center) + [1])
        center = np.dot(self.calibration_matrix, center)
        center = np.array(
            [center[0] / center[2], center[1] / center[2]]) - self.target_position

        if not self.grounded:
            import copy
            self.rec_hit = copy.deepcopy(self.rec[-self.context_length:])
        # if self.debug:
        # self.image_display.put(self.rec_hit[self.context_length // 2])
        return center, "hit ground"

    def camera_loop(self):
        """
        Camera loop for basketball env.
        """
        while True:
            # import time
            # st=time.time()
            ret, frame = self.cap.read()
            if not ret:
                print_red("Failed to read frame from camera.")
                break

            capture_pos, info = self.capture_ball_pos(frame)
            if self.debug:
                if capture_pos is not None:
                    center = capture_pos
                    center = np.array(list(center) + [1])
                    center = np.dot(self.calibration_matrix, center)
                    center = np.array(
                        [center[0] / center[2], center[1] / center[2]]) - self.target_position
                    print_yellow(
                        f'[{len(self.rec)}] ' + str(capture_pos) + ' / ' + str(center) + ' ' + str(info))
                else:
                    print_yellow(f'[{len(self.rec)}] ' +
                                 str(capture_pos) + ' ' + str(info))

            contact_pos, info = self.hit_ground()
            if contact_pos is not None:
                with self.camera_lock:
                    if not self.grounded:
                        print_green(
                            f"[{len(self.rec)}] Ball hit the ground, contact position: {contact_pos}")
                        self.grounded = True
                        self.camera_reward = 20 - np.linalg.norm(contact_pos)
                        print_green(f'Reward: {self.camera_reward}')
                    else:
                        print(
                            f"[{len(self.rec)}] Ball hit the ground, contact position: {contact_pos}")

            with self.camera_lock:
                if not self.camera_running:
                    print_yellow("Camera loop stopped.")
                    break
            # et=time.time()
            # print('Time elapsed:',et-st)

    def init_cameras(self):
        """
        USB Camera can be accessed through v4l device.
        """
        if self.camera_loop_thread is not None:
            self.close_cameras()

        self.cap = cv2.VideoCapture(
            '/dev/v4l/by-id/usb-UGREEN_Camera_UGREEN_Camera_SN0001-video-index0', cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open camera.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.region_mask = np.  zeros([480, 640], dtype=np.bool_)
        self.region_mask[self.trusted_region[0][0]:self.trusted_region[1]
                         [0], self.trusted_region[0][1]:self.trusted_region[1][1]] = True
        # print(self.region_mask.shape)
        # plt.imshow(self.trusted_region*255, vmin=0, vmax=255, cmap='gray')
        # plt.show()

        self.calibration_matrix = None
        if self.calibration_pos is not None:
            p0 = np.array([p[0] for p in self.calibration_pos])
            p1 = np.array([p[1] for p in self.calibration_pos])
            self.calibration_matrix = cv2.findHomography(p0, p1)[0]
        if self.calibration_matrix is None:
            # self.calibration_matrix = np.array([[0.1,0,0],[0,0.1,0],[0,0,1]])
            self.calibration_matrix = np.eye(3)
        print('Calibration')
        print(self.calibration_matrix)

        with self.camera_lock:
            self.camera_running = True
        self.camera_loop_thread = threading.Thread(target=self.camera_loop)
        self.camera_loop_thread.start()
        print_green("Camera initialized.")

    def close_cameras(self):
        with self.camera_lock:
            self.camera_running = False
        self.camera_loop_thread.join()
        self.cap.release()
        self.camera_loop_thread = None
        print_green("Camera closed.")

    def close(self):
        if self.use_camera:
            self.close_cameras()
        super().close()

    def _recover(self):
        """Internal function to recover the robot from error state."""
        requests.post(self.url + "clearerr")

    def clip_safety_box(self, pose):
        """
        Clip the joint pose to be in safety range.
        """
        pose = np.clip(
            pose, self.joint_bounding_box_low, self.joint_bounding_box_high
        )
        return pose

    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        ps = requests.post(self.url + "getstate").json()
        self.currpos[:] = np.array(ps["pose"], dtype=np.float32)
        self.currvel[:] = np.array(ps["vel"], dtype=np.float32)

        self.currforce[:] = np.array(ps["force"], dtype=np.float32)
        self.currtorque[:] = np.array(ps["torque"], dtype=np.float32)

        self.q[:] = np.array(ps["q"], dtype=np.float32)
        self.dq[:] = np.array(ps["dq"], dtype=np.float32)

        # self.curr_gripper_pos = np.array(ps["gripper_pos"])

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action_norm = np.linalg.norm(action)
        action_max = np.max(np.abs(action))

        self.nextjointpos = self.q.copy()
        self.nextjointpos += action * self.action_scale

        self._send_joint_command(self.clip_safety_box(self.nextjointpos))

        self.curr_path_length += 1
        dt = time.time() - start_time
        # print("Step delta time: ", dt)
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        # done = self.curr_path_length >= self.max_episode_length or reward > 0
        done = self.curr_path_length >= self.max_episode_length
        with self.camera_lock:
            done = done or self.grounded
        if reward < -9:
            done = True
        return ob, reward, done, False, {'action_norm': action_norm, 'action_max': action_max}

    def _send_joint_command(self, joint: np.ndarray):
        """Internal function to send joint command to the robot."""
        self._recover()
        arr = np.array(joint).astype(np.float32)
        data = {"arr": arr.tolist()}
        requests.post(self.url + "pose", json=data)

    def _get_obs(self) -> dict:
        state_observation = {
            "joint_pos": self.q,
            "joint_vel": self.dq,
            "tcp_pose": self.currpos,
        }
        return copy.deepcopy(dict(state=state_observation))
