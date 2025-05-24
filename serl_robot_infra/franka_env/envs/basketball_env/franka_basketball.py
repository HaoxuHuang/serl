import numpy as np
import gym
import time
import requests
import copy
import cv2
import threading

from franka_env.envs.franka_env import FrankaEnv
from franka_env.utils.rotations import euler_2_quat
from franka_env.envs.basketball_env.config import BasketballEnvConfig


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


def print_red(x):
    return print("\033[91m {}\033[00m".format(x))


class FrankaBasketball(FrankaEnv):
    def __init__(self, *args, **kwargs):
        self.camera_lock = threading.Lock()
        self.camera_running = False
        self.camera_loop_thread = None
        self.camera_reward = None
        self.ball_pos = []
        self.second_derivative_range = kwargs.get(
            "second_derivative_range", (0.1, 0.5))
        super().__init__(*args, **kwargs)

    def compute_reward(self, obs):
        """
        Reward from camera image.
        Alignment camera frequency 30Hz with policy frequency 50Hz.
        Individual process for camera. Communicate with shared memory or network or whatever.
        """
        with self.camera_lock:
            if self.camera_reward is not None:
                reward = self.camera_reward
                self.camera_reward = None
                return reward
            else:
                return 0.0

    def go_to_rest(self, joint_reset=False):
        # stop
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.5)

        # reset
        super().go_to_rest(joint_reset)

    def capture_ball_pos(self, frame):
        """
        TODO: Capture ball position from camera image.
        """
        # Placeholder for actual ball detection logic
        # For now, we just return a dummy position
        pos = np.array([0.0, 0.0, 0.0])
        self.ball_pos.append(pos)
        if len(self.ball_pos) > 9:
            self.ball_pos.pop(0)

    def hit_ground(self):
        """
        Check if the ball hit the ground at the center of ball_pos list.
        ball_pos is a list of positions. Smooth it with a Gaussian filter. Compute the second derivative of the ball position. Apply a threshold and non-maximum suppression.
        """
        if len(self.ball_pos) < 9:
            return None, 'not enough data'
        ball_pos = np.array(self.ball_pos)
        ball_pos = np.convolve(ball_pos, np.ones(3) / 3, mode='valid')
        ball_pos = np.gradient(ball_pos, axis=0)
        ball_pos = np.gradient(ball_pos, axis=0)
        target_pos = np.abs(ball_pos[4])
        if target_pos < self.second_derivative_range[0] or target_pos > self.second_derivative_range[1]:
            return None, 'out of range'
        if target_pos < np.max(np.abs(ball_pos)):
            return None, 'not maximum'
        return ball_pos[4], 'hit ground'

    def camera_loop(self):
        """
        Camera loop for basketball env.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print_red("Failed to read frame from camera.")
                break

            self.capture_ball_pos(frame)
            contact_pos, info = self.hit_ground()
            if contact_pos is not None:
                print_yellow(
                    f"Ball hit the ground, contact position: {contact_pos}")
                with self.camera_lock:
                    self.camera_reward = 2 - np.linalg.norm(contact_pos)

            with self.camera_lock:
                if not self.camera_running:
                    print_yellow("Camera loop stopped.")
                    break

    def init_cameras(self):
        """
        USB Camera can be accessed through v4l device.
        """
        if self.camera_loop_thread is not None:
            self.close_cameras()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
