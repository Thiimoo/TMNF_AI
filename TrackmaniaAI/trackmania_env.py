import time
import gym
from gym import spaces
import numpy as np
import cv2
import pymem
from collections import deque

from pygame.math import clamp

import ai_control  # Ensure this module is correctly implemented and available

class TrackmaniaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, frame_stack=4, render=False):
        super(TrackmaniaEnv, self).__init__()

        self.frame_stack = frame_stack
        self.render_mode = render

        self.action_space = spaces.Discrete(3)

        # Observation space: stacked grayscale frames
        self.observation_shape = (80, 60, 1)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.observation_shape[0],
                                                   self.observation_shape[1],
                                                   self.observation_shape[2] * self.frame_stack),
                                            dtype=np.uint8)

        self.frames = deque(maxlen=self.frame_stack)
        self.pm = None
        self.distance_address = 0x203BC8B0
        self.speed_address = 0x100B47C0
        self.alt_speed_address = 0x203B80CC
        self.cpk_address = 0x100B47B0
        self.alt_cpk_address = 0x100B47B4
        self._initialize_memory()

        # State variables
        self.previous_distance = None
        self.previous_time = None
        self.previous_cpk = None
        self.previous_checkpoint_time = None  # Store time of last checkpoint


    def _initialize_memory(self):
        try:
            self.pm = pymem.Pymem("TmForever.exe")
            print("Successfully attached to Trackmania process.")
        except pymem.exception.ProcessNotFound:
            print("Trackmania process not found. Ensure the game is running.")
            raise
        except Exception as e:
            print(f"Failed to attach to Trackmania process: {e}")
            raise

    def _read_speed(self):
        try:
            speed = self.pm.read_int(self.speed_address)
            if speed == 0:
                speed = self.pm.read_int(self.alt_speed_address)
            return speed
        except Exception as e:
            print(f"Error reading speed from memory: {e}")
            return 0

    def reset(self):
        print("Resetting environment.")
        ai_control.reset_game()
        time.sleep(2)

        self.previous_distance = self._read_distance()
        self.previous_time = time.time()
        self.previous_cpk = None  # Reset the checkpoint variable for a new trial
        self.frames.clear()
        initial_state = self._get_state()
        for _ in range(self.frame_stack):
            self.frames.append(initial_state)

        return self._get_observation()

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Perform the action (e.g., accelerate, steer, etc.)
        self._perform_action(action)
        time.sleep(0.05)

        # Get the new state (screen capture)
        state = self._get_state()
        self.frames.append(state)
        observation = self._get_observation()

        # Calculate reward
        reward = self._compute_reward()

        # Check if the car is stuck (no progress)
        current_distance = self._read_distance()
        # Update the previous distance for the next step
        self.previous_distance = current_distance

        # Check if the episode is done (e.g., crash or other conditions)
        done = self._check_done()

        if self.render_mode:
            self.render()

        return observation, reward, done, {}

    def _perform_action(self, action):
        action_map = {
            0: ai_control.accelerate,
            1: ai_control.steer_left,
            2: ai_control.steer_right,
        }
        action_func = action_map.get(action)
        try:
            action_func()
        except AttributeError:
            print(f"ai_control does not have method for action: {action}")
            pass

    def _get_state(self):
        try:
            screen = ai_control.capture_screen(bbox=(0, 40, 800, 640))
            gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            resized_screen = cv2.resize(gray_screen, (self.observation_shape[1], self.observation_shape[0]))
            return resized_screen
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return np.zeros((self.observation_shape[0], self.observation_shape[1]), dtype=np.uint8)

    def _get_observation(self):
        stacked_frames = np.stack(self.frames, axis=-1)
        return stacked_frames

    def _read_distance(self):
        try:
            distance = self.pm.read_int(self.distance_address)
            return distance
        except Exception as e:
            print(f"Error reading distance from memory: {e}")
            return 0.0

    def _compute_reward(self):
        current_distance = self._read_distance()
        current_speed = self._read_speed()
        current_time = time.time()
        time_delta = current_time - self.previous_time

        distance_delta = current_distance - self.previous_distance if self.previous_distance is not None else 0.0
        self.previous_distance = current_distance
        self.previous_time = current_time

        is_crashed = self._check_crash()
        passed_checkpoint = self._check_checkpoint()
        is_on_grass = self._check_on_grass()

        reward = 0.0
        if current_speed>600:
            current_speed = 0
        # Speed-based reward, with higher speeds yielding higher rewards
        reward += current_speed



        if passed_checkpoint:
            # Base reward for checkpoint
            checkpoint_reward = 200.0

            # Calculate the time taken since the last checkpoint, if previous_checkpoint_time is set
            if self.previous_checkpoint_time is not None:
                time_diff = current_time - self.previous_checkpoint_time
                checkpoint_speed_reward = max(0.0, 100.0 - time_diff * 10)  # Adjust the factor for desired sensitivity
                checkpoint_reward += checkpoint_speed_reward
            else:
                time_diff = 0  # Set time_diff to 0 if it's the first checkpoint

            reward += checkpoint_reward

            # Update checkpoint time
            self.previous_checkpoint_time = current_time
            print(f"Checkpoint reached! Time: {time_diff:.2f}s, Reward: {checkpoint_reward:.2f}")

        # Check for finish line
        finish_reached = self._check_finish()
        if finish_reached:
            reward += 1000.0
            print("Finish line reached! Reward for completion.")
            self.reset()  # Reset for a new trial

        # Reward for progress, based on distance traveled since last step
        progress_reward = distance_delta * 0.5
        reward += progress_reward
        print(f"Reward: {reward}, Speed: {current_speed:.2f}, Finish: {finish_reached}, Distance: {distance_delta:.2f}, ")
        return reward


    def _check_done(self):
        return self._check_crash()

    def _check_crash(self):
        state = self._get_state()
        mean_pixel = np.mean(state)
        return mean_pixel < 50

    def _check_checkpoint(self):
        try:
            cpk = self.pm.read_int(self.cpk_address)
            if cpk == 0:
                cpk = self.pm.read_int(self.alt_cpk_address)
            if cpk == 0:
                return False
            if self.previous_cpk == cpk:
                self.previous_cpk = cpk
                return False
            else:
                self.previous_cpk = cpk
                return True
        except Exception as e:
            print(f"Error reading checkpoint from memory: {e}")
            return False

    def _check_finish(self):
        try:
            cpk = self.pm.read_int(self.alt_cpk_address)
            if cpk == 14:  # Finish line reached
                return True
            return False
        except Exception as e:
            print(f"Error reading finish line from memory: {e}")
            return False

    def _check_on_grass(self):
        state = self._get_state()
        grass_pixels = np.sum(state < 100)
        return grass_pixels > 500

    def render(self, mode='human'):
        if self.render_mode:
            state = self._get_state()
            cv2.imshow('TrackmaniaEnv', state)
            cv2.waitKey(1)

    def close(self):
        if self.render_mode:
            cv2.destroyAllWindows()
        if self.pm:
            self.pm.close_process()
        print("Environment closed.")
