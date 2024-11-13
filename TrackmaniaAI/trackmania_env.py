import time
import gym
from gym import spaces
import numpy as np
import cv2
import pymem
from collections import deque
import TrackmaniaAI.ai_control as ai_control

class TrackmaniaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, frame_stack=4, render=False):
        super(TrackmaniaEnv, self).__init__()

        self.frame_stack = frame_stack
        self.render_mode = render

        self.action_space = spaces.Discrete(3)

        self.observation_shape = (80, 60, 1)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.observation_shape[0],
                                                   self.observation_shape[1],
                                                   self.observation_shape[2] * self.frame_stack),
                                            dtype=np.uint8)

        self.frames = deque(maxlen=self.frame_stack)
        self.pm = None
        self.distance_address = 0x16867760
        self.speed_address = 0x16867764
        self.alt_speed_address = 0x16867764
        self.cpk_address = 0x10092190
        self.alt_cpk_address = 0x10092194
        self._initialize_memory()
        self.previous_speed = 0
        self.previous_distance = None
        self.previous_time = None
        self.previous_cpk = None
        self.previous_checkpoint_time = None

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
        self.previous_cpk = None
        self.frames.clear()
        initial_state = self._get_state()
        for _ in range(self.frame_stack):
            self.frames.append(initial_state)

        return self._get_observation()

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self._perform_action(action)
        time.sleep(0.05)

        state = self._get_state()
        self.frames.append(state)
        observation = self._get_observation()

        reward = self._compute_reward()

        current_distance = self._read_distance()
        self.previous_distance = current_distance

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

        distance_delta = current_distance - self.previous_distance if self.previous_distance is not None else 0.0
        self.previous_distance = current_distance

        reward = 0.0

        if current_speed == 0 and self.previous_speed > 0:
            reward -= 100  # Negative reward for crash

        self.previous_speed = current_speed

        if current_speed < 600:
            reward += current_speed
        else:
            reward -= 50

        if self._check_checkpoint():
            checkpoint_reward = 200.0
            reward += checkpoint_reward

        finish_reached = self._check_finish()
        if finish_reached:
            reward += 1000.0
            self.reset()

        progress_reward = distance_delta * 0.5
        reward += progress_reward

        print(f"Reward: {reward}, Speed: {current_speed:.2f}, Finish: {finish_reached}, Distance: {distance_delta:.2f}")
        self.previous_speed = current_speed
        return reward

    def _check_done(self):
        return self._check_crash()

    def _check_crash(self):
        current_speed = self._read_speed()
        if current_speed == 0 and self.previous_speed > 0:
            return True
        return False

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
            if cpk == 14:
                return True
            return False
        except Exception as e:
            print(f"Error reading finish line from memory: {e}")
            return False

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
