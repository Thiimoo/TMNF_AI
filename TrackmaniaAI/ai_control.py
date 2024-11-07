import cv2
import numpy as np
from PIL import ImageGrab
import pydirectinput
import time

def capture_screen(bbox=(0, 0, 800, 600)):
    """Capture a screenshot of the specified bounding box."""
    screen = np.array(ImageGrab.grab(bbox=bbox))
    return screen

def steer_left(duration=0.1):
    """Simulate pressing the 'A' key to steer left."""
    pydirectinput.keyDown("a")
    time.sleep(duration)
    pydirectinput.keyUp("a")

def steer_right(duration=0.1):
    """Simulate pressing the 'D' key to steer right."""
    pydirectinput.keyDown("d")
    time.sleep(duration)
    pydirectinput.keyUp("d")

def accelerate(duration=0.1):
    """Simulate pressing the 'W' key to accelerate."""
    pydirectinput.keyDown("w")
    time.sleep(duration)
    pydirectinput.keyUp("w")

def release_all_keys():
    """Release all control keys to reset the state."""
    pydirectinput.keyUp("w")
    pydirectinput.keyUp("a")
    pydirectinput.keyUp("d")
    pydirectinput.keyUp("space")

def reset_game():
    """Simulate pressing the 'Delete' key to reset the game."""
    pydirectinput.press("delete")
