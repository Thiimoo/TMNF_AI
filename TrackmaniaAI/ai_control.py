import cv2
import numpy as np
from PIL import ImageGrab
import pydirectinput
import time

# Track the last pressed key
last_key = None

def capture_screen(bbox=(0, 0, 1920, 1080)):
    """Capture a screenshot of the specified bounding box."""
    screen = np.array(ImageGrab.grab(bbox=bbox))
    return screen

def steer_left():
    """Simulate pressing the 'A' key to steer left."""
    global last_key
    if last_key != "a":
        release_all_keys()
    pydirectinput.keyDown("a")
    last_key = "a"

def steer_right():
    """Simulate pressing the 'D' key to steer right."""
    global last_key
    if last_key != "d":
        release_all_keys()
    pydirectinput.keyDown("d")
    last_key = "d"

def accelerate():
    """Simulate pressing the 'W' key to accelerate."""
    global last_key
    if last_key != "w":
        release_all_keys()
    pydirectinput.keyDown("w")
    last_key = "w"

def release_all_keys():
    """Release all control keys to reset the state."""
    pydirectinput.keyUp("w")
    pydirectinput.keyUp("a")
    pydirectinput.keyUp("d")

def reset_game():
    """Simulate pressing the 'Delete' key to reset the game."""
    pydirectinput.press("delete")
