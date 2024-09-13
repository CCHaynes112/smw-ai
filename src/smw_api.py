import os
import pydirectinput
import time
import win32con
import win32gui


MARIO_X_POSITION_FILE = "../scripts/MarioData.txt"


class MarioAPI:
    def __init__(self):
        self.initialize_emulator()
        self.mario_x_position = 0
        self.mario_y_position = 0
        self.mario_is_grounded = False
        self.mario_is_running = False
        self.mario_x_velocity = 0
        self.mario_died = False
        self.current_timer = 999  # Default value
        # x1 speed is 0.12. x2 speed is 0.06. x3 speed is 0.04. x4 speed is 0.03.
        self.default_press_duration = 0.04
        pydirectinput.keyDown("a")  # Always run
        pydirectinput.PAUSE = 0.01

    def move_right(self):
        pydirectinput.keyDown("right")
        time.sleep(self.default_press_duration)
        pydirectinput.keyUp("right")

    def move_left(self):
        pydirectinput.keyDown("left")
        time.sleep(self.default_press_duration)
        pydirectinput.keyUp("left")

    def crouch(self):
        pydirectinput.keyDown("down")
        time.sleep(self.default_press_duration)
        pydirectinput.keyUp("down")

    def look_up(self):
        pydirectinput.keyDown("up")
        time.sleep(self.default_press_duration)
        pydirectinput.keyUp("up")

    def jump(self):
        pydirectinput.keyDown("z")
        time.sleep(self.default_press_duration)
        pydirectinput.keyUp("z")

    def spin_jump(self):
        pydirectinput.keyDown("x")
        time.sleep(self.default_press_duration)
        pydirectinput.keyUp("x")

    def stop_run(self):
        pydirectinput.keyUp("a")
        time.sleep(self.default_press_duration)
        pydirectinput.keyDown("a")

    def reset_rom(self):
        pydirectinput.press("f1")

    def initialize_emulator(self):
        # Clear the file content
        with open(MARIO_X_POSITION_FILE, "w") as f:
            f.write("")

        # Bring window to front
        hwnd = win32gui.FindWindow(None, "Super Mario World (USA) [SNES] - BizHawk")  # Get window handle by name
        if hwnd:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)  # Restore the window if minimized
            win32gui.SetForegroundWindow(hwnd)  # Bring window to the foreground

    def get_current_state_from_emulator(self):
        # The file includes the following information:
        # - Mario's X position
        # - Mario's Y position
        # - Mario is grounded (true or false)
        # - Mario is running (true or false)
        # - Mario's X velocity
        # - Mario died (true or false)
        # The file is formatted as: "int,int,bool,bool,int,bool"
        with open(MARIO_X_POSITION_FILE, "r") as f:
            f.seek(0, os.SEEK_END)  # Go to the end of the file
            while True:
                line = f.readline()
                if line:
                    data = line.strip().split(",")
                    self.mario_x_position = int(data[0].strip())
                    self.mario_y_position = int(data[1].strip())
                    self.mario_is_grounded = data[2].strip() == "true"
                    self.mario_is_running = data[3].strip() == "true"
                    self.mario_x_velocity = int(data[4].strip())
                    self.mario_died = data[5].strip() == "true"
                    self.current_timer = int(data[6].strip())
                    break
                else:
                    time.sleep(0.01)
