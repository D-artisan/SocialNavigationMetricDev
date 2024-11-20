import sys
import os
import cv2
import imageio
import json
import jsbeautifier
import numpy as np
import pygame
import time
import pickle
from PySide2 import QtGui, QtWidgets, QtCore
from mainUI import Ui_MainWindow

sys.path.append(os.path.join(os.path.dirname(__file__), '../SocNavGym'))
import socnavgym
import gym

# Adjusted grid parameters
GRID_CELL_SIZE = 0.05  # Cell size in meters
GRID_WIDTH = int(5 / GRID_CELL_SIZE)  # Should be 100 for a 5m room
GRID_HEIGHT = int(5 / GRID_CELL_SIZE)  # Should be 100 for a 5m room

UPDATE_PERIOD = 0.1  # Update period for data saving

class MainWindow(QtWidgets.QWidget, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.init_joystick()

        self.save_dir = './trajectory_dataset/'
        self.data_file_index = 0
        self.update_data_index()
        self.images_for_video = list()
        self.data = list()

        # Load the environment with the updated configuration file
        self.env = gym.make("SocNavGym-v1", config="socnavgym_conf.yaml")
        self.simulation_time = 0
        self.last_save_simulation_time = -1
        self.n_steps = 0
        self.regenerate()

        self.last_data_update = time.time()

        self.start_saving_button.toggled.connect(self.start_saving)
        self.regenerate_button.clicked.connect(self.regenerate)
        self.quit_button.clicked.connect(self.quit_slot)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.compute)
        self.timer.start(self.env.TIMESTEP * 1000)

    def init_joystick(self):
        pygame.init()
        pygame.joystick.init()
        self.joystick_count = pygame.joystick.get_count()

        if self.joystick_count == 0:
            print("No joystick detected.")
            self.joystick = None
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        axes = self.joystick.get_numaxes()
        print(f"Joystick detected: {self.joystick.get_name()}")
        print(f"Number of axes: {axes}")

        try:
            with open('joystick_calibration.pickle', 'rb') as f:
                self.centre, self.values, self.min_values, self.max_values = pickle.load(f)
                print("Joystick calibration loaded.")
        except:
            print("Joystick calibration file not found. Starting calibration.")
            self.calibrate_joystick()

    def calibrate_joystick(self):
        axes = self.joystick.get_numaxes()
        print("Leave the controller neutral for 3 seconds")
        t = time.time()
        neutral_values = {axis: [] for axis in range(axes)}
        while time.time() - t < 3:
            pygame.event.pump()
            for axis in range(axes):
                neutral_values[axis].append(self.joystick.get_axis(axis))
            time.sleep(0.05)
        self.centre = {}
        for axis in range(axes):
            self.centre[axis] = sum(neutral_values[axis]) / len(neutral_values[axis])
            print(f"Axis {axis} neutral position: {self.centre[axis]}")

        print("Move the joystick around for 5 seconds to reach max and min values for the axes")
        t = time.time()
        self.max_values = {axis: float('-inf') for axis in range(axes)}
        self.min_values = {axis: float('inf') for axis in range(axes)}
        while time.time() - t < 5:
            pygame.event.pump()
            for axis in range(axes):
                value = self.joystick.get_axis(axis) - self.centre[axis]
                self.max_values[axis] = max(self.max_values[axis], value)
                self.min_values[axis] = min(self.min_values[axis], value)
            time.sleep(0.05)
        for axis in self.max_values:
            if abs(self.max_values[axis]) < 0.1:
                self.max_values[axis] = 0.1
            if abs(self.min_values[axis]) < 0.1:
                self.min_values[axis] = -0.1
            print(f"Axis {axis} max: {self.max_values[axis]}, min: {self.min_values[axis]}")

        # Initialize values dictionary
        self.values = {axis: 0.0 for axis in range(axes)}

        # Save calibration data
        with open('joystick_calibration.pickle', 'wb') as f:
            pickle.dump([self.centre, self.values, self.min_values, self.max_values], f)
            print("Joystick calibration saved.")

    def update_data_index(self):
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        file_list = [f for f in os.listdir(self.save_dir) if f.endswith('.mp4')]
        max_index = -1
        for f in file_list:
            ind_str = f.split(self.dataID.text())
            if len(ind_str) > 1:
                ind = int(ind_str[1].split('.')[0])
                if ind > max_index:
                    max_index = ind
        self.data_file_index = max_index + 1

    def normalize_axis(self, axis):
        value = self.values.get(axis, 0)
        # Implement dead zone
        if abs(value) < 0.05:
            return 0.0
        if value >= 0:
            max_value = self.max_values.get(axis, 1)
            return value / max_value if max_value != 0 else 0
        else:
            min_value = self.min_values.get(axis, -1)
            return value / abs(min_value) if min_value != 0 else 0

    def get_robot_movement(self):
        if not self.joystick:
            return [0, 0, 0]
        pygame.event.pump()
        axes = self.joystick.get_numaxes()
        for axis in range(axes):
            self.values[axis] = self.joystick.get_axis(axis) - self.centre.get(axis, 0)

        # Axis mapping for Logitech Extreme 3D Pro
        # Axis 0: Left/Right (X-axis)
        # Axis 1: Forward/Backward (Y-axis)
        # Axis 2: Twist (Rotation)
        vel_x = -self.normalize_axis(1)  # Forward/Backward
        vel_y = -self.normalize_axis(0)  # Left/Right
        vel_a = -self.normalize_axis(2)  # Rotation (Twist)

        if self.env.robot.type == "diff-drive":
            vel_y = 0

        return [vel_x, vel_y, vel_a]

    def compute(self):
        robot_vel = self.get_robot_movement()
        obs, reward, terminated, truncated, info = self.env.step(robot_vel)

        image = self.env.render_without_showing(draw_human_goal=False)
        image = image.astype(np.uint8)

        people, objects, walls, interactions, robot = self.get_data()

        if self.n_steps == 0:
            self.grid = self.generate_grid(objects, walls)

        self.n_steps += 1
        self.simulation_time = self.n_steps * self.env.TIMESTEP

        observation = {}
        observation["timestamp"] = self.simulation_time
        observation["SNGNN"] = info.get('DISCOMFORT_SNGNN', 0)
        observation["robot"] = robot
        observation["people"] = people
        observation["objects"] = objects
        observation["walls"] = walls
        observation["interactions"] = interactions

        done = terminated or truncated

        if not done:
            if self.simulation_time - self.last_save_simulation_time >= UPDATE_PERIOD:
                self.images_for_video.append(cv2.resize(image, (500, 500)))
                self.data.append(observation)
                self.last_save_simulation_time = self.simulation_time
        else:
            self.regenerate()
            self.last_data_update = time.time()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        labelSize = ((self.label.width() // 4) * 4, self.label.height())
        image = cv2.resize(image, labelSize)
        self.label.setPixmap(QtGui.QPixmap(QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)))

    def get_data(self):
        people = []
        for human in self.env.static_humans + self.env.dynamic_humans:
            person = {
                "id": human.id,
                "x": human.x,
                "y": human.y,
                "angle": human.orientation,
                "speed": human.speed
            }
            people.append(person)

        # Objects list will be empty as we have no objects
        objects = []

        walls = []
        for wall in self.env.walls:
            x1 = wall.x - np.cos(wall.orientation) * wall.length / 2
            x2 = wall.x + np.cos(wall.orientation) * wall.length / 2
            y1 = wall.y - np.sin(wall.orientation) * wall.length / 2
            y2 = wall.y + np.sin(wall.orientation) * wall.length / 2
            walls.append([x1, y1, x2, y2])

        interactions = []  # No interactions as per configuration

        robot = {
            "x": self.env.robot.x,
            "y": self.env.robot.y,
            "angle": self.env.robot.orientation,
            "speed_x": float(self.env.robot.vel_x),
            "speed_y": float(self.env.robot.vel_y),
            "speed_a": float(self.env.robot.vel_a),
            "goal_x": self.env.robot.goal_x,
            "goal_y": self.env.robot.goal_y
        }

        return people, objects, walls, interactions, robot

    def generate_grid(self, objects, walls):
        grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), np.int8)
        grid.fill(-1)
        room = []
        for w in walls:
            p1 = self.world_to_grid((w[0], w[1]))
            p2 = self.world_to_grid((w[2], w[3]))
            room.append(p1)
            room.append(p2)

        cv2.fillPoly(grid, [np.array(room, np.int32)], 0)
        cv2.polylines(grid, [np.array(room, np.int32)], True, 1)

        # No objects to draw as per configuration
        # The objects list is empty, so this loop will not execute
        for o in objects:
            pass

        # Visualization (Optional)
        v2gray = {-1: 128, 0: 255, 1: 0}
        visible_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), np.uint8)
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                visible_grid[y][x] = v2gray[grid[y][x]]

        visible_grid = cv2.flip(visible_grid, 0)
        cv2.imshow("grid", visible_grid)
        cv2.waitKey(1)
        return grid

    def world_to_grid(self, pW):
        pGx = pW[0] / GRID_CELL_SIZE + GRID_WIDTH / 2
        pGy = pW[1] / GRID_CELL_SIZE + GRID_HEIGHT / 2
        return (int(pGx), int(pGy))

    def rotate_points(self, points, center, angle):
        r_points = []
        for p in points:
            dx = p[0] - center[0]
            dy = p[1] - center[1]
            r_x = center[0] + dx * np.cos(angle) - dy * np.sin(angle)
            r_y = center[1] + dx * np.sin(angle) + dy * np.cos(angle)
            r_points.append((r_x, r_y))
        return r_points

    def regenerate(self):
        self.end_episode = time.time()
        if self.start_saving_button.isChecked():
            self.save_data()
        self.new_episode()

    def new_episode(self):
        self.images_for_video.clear()
        self.data.clear()
        self.env.reset()  # Resets the environment with the updated configuration
        self.ini_episode = time.time()
        self.simulation_time = 0
        self.last_save_simulation_time = -1
        self.n_steps = 0

    def save_data(self):
        if not self.images_for_video or not self.data:
            print("No data to save.")
            return
        file_name = self.dataID.text() + '{0:06d}'.format(self.data_file_index)

        final_data = {
            "grid": {
                "width": GRID_WIDTH,
                "height": GRID_HEIGHT,
                "cell_size": GRID_CELL_SIZE,
                "data": self.grid.tolist()
            },
            "sequence": self.data
        }
        try:
            with open(self.save_dir + file_name + '.json', 'w') as f:
                options = jsbeautifier.default_options()
                options.indent_size = 2
                f.write(jsbeautifier.beautify(json.dumps(final_data), options))
        except Exception as e:
            print("Error saving JSON data:", e)
            return

        # Convert images from BGR (OpenCV format) to RGB (imageio format)
        images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.images_for_video]

        fps = len(self.images_for_video) / (self.end_episode - self.ini_episode)
        file_path = self.save_dir + file_name + '.mp4'

        # Write video using imageio with H.264 codec
        imageio.mimwrite(file_path, images_rgb, fps=fps, codec='libx264')

        self.data_file_index += 1
        print(f"Data saved as {file_name}.json and {file_name}.mp4")



    def start_saving(self, save):
        if save:
            self.new_episode()

    def quit_slot(self):
        self.close()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
