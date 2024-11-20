#!/usr/bin/python3
#
# -*- coding: utf-8 -*-
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import time, sys
import numpy as np
import cv2
import json
import copy

def world_to_grid(pW, GRID_CELL_SIZE, GRID_HEIGHT, GRID_WIDTH):
    pGx = pW[0]/GRID_CELL_SIZE + GRID_WIDTH/2
    pGy = pW[1]/GRID_CELL_SIZE + GRID_HEIGHT/2
    return (int(pGx), int(pGy))

IMAGE_SIDE = 500
HUMAN_RADIUS = 0.35
ROBOT_RADIUS = 0.3
GOAL_RADIUS = 0.4
data = json.load(open(sys.argv[-1], 'r'))
grid = data["grid"]["data"]
GRID_HEIGHT = data["grid"]["height"]
GRID_WIDTH = data["grid"]["width"]
GRID_CELL_SIZE = data["grid"]["cell_size"]
grid = np.array(grid, np.int8)

v2gray = {-1:[128, 128, 128], 0: [255, 255, 255], 1: [0, 0, 0]}
global_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3), np.uint8)
for y in range(grid.shape[0]):
    for x in range(grid.shape[1]):
        global_grid[y][x] = v2gray[grid[y][x]]

scale = IMAGE_SIDE/GRID_WIDTH
GRID_WIDTH = IMAGE_SIDE
GRID_HEIGHT = int(GRID_HEIGHT*scale)
GRID_CELL_SIZE = GRID_CELL_SIZE/scale

global_grid = cv2.resize(global_grid, (GRID_HEIGHT, GRID_WIDTH))

last_timestamp = -1
for s in data["sequence"]:
    local_grid = copy.deepcopy(global_grid)

    # DRAW HUMANS
    for p in s["people"]:
        c = world_to_grid((p['x'], p['y']), GRID_CELL_SIZE, GRID_HEIGHT, GRID_WIDTH)
        r_p = world_to_grid((p['x']+HUMAN_RADIUS, p['y']), GRID_CELL_SIZE, GRID_HEIGHT, GRID_WIDTH)
        x_a = p['x'] + (HUMAN_RADIUS+0.05)*np.cos(p['angle'])
        y_a = p['y'] + (HUMAN_RADIUS+0.05)*np.sin(p['angle'])
        a = world_to_grid((x_a, y_a), GRID_CELL_SIZE, GRID_HEIGHT, GRID_WIDTH)
        r = abs(c[0]-r_p[0])
        cv2.circle(local_grid, c, r, [0, 0, 255], 2)
        cv2.line(local_grid, c, a, [0, 0, 255], 2)

    # DRAW ROBOT
    c = world_to_grid((s["robot"]['x'], s["robot"]['y']), GRID_CELL_SIZE, GRID_HEIGHT, GRID_WIDTH)
    r_p = world_to_grid((s["robot"]['x']+ROBOT_RADIUS, s["robot"]['y']), GRID_CELL_SIZE, GRID_HEIGHT, GRID_WIDTH)
    x_a = s["robot"]['x'] + (ROBOT_RADIUS+0.05)*np.cos(s["robot"]['angle'])
    y_a = s["robot"]['y'] + (ROBOT_RADIUS+0.05)*np.sin(s["robot"]['angle'])
    a = world_to_grid((x_a, y_a), GRID_CELL_SIZE, GRID_HEIGHT, GRID_WIDTH)
    r = abs(c[0]-r_p[0])
    cv2.circle(local_grid, c, r, [255, 0, 0], 2)
    cv2.line(local_grid, c, a, [255, 0, 0], 2)

    # DRAW GOAL
    c = world_to_grid((s["robot"]['goal_x'], s["robot"]['goal_y']), GRID_CELL_SIZE, GRID_HEIGHT, GRID_WIDTH)
    r_p = world_to_grid((s["robot"]['goal_x']+GOAL_RADIUS, s["robot"]['goal_y']), GRID_CELL_SIZE, GRID_HEIGHT, GRID_WIDTH)
    r = abs(c[0]-r_p[0])
    cv2.circle(local_grid, c, r, [0, 255, 0], 2)

    visible_grid = cv2.flip(local_grid, 0)                


    cv2.imshow("grid", visible_grid)
    cv2.waitKey(1)

    sleeptime = s["timestamp"]-last_timestamp
    if last_timestamp == -1:
        sleeptime = 0
    last_timestamp = s["timestamp"]
    time.sleep(sleeptime)

