"""
prediction_utils.py
=================
"""

import numpy as np
import cv2


def draw_line(img, line, points2D, img_size, color):
    array_sum = np.sum(np.array(points2D))
    array_has_nan = np.isnan(array_sum)
    if ((not array_has_nan) and int(points2D[line[0]][0]) < img_size[0]-1
            and int(points2D[line[0]][0]) > 0
            and  int(points2D[line[1]][0]) < img_size[0]-1
            and int(points2D[line[1]][0]) > 0
            and int(points2D[line[0]][1]) < img_size[1]-1
            and int(points2D[line[0]][1]) > 0
            and int(points2D[line[1]][1]) < img_size[1]-1
            and int(points2D[line[1]][1]) > 0):
        cv2.line(img,
                (int(points2D[line[0]][0]), int(points2D[line[0]][1])),
                (int(points2D[line[1]][0]), int(points2D[line[1]][1])),
                color, 1)


def draw_point(img, point, img_size, color):
    array_sum = np.sum(np.array(point))
    array_has_nan = np.isnan(array_sum)
    if ((not array_has_nan) and (point[0] < img_size[0]-1
            and point[0] > 0 and point[1] < img_size[1]-1
            and point[1] > 0)):
        cv2.circle(img, (int(point[0]), int(point[1])),
                3, color, thickness=3)



def get_colors_and_lines(skeletonPreset):
    colors = []
    line_idxs = []
    if skeletonPreset == "Hand":
        colors = [(255,0,0), (255,0,0),(255,0,0),(255,0,0),
                  (0,255,0),(0,255,0),(0,255,0),(0,255,0),
                  (0,0,255),(0,0,255),(0,0,255),(0,0,255),
                  (255,255,0),(255,255,0),(255,255,0),
                  (255,255,0),(0,255,255),(0,255,255),
                  (0,255,255),(0,255,255),(255,0,255),
                  (100,0,100),(100,0,100)]
        line_idxs = [[0,1], [1,2], [2,3], [4,5], [5,6], [6,7],
                     [8,9], [9,10], [10,11], [12,13], [13,14],
                     [14,15], [16,17], [17,18], [18,19],
                     [15,18], [15,19], [15,11], [11,7], [7,3],
                     [3,21], [21,22], [19,22]]
    elif skeletonPreset == "HumanBody":
        colors = [(255,0,0),(255,0,0),(255,0,0),(255,0,0),
                  (100,100,100),(0,255,0),(0,255,0),(0,255,0),
                  (0,255,0),(0,0,255),(0,0,255),(0,0,255),
                  (0,0,255),(100,100,100),(255,0,255),
                  (255,0,255),(255,0,255),(255,0,255),
                  (255,255,0),(255,255,0),(255,255,0),(255,255,0)]
        line_idxs = [[4,5], [5,6], [6,7], [7,8], [4,9], [9,10],
                     [10,11], [11,12], [4,13], [13,14], [14,15],
                     [15,16], [16,17], [13,18], [18,19],
                     [19,20], [20,21]]
    elif skeletonPreset == "MonkeyBody":
        colors = [(255,0,0),(255,0,0),
                  (100,100,100),(100,100,100),(0,255,0),(0,255,0), (0,255,0),
                  (0,0,255),(0,0,255), (0,0,255),
                  (255,0,255) ,(255,0,255),(255,0,255),
                  (255,255,0),(255,255,0),(255,255,0)]
        line_idxs = [[2,4], [2,10], [4,5], [5,6], [3,7], [7,8], [8,9],
                     [3,2], [10,11], [11,12], [13,14],
                     [14,15], [3,13]]
    elif skeletonPreset == "RodentBody":
        colors = [(255,0,0), (255,0,0), (255,0,0),
                  (100,100,100), (0,255,0), (0,255,0),
                  (0,255,0), (0,255,0), (0,0,255), (0,0,255),
                  (0,0,255), (0,0,255),(100,100,100),
                  (0,255,255), (0,255,255), (0,255,255),
                  (255,0,255), (255,0,255), (255,0,255),
                  (255,255,0), (255,255,0), (255,255,0)]
        line_idxs = [[0,1], [0,2], [1,2],[1,3], [2,3], [3,7],
                     [7,6], [6,5], [5,4], [3,11], [11,10],
                     [10,9], [9,8], [3,12], [12,15], [15,14],
                     [14,13], [12,18], [18,17], [17,16], [12,19]]        

    return colors, line_idxs
