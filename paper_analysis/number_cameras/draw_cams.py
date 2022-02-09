import numpy as np
import random
import os

total_num_cameras = 12

number_camera_sets = 10

all_sets = []

for num_cams in range(2,12):
    sets = []
    while len(sets) < number_camera_sets:
            drawn_set = random.sample(range(total_num_cameras), num_cams)
            drawn_set.sort()
            if not drawn_set in sets:
                sets.append(drawn_set)
            else:
                print ("Found Duplicate, redrawing")
    all_sets.append(sets)

os.makedirs("camera_sets", exist_ok = True)

for i,set in enumerate(all_sets):
    np.savetxt(os.path.join('camera_sets', f'Set_{i+2}.csv'), set, delimiter=",")
