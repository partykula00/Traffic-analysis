from scipy.spatial import distance as dist
import numpy as np
import random
from coordinates import PixelMapper
from coordinates import haversine
import math

def calculate_speed(centroids_1, centroids_2, fps, frames_passed):
    # # Average car dimensions in meters
    quad_coords = {
        "lonlat": np.array([
            [5.974829, 45.526886],  # Górny prawy róg
            [5.974154, 45.527237],  # Górny lewy róg
            [5.974756, 45.526366],  # Dolny lewy róg
            [5.975188, 45.526461]   # Dolny prawy róg
        ]),
        "pixel": np.array([
            [830, 341],  # Górny prawy róg
            [530, 297],  # Górny lewy róg
            [215, 409],  # Dolny lewy róg
            [998, 448]   # Dolny prawy róg
        ])
    }
    pm = PixelMapper(quad_coords["pixel"], quad_coords["lonlat"])

    #print(centroids_1.shape)

    y = list(centroids_1)
    x1 = y[0].item()
    x2 = y[1].item()
    #print(type(x1), type(x2))
    centroids_1_tup = (x1, x2)
    y = list(centroids_2)
    x1 = y[0].item()
    x2 = y[1].item()
    centroids_2_tup = (x1, x2)
    #print(centroids_2_tup, centroids_1_tup)

    centroids_1_real = pm.pixel_to_lonlat(centroids_1_tup)
    centroids_2_real = pm.pixel_to_lonlat(centroids_2_tup)
    #print(type(centroids_1_real), centroids_1_real.shape)
    #print(type(centroids_2_real), centroids_2_real)

    #print(centroids_1_real[0][0])
    distance = haversine(centroids_1_real[0][0], centroids_1_real[0][1],centroids_2_real[0][0], centroids_2_real[0][1])

    #print(distance)

    delta_time = 0.3
    #print(delta_time)

    speed = (distance/delta_time) * 3600
#    if speed < 85 or speed > 140:
#        speed = random.randint(81, 139)
    return speed
