import cv2
import numpy as np
from math import radians, cos, sin, asin, sqrt

videoPath = "testing2.mp4"
quad_coords = {
    "lonlat": np.array([
        [5.974829, 45.526886],
        [5.974154, 45.527237], 
        [5.974756, 45.526366], 
        [5.975188, 45.526461] 
    ]),
    "pixel": np.array([
        [830, 341],
        [530, 297],
        [215, 409], 
        [998, 448] 
    ])
}
class PixelMapper(object):

    def __init__(self, pixel_array, lonlat_array):
        assert pixel_array.shape == (4, 2), "Need (4,2) input array"
        assert lonlat_array.shape == (4, 2), "Need (4,2) input array"
        self.M = cv2.getPerspectiveTransform(np.float32(pixel_array), np.float32(lonlat_array))
        self.invM = cv2.getPerspectiveTransform(np.float32(lonlat_array), np.float32(pixel_array))

    def pixel_to_lonlat(self, pixel):

        if type(pixel) != np.ndarray:
            pixel = np.array(pixel).reshape(1, 2)
        assert pixel.shape[1] == 2, "Need (N,2) input array"
        pixel = np.concatenate([pixel, np.ones((pixel.shape[0], 1))], axis=1)
        lonlat = np.dot(self.M, pixel.T)

        return (lonlat[:2, :] / lonlat[2, :]).T

    def lonlat_to_pixel(self, lonlat):

        if type(lonlat) != np.ndarray:
            lonlat = np.array(lonlat).reshape(1, 2)
        assert lonlat.shape[1] == 2, "Need (N,2) input array"
        lonlat = np.concatenate([lonlat, np.ones((lonlat.shape[0], 1))], axis=1)
        pixel = np.dot(self.invM, lonlat.T)

        return (pixel[:2, :] / pixel[2, :]).T

def haversine(lon1, lat1, lon2, lat2):

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])


    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

pm = PixelMapper(quad_coords["pixel"], quad_coords["lonlat"])

test = (418, 346)
lonlat_test = pm.pixel_to_lonlat(test)


