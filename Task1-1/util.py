import cv2 as cv
import numpy as np

SEED = 2022
np.random.seed(SEED)

import json
from typing import Tuple, List

def decode_json(json_file_path: str) -> Tuple:
    ''' This function processes the data json file for an image\n
        it loop over all the points and generate an points list\n
        containing x and y cordinates\n
        Also, it calculats the number different elements in the data\n
        just to generate random colors for the polygones\n
        ------------------------------------------------------------\n
        Takes - data json file path\n
        Returns - a tuple\n
        first element a numpy array of the points to draw polygones\n
        second element a list of car parts for the lable\n
        third width of the image\n
        fourth height of the image
    '''
    try:
        with open(json_file_path, 'r') as f:
            f = json.load(f)
    except FileNotFoundError as fnf:
        print(f'{fnf.strerror}')
        return None, None

    pts = []
    car_parts = []

    width = f[1]['original_width']
    height = f[1]['original_height']

    for i in range(len(f)):
        if f[i]['type'] == 'polygonlabels':
            pt = f[i]['value']['points']
            temp = []
            for p in pt:
                x = (p[0] * width) / 100
                y = (p[1] * height) / 100
                temp.append([x, y]) 
            pt = np.array(object=temp, dtype=np.int32)
            pt = pt.reshape((1, -1, 2))
            pts.append(pt)

            car_parts.append(f[i]['value']['polygonlabels'][0])
    
    return pts, car_parts, width, height

def decode_points(pts: List) -> Tuple:
    ''' This function will extract the polygone points of x and y\n
        cordinates and generates points that are inside the polygone\n
        ------------------------------------------------------------\n
        Takes - list of polygone points
        Returns - a tuple
        first element is list of inside points
        second element is xmins
        third element is ymins
        fourth element is xmaxs
        fifth element is ymaxs
    '''
    inside_pts = []
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []

    for p in pts:

        min = np.min(a=p, axis=1)
        xmin = min[0][0]
        ymin = min[0][1]

        max = np.max(p, axis=1)
        xmax = max[0][0]
        ymax = max[0][1]

        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)

        temp_ip = []

        for val in p[0]:

            x = val[0]
            y = val[1]

            if x >= xmin and x <= xmax:
                if y >= ymin and y <= ymax:
                    temp_ip.append([x, y])
        temp_ip = np.array(object=temp_ip, dtype=np.int32)
        temp_ip = temp_ip.reshape((1, -1, 2))
        inside_pts.append(temp_ip)
    
    return inside_pts, xmins, ymins, xmaxs, ymaxs

def draw_polygones(pts: List, inside_pts: List, colors: List, image: np.array, opacity: float, xmins: List, ymins: List, xmaxs: List, ymaxs: List) -> np.array:
    ''' This function will draw polygone on the image, provided the\n
        points with x, y cordinates and also it will fill the polygones\n
        with the same color and the third thing is to alpha blend the images\n
        --------------------------------------------------------------------\n
        Takes - list of points for polygone, and list of inside points to fill\n
                color, image and opacity to control\n
                the transparency of the color inside the polygone\n
        Returns - final image of np.array
    '''
    foreground = image.copy()
    background = image.copy()

    for i in range(len(pts)):
        _ = cv.polylines(
            img=foreground,
            pts=pts[i],
            isClosed=True,
            color=colors[i],
            thickness=2
        )

        cv.rectangle(
            img=foreground,
            pt1=(xmins[i], ymins[i]),
            pt2=(xmaxs[i], ymaxs[i]),
            color=colors[i],
            thickness=3
        )

        _ = cv.fillPoly(
            img=background,
            pts=inside_pts[i],
            color=colors[i] * 0.9,
        )

        alpha = opacity
        beta = 1 - alpha
        result_image = cv.addWeighted(foreground, alpha, background, beta, 0.0)

    return result_image

def draw_rectangle_w_text(image: np.array, inside_pts: List, xmins: List, ymins: List, xmaxs: List, ymaxs: List, colors: List, car_parts: List, w: int, opacity: float) -> np.array:
    ''' This function will draw rectangles around the objects and also put\n
        text on the top left corner of the rectange as well\n
        ------------------------------------------------------------------\n
        Takes - an image of np.array, xmins, ymins, xmaxs, and ymaxs are list\n
                list of colors, list of car parts, width and height
        Returns - np.array of image
    '''
    for i in range(len(xmins)):

        font_scale = np.min([1, np.max([3, int(w/500)])])
        font_thickness = np.min([2, np.max([5, int(w/50)])])

        p1, p2 = (int(xmins[i]), int(ymins[i])), (int(xmaxs[i]), int(ymaxs[i]))

        (tw, th), _ = cv.getTextSize(
            car_parts[i],
            cv.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            thickness=font_thickness
        )

        p2 = p1[0] + tw, p1[1] - th - 10

        cv.rectangle(
            image,
            p1,
            p2,
            color=colors[i],
            thickness=-1
        )

        cv.putText(
            image,
            car_parts[i],
            (xmins[i] + 1, ymins[i] -10),
            cv.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness=font_thickness 
        )
    
    return image

def visualise_main(image_path: str, meta_data_json_path: str, opacity: float) -> Tuple:
    ''' This function will combine all the tasks and returns\n
        two images one with filled transparent polygones and\n
        second with rectangles with appropriate part name\n
        ----------------------------------------------------\n
        Takes - image path, meta data json file path, and opaciy\n
                to control the transparency\n
        Returns - a tuple of two np.array images\n
        first image is of filled polygones\n
        second image is of ractangles with text
    '''
    pts, car_parts, w, h = decode_json(meta_data_json_path)
    colors = np.random.uniform(0, 255, size=(len(car_parts), 3))
    inside_pts, xmins, ymins, xmaxs, ymaxs = decode_points(pts)

    image = cv.imread(image_path)

    image_w_polygones = draw_polygones(
        pts,
        inside_pts,
        colors,
        image,
        opacity,
        xmins,
        ymins,
        xmaxs,
        ymaxs
    )

    new_image = image_w_polygones.copy()

    image_w_rectangles = draw_rectangle_w_text(
        new_image,
        inside_pts,
        xmins,
        ymins,
        xmaxs,
        ymaxs,
        colors,
        car_parts,
        w,
        opacity
    )

    return image_w_polygones, image_w_rectangles