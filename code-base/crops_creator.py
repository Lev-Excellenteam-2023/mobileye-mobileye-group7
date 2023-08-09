import json
from typing import Dict, Any

import numpy as np

from consts import CROP_DIR, CROP_RESULT, SEQ, IS_TRUE, IGNOR, CROP_PATH, X0, X1, Y0, Y1, COLOR, SEQ_IMAG, COL, X, Y, \
    GTIM_PATH

from pandas import DataFrame


def make_crop(*args, **kwargs):
    """
    The function that creates the crops from the image.
    Your return values from here should be the coordinates of the crops in this format (x0, x1, y0, y1, crop content):
    'x0'  The bigger x value (the right corner)
    'x1'  The smaller x value (the left corner)
    'y0'  The smaller y value (the lower corner)
    'y1'  The bigger y value (the higher corner)
    """
    x = args[0]
    y = args[1]
    color = args[2]
    # find coordinates of the crop to create a rectangle around the traffic light
    x0 = x - 40
    x1 = x + 40
    y0 = y - 100 if color == 'r' else y - 40
    y1 = y + 40 if color == 'r' else y + 100
    return x, x, y, y, color


def check_crop_in_json(x0: int, x1: int, y0: int, y1: int, image_path: str) -> bool:
    """
    Check if a given crop region intersects with any "traffic light" polygons in a JSON annotation file.

    Parameters:
    x0 (int): The leftmost x-coordinate of the crop region.
    x1 (int): The rightmost x-coordinate of the crop region.
    y0 (int): The topmost y-coordinate of the crop region.
    y1 (int): The bottommost y-coordinate of the crop region.
    image_path (str): Path to the image for which the annotation JSON is being checked.

    Returns:
    bool: True if the crop region intersects with any "traffic light" polygon, False otherwise.
    """

    json_path = image_path.replace("_leftImg8bit.png", "_gtFine_polygons.json")

    # Load the JSON data from the file
    with open(json_path) as json_file:
        data = json.load(json_file)

        # Iterate through the "objects" array and find objects labeled as "traffic light"
        for obj in data.get('objects', []):
            if obj.get('label') == 'traffic light':
                points = obj.get('polygon', [])
                x_values, y_values = zip(*points)
                min_x, max_x = min(x_values), max(x_values)
                min_y, max_y = min(y_values), max(y_values)
                if (x0 > min_x - 5 and x1 < max_x + 5 and y0 > min_y - 5 and y1 < max_y + 5):
                    return True
    return False


def check_crop(x0, x1, y0, y1, row):
    """
    Here you check if your crop contains a traffic light or not.
    Try using the ground truth to do that (Hint: easier than you think for the simple cases, and if you found a hard
    one, just ignore it for now :). )
    """
    x0 = 1635
    x1 = 1670
    y0 = 100
    y1 = 200
    image_path = "C:\mobileye-mobileye-group7\data\\fullImages\\aachen_000010_000019_leftImg8bit.png"
    if check_crop_in_json(x0, x1, y0, y1, image_path):
        return True


def create_crops(df: DataFrame) -> DataFrame:
    # Your goal in this part is to take the coordinates you have in the df, run on it, create crops from them, save them
    # in the 'data' folder, then check if crop you have found is correct (meaning the TFL is fully contained in the
    # crop) by comparing it to the ground truth and in the end right all the result data you have in the following
    # DataFrame (for doc about each field and its input, look at 'CROP_RESULT')
    #
    # *** IMPORTANT ***
    # All crops should be the same size or smaller!!!

    # creates a folder for you to save the crops in, recommended not must
    if not CROP_DIR.exists():
        CROP_DIR.mkdir()

    # For documentation about each key end what it means, click on 'CROP_RESULT' and see for each value what it means.
    # You wanna stick with this DataFrame structure because its output is the same as the input for the next stages.
    result_df = DataFrame(columns=CROP_RESULT)

    # A dict containing the row you want to insert into the result DataFrame.
    result_template: Dict[Any] = {SEQ: '', IS_TRUE: '', IGNOR: '', CROP_PATH: '', X0: '', X1: '', Y0: '', Y1: '',
                                  COL: ''}
    for index, row in df.iterrows():
        result_template[SEQ] = row[SEQ_IMAG]
        result_template[COL] = row[COLOR]

        # example code:
        # ******* rewrite ONLY FROM HERE *******
        #  x0, x1, y0, y1, crop = make_crop(df[X], df[Y],df[COLOR])
        x0, x1, y0, y1, crop = 0, 0, 0, 0, 0
        result_template[X0], result_template[X1], result_template[Y0], result_template[Y1] = x0, x1, y0, y1
        crop_path: str = '/data/crops/' + f'tfl_cropped_image_{df[SEQ_IMAG]} cord:({df[X], df[Y]})'
        # crop.save(CROP_DIR / crop_path)
        result_template[CROP_PATH] = crop_path
        result_template[IS_TRUE], result_template[IGNOR] = check_crop(x0, x1, y0, y1, crop, row)
        # ******* TO HERE *******

        # added to current row to the result DataFrame that will serve you as the input to part 2 B).
        result_df = result_df._append(result_template, ignore_index=True)
    return result_df
