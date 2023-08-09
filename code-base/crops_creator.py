from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from consts import CROP_DIR, CROP_RESULT, SEQ, IS_TRUE, IGNOR, CROP_PATH, X0, X1, Y0, Y1, COLOR, SEQ_IMAG, COL, X, Y, \
    GTIM_PATH,IMAG_PATH

from pandas import DataFrame, Series


def make_crop(*args, **kwargs) -> (int, int, int, int, np.ndarray):
    """
    The function that creates the crops from the image.
    Your return values from here should be the coordinates of the crops in this format (x0, x1, y0, y1, crop content):
    'x0'  The bigger x value (the right corner)
    'x1'  The smaller x value (the left corner)
    'y0'  The smaller y value (the lower corner)
    'y1'  The bigger y value (the higher corner)
    """
    row:Series = args[0]
    x = row[X]
    y = row[Y]
    color = row[COLOR]
    # use this to ignore a crop if is on the edge of the image
    ignore = False

    zoom_rate = row['zoom']
    imag_path = row[IMAG_PATH]
    c_image = np.array(Image.open(imag_path))
    copy_image = c_image.copy()
    if zoom_rate != 1.0:
        x = int(x * zoom_rate)
        y = int(y * zoom_rate)
        import run_attention
        c_image = run_attention.resize_image(copy_image, zoom_rate)


    # find coordinates of the crop to create a rectangle around the traffic light
    x0 = x - 12
    x1 = x + 12
    y0 = y - 50 if color == 'g' else y - 10
    y1 = y + 10 if color == 'g' else y + 50
    # check if the coordinates are out of the image
    if x0 < 0:
        x1 = x1 - x0
        x0 = 0
    if x1 > c_image.shape[1]:
        x0 = x0 - (x1 - c_image.shape[1])
        x1 = c_image.shape[1]
    if y0 < 0:
        y1 = y1 - y0
        y0 = 0
    if y1 > c_image.shape[0]:
        y0 = y0 - (y1 - c_image.shape[0])
        y1 = c_image.shape[0]
    # create the crop
    crop = c_image[int(y0):int(y1), int(x0):int(x1)]

    return x, x, y, y, crop, ignore





def check_crop(*args, **kwargs):
    """
    Here you check if your crop contains a traffic light or not.
    Try using the ground truth to do that (Hint: easier than you think for the simple cases, and if you found a hard
    one, just ignore it for now :). )
    """
    return True, kwargs['ignore']


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
        # ---------------------------------------------------------------------------
        print(f'creating crop number {index} from image {row[SEQ_IMAG]}')
        # ---------------------------------------------------------------------------

        result_template[SEQ] = row[SEQ_IMAG]
        result_template[COL] = row[COLOR]

        # example code:
        # ******* rewrite ONLY FROM HERE *******
        x0, x1, y0, y1, crop, ignore = make_crop(row)
        result_template[X0], result_template[X1], result_template[Y0], result_template[Y1] = x0, x1, y0, y1
        crop_path: str = f'tfl_cropped_image_{row[SEQ_IMAG]}_cord_{int(row[X]),int(row[Y])}.png'
        # converts the crop to an image and saves it in the 'data' folder
        # create the png file
        # print(np.max(crop))
        crop = crop.astype(np.uint8)
        crop_image = Image.fromarray(crop)
        crop_image.save(CROP_DIR / crop_path)

        result_template[CROP_PATH] = crop_path
        result_template[IS_TRUE], result_template[IGNOR] = check_crop(crop, x0, x1, y0, y1,ignore=ignore)
        # ******* TO HERE *******

        # added to current row to the result DataFrame that will serve you as the input to part 2 B).
        result_df = result_df._append(result_template, ignore_index=True)
    return result_df