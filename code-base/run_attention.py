# This file contains the skeleton you can use for traffic light attention
import json
import argparse
import math
from datetime import datetime
from argparse import Namespace
from pathlib import Path
from typing import Sequence, Optional, List, Any, Dict

from matplotlib.axes import Axes

# Internal imports... Should not fail
from consts import IMAG_PATH, JSON_PATH, NAME, SEQ_IMAG, X, Y, COLOR, RED, GRN, DATA_DIR, TFLS_CSV, CSV_OUTPUT, \
    CSV_INPUT, SEQ, CROP_DIR, CROP_CSV_NAME, ATTENTION_RESULT, ATTENTION_CSV_NAME, ZOOM, RELEVANT_IMAGE_PATH, COL, \
    ATTENTION_PATH, ZOOM
from misc_goodies import show_image_and_gt
from data_utils import get_images_metadata
from crops_creator import create_crops

import tqdm  # for the progress bar
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from scipy import signal as sg
import scipy.ndimage as ndimage
from scipy.ndimage import maximum_filter
from PIL import Image
import matplotlib.pyplot as plt

RED_THRESHOLD = 80
GREEN_THRESHOLD = 80

COUNTER = [0, 0, 0]


def convolve_rgb_image ( image: np.ndarray, kernel, mode='same' ) -> \
        (List[int], List[int], List[int], List[int], np.ndarray, np.ndarray):
    """
    Convolve a 2D image with a given kernel.
    :param image: a 2D array.
    :param kernel: a 2D array.
    :return: The convolved image.
    """

    red = convolve_2d_with_kernel(image[:, :, 0], kernel, mode=mode)
    green = convolve_2d_with_kernel(image[:, :, 1], kernel, mode=mode)
    red_peaks_x, red_peaks_y = find_light_point(red, RED_THRESHOLD, )
    green_peaks_x, green_peaks_y = find_light_point(green, GREEN_THRESHOLD, )
    return red_peaks_x, red_peaks_y, green_peaks_x, green_peaks_y, red, green


def convolve_2d_with_kernel ( image_channel: np.ndarray, kernel, mode='same' ) -> np.ndarray:
    """
    Convolve a 2D image with a given kernel.
    :param image: a 2D array.
    :param kernel: a 2D array.
    :return: The convolved image.
    """

    return sg.correlate2d(image_channel, kernel, mode=mode, boundary='symm')


def circle_kernel( size: int, radius: int, shift: int ):
    """
    Creates a circle kernel.
    :param size: The size of the kernel.
    :param radius: The radius of the circle.
    :param shift: The shift from the center of the kernel. all values below the shift will be as the center.
    :return: The circle kernel normalized to sum up to 0.
    """
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            # calculate the distance from the center of the kernel.
            dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            # shift the distance to the center of the kernel. all values below the shift will be as the center.
            dist = dist - shift
            dist = 0 if dist < 0 else dist
            # clip the distance to the radius. all values above the radius will have the tiniest value. (a black circle
            # around the kernel)
            dist = np.clip(dist, 0, radius)
            # make the kernel values -1 to 1. 1 in the center of the kernel and -1 in the edges.
            kernel[i, j] = 1 - (dist / radius)
    # make the kernel high-pass filter. (sum of the kernel values is 0)
    kernel = kernel - np.mean(kernel)
    # center of the kernel will be 1.
    kernel = kernel / np.max(kernel)
    kernel = kernel.astype(np.float32)
    return kernel


def find_light_point ( filtered_image, threshold_value ) -> (List[int], List[int]):
    """
    get- original_image- array of the original image
        filtered_image- array of image after convolving by red/green kernel
        threshold_value- (int) the level of clarity we want to use to filter
        color-string represent the color of the traffic light  "r" or "g"
    return list of cord(x,y) of the lightest points in the image
    """

    peaks_x = []
    peaks_y = []
    filter_size = 50
    max_filtered_image = maximum_filter(filtered_image, size=filter_size)
    # Invert pixel values using a for loop
    for i in range(filtered_image.shape[0]):
        for j in range(filtered_image.shape[1]):
            if max_filtered_image[i, j] == filtered_image[i, j] and filtered_image[i, j] > threshold_value:

                peaks_y.append(i)
                peaks_x.append(j)
    return peaks_x, peaks_y


def resize_image ( c_image: np.ndarray, ratio: float ) -> np.ndarray:
    """
    Resize the image by a given ratio while preserving the third dimension.

    :param c_image: np.ndarray, input image with shape (height, width, channels)
    :param ratio: float, resizing ratio
    :return: np.ndarray, resized image with shape (new_height, new_width, channels)
    """
    height, width, channels = c_image.shape
    new_height = int(height * ratio)
    new_width = int(width * ratio)

    # Calculate the actual ratio applied to the dimensions
    actual_ratio_height = new_height / height
    actual_ratio_width = new_width / width

    resized_image = np.zeros((new_height, new_width, channels), dtype=c_image.dtype)

    for c in range(channels):
        resized_image[:, :, c] = ndimage.zoom(c_image[:, :, c], (actual_ratio_height, actual_ratio_width))

    return resized_image


def filter_close_peaks ( peaks_x: List[int], peaks_y: List[int], color_list: List[str], zoom: List[float] ) -> \
        (List[int], List[int], List[str], List[float]):
    """
    Filter peaks that are too close to each other
    :param peaks_x: list of x coordinates of peaks
    :param peaks_y: list of y coordinates of peaks
    :param color_list: list of colors of peaks
    :param zoom: list of zoom ratios
    :return: filtered peaks
    """
    filtered_peaks = []
    for x, y, color, z in zip(peaks_x, peaks_y, color_list, zoom):
        conflicting_indices = [i for i, (fx, fy, _, _) in enumerate(filtered_peaks)
                               if abs(x - fx) < 5 and abs(y - fy) < 5]
        if conflicting_indices:
            filtered_peaks = [p for i, p in enumerate(filtered_peaks) if i not in conflicting_indices]
        filtered_peaks.append((x, y, color, z))

    return zip(*filtered_peaks) if filtered_peaks else ([], [], [], [])


def find_tfl_lights ( c_image: np.ndarray, **kwargs ) -> Dict[str, Any]:
    """
    Detect candidates for TFL lights.

    :param c_image: A H*W*3 RGB image of dtype np.uint8 (RGB, 0-255).
    :param kwargs: Whatever you want.
    :return: Dictionary with 'x', 'y', 'col', 'conv_im', and 'zoom' keys.
    """

    def preprocess_image ( image ):
        """
        Preprocess the image by removing the bottom part.
        """
        return image[:int(image.shape[0] * BOTTOM_PERCENT), :, :]

    def resize_and_convolve ( image, zoom_ratio,kernel ):
        """
        Resize the image and convolve it with a circle kernel.
        """
        resized = resize_image(image, ratio=zoom_ratio) if zoom_ratio != 1 else image
        return convolve_rgb_image(resized, kernel)

    def apply_zoom ( peaks, zoom_ratio ):
        """
        Apply the zoom ratio to the peaks.
        """
        return [int(x / zoom_ratio) for x in peaks]
    # Constants
    ZOOM_RATIOS = [1, 0.85, 0.5, 0.3]
    KERNEL_SIZE = 20
    FILTER_RADIUS = 5
    BOTTOM_PERCENT = 0.55

    # Initializations
    original_image = c_image
    # remove the bottom part of the image
    c_image = preprocess_image(c_image)
    kernel = circle_kernel(KERNEL_SIZE, FILTER_RADIUS, FILTER_RADIUS)
    filtered_peaks_x, filtered_peaks_y = [], []
    color_list, zoom_list = [], []
    red_convolved = np.ndarray([])
    green_convolved = None

    for zoom_ratio in ZOOM_RATIOS:
        red_peaks_x, red_peaks_y, green_peaks_x, green_peaks_y, green_convolved_new, red_convolved_new \
            = resize_and_convolve(c_image, zoom_ratio, kernel)

        if zoom_ratio == 1:
            # keep the convolved images for showing them later.
            green_convolved = green_convolved_new
            red_convolved = red_convolved_new
        else:
            # convert the peaks to the original image coordinates.
            red_peaks_x = apply_zoom(red_peaks_x, zoom_ratio)
            red_peaks_y = apply_zoom(red_peaks_y, zoom_ratio)
            green_peaks_x = apply_zoom(green_peaks_x, zoom_ratio)
            green_peaks_y = apply_zoom(green_peaks_y, zoom_ratio)

        filter_peaks(red_peaks_x, red_peaks_y, green_peaks_x, green_peaks_y, original_image)
        filtered_peaks_x += red_peaks_x + green_peaks_x
        filtered_peaks_y += red_peaks_y + green_peaks_y
        color_list += [RED] * len(red_peaks_x) + [GRN] * len(green_peaks_x)
        zoom_list += [zoom_ratio] * (len(red_peaks_x) + len(green_peaks_x))

    peaks_x, peaks_y, color_list, zoom_list = filter_close_peaks(filtered_peaks_x, filtered_peaks_y, color_list,
                                                                 zoom_list)

    return {
        X: peaks_x,
        Y: peaks_y,
        COLOR: color_list,
        'conv_im': [red_convolved, green_convolved],
        ZOOM: zoom_list
    }


def filter_peaks ( red_peaks_x, red_peaks_y, green_peaks_x, green_peaks_y, c_image ):
    """
    Filter the lists of red and green peaks based on their color using the checkColor function.

    Parameters:
    red_peaks_x (list): List of x-coordinates of red peaks.
    red_peaks_y (list): List of y-coordinates of red peaks.
    green_peaks_x (list): List of x-coordinates of green peaks.
    green_peaks_y (list): List of y-coordinates of green peaks.
    c_image: The image for color checking.
    """
    for x, y in zip(red_peaks_x, red_peaks_y):
        if check_color(x, y, c_image) != 'r' or y < 10:
            red_peaks_x.remove(x)
            red_peaks_y.remove(y)
    for x, y in zip(green_peaks_x, green_peaks_y):
        if check_color(x, y, c_image) != 'g' or y < 10:
            green_peaks_x.remove(x)
            green_peaks_y.remove(y)


def check_color ( center_x: int, center_y: int, c_image: np.ndarray ):
    """
    Check if the green color is significantly larger than red within circles of increasing radius around the center point.

    Args:
        center_x (int): x-coordinate of the center point.
        center_y (int): y-coordinate of the center point.
        c_image (np.ndarray): The input image as a NumPy array (RGB format).

    Returns:
        bool: True if green is significantly larger than red in circles around the center point; False otherwise.
    """
    max_radius = 10
    threshold_ratio = 1.2
    check_coords = []

    for radius in range(1, max_radius + 1):
        check_coords = []
        # Iterate over points in the circle around the center point
        for angle in np.linspace(0, 2 * math.pi, int(2 * math.pi * radius)):
            point_x = int(center_x + radius * math.cos(angle))
            point_y = int(center_y + radius * math.sin(angle))

            # Ensure the point is within the image bounds
            if 0 <= point_x < c_image.shape[1] and 0 <= point_y < c_image.shape[0]:
                red_value = c_image[point_y, point_x, 0]
                green_value = c_image[point_y, point_x, 1]

                # Check if the green value is significantly larger than red
                if green_value > red_value * threshold_ratio:
                    check_coords.append('g')  # Green is significantly larger than red within the circle
                elif green_value < red_value:
                    check_coords.append('r')
        if check_coords.count('g') > check_coords.count('r'):
            return "g"
        elif check_coords.count('g') < check_coords.count('r'):
            return "r"
    print(f"checkColor: {check_coords} , center_x: {center_x} , center_y: {center_y}")
    return "b"


def test_find_tfl_lights ( row: Series, args: Namespace ) -> DataFrame:
    """
    Run the attention code-base
    """
    image_path: str = row[IMAG_PATH]
    json_path: str = row[JSON_PATH]
    image: np.ndarray = np.array(Image.open(image_path), dtype=np.float32) / 255
    if args.debug and json_path is not None:
        # This code-base demonstrates the fact you can read the bounding polygons from the json files
        # Then plot them on the image. Try it if you think you want to. Not a must...
        gt_data: Dict[str, Any] = json.loads(Path(json_path).read_text())
        what: List[str] = ['traffic light']
        objects: List[Dict[str, Any]] = [o for o in gt_data['objects'] if o['label'] in what]
        ax: Optional[Axes] = show_image_and_gt(image, objects, f"{row[SEQ_IMAG]}: {row[NAME]} GT")
    else:
        ax = None
    # In case you want, you can pass any parameter to find_tfl_lights, because it uses **kwargs
    attention_dict: Dict[str, Any] = find_tfl_lights(image)
    convolved_image = attention_dict.pop('conv_im')[0]
    attention: DataFrame = pd.DataFrame(attention_dict)
    # Copy all image metadata from the row into the results, so we can track it later
    for k, v in row.items():
        attention[k] = v
    tfl_x: np.ndarray = attention[X].values
    tfl_y: np.ndarray = attention[Y].values
    color: np.ndarray = attention[COLOR].values
    is_red = color == RED
    is_green = color == GRN
    print(f"Image: {image_path}, {len(is_red)} reds, {len(is_green)} greens..")
    if args.debug:
        # And here are some tips & tricks regarding matplotlib
        # They will look like pictures if you use jupyter, and like magic if you use pycharm!
        # You can zoom one image, and the other will zoom accordingly.
        # I think you will find it very very useful!
        plt.figure(f"{row[SEQ_IMAG]}: {row[NAME]} detections")
        plt.clf()
        plt.subplot(211, sharex=ax, sharey=ax)
        plt.imshow(image)
        plt.title('Original image.. Always try to compare your output to it')
        plt.plot(tfl_x[is_red], tfl_y[is_red], 'rx', markersize=4)
        plt.plot(tfl_x[~is_red], tfl_y[~is_red], 'g+', markersize=4)
        hp_result: np.ndarray = convolved_image
        plt.subplot(212, sharex=ax, sharey=ax)
        plt.imshow(hp_result)
        plt.title('Some useless image for you')
        plt.suptitle("When you zoom on one, the other zooms too :-)")
    return attention


def prepare_list( in_csv_file: Path, args: Namespace ) -> DataFrame:
    """
    We assume all students are working on the same CSV with files.
    This filters the list, so if you want to test some specific images, it's easy.
    This way, you can ask your friends how they performed on image 42 for example
    You may want to set the default_csv_file to anything on your computer, to spare the -f parameter.
    Note you will need different CSV files for attention and NN parts.
    The CSV must have at least columns: SEQ, NAME, TRAIN_TEST_VAL.
    """
    if args.image is not None:
        # Don't limit by count, take explicit images
        args.count = None

    csv_list: DataFrame = get_images_metadata(in_csv_file,
                                              max_count=args.count,
                                              take_specific=args.image)
    return pd.concat([pd.DataFrame(columns=CSV_INPUT), csv_list], ignore_index=True)


def run_on_list ( meta_table: pd.DataFrame, func: callable, args: Namespace ) -> pd.DataFrame:
    """
    Take a function, and run it on a list. Return accumulated results.

    :param meta_table: A DF with the columns your function requires
    :param func: A function to take a row of the DF, and return a DF with some results
    :param args:
    """
    acc: List[DataFrame] = []
    time_0: datetime = datetime.now()
    for _, row in tqdm.tqdm(meta_table.iterrows()):
        res: DataFrame = func(row, args)
        acc.append(res)
    time_1: datetime = datetime.now()
    all_results: DataFrame = pd.concat(acc).reset_index(drop=True)
    print(f"Took me {(time_1 - time_0).total_seconds()} seconds for "
          f"{len(all_results)} results from {len(meta_table)} files")

    return all_results


def save_df_for_part_2 ( crops_df: DataFrame, results_df: DataFrame ):
    if not ATTENTION_PATH.exists():
        ATTENTION_PATH.mkdir()
    # Order the df by sequence, a nice to have.
    crops_sorted: DataFrame = crops_df.sort_values(by=SEQ)
    results_sorted: DataFrame = results_df.sort_values(by=SEQ_IMAG)
    attention_df: DataFrame = DataFrame(columns=ATTENTION_RESULT)
    row_template: Dict[str, Any] = {RELEVANT_IMAGE_PATH: '', X: '', Y: '', ZOOM: 0, COL: ''}
    for index, row in results_sorted.iterrows():
        row_template[RELEVANT_IMAGE_PATH] = row[IMAG_PATH]
        row_template[X], row_template[Y] = row[X], row[Y]
        row_template[COL] = row[COLOR]
        row_template[ZOOM] = row[ZOOM]
        attention_df = attention_df._append(row_template, ignore_index=True)
    if (ATTENTION_PATH / ATTENTION_CSV_NAME).exists():
        # read existing attention csv
        existing_attention = pd.read_csv(ATTENTION_PATH / ATTENTION_CSV_NAME)
        # concat existing attention with new attention.
        updated_attention = pd.concat([attention_df,existing_attention], ignore_index=True)
        # filter out duplicates by path and x,y coordinates.
        updated_attention = updated_attention.drop_duplicates(subset=[RELEVANT_IMAGE_PATH, X, Y])
        # write updated attention to csv.
        updated_attention.to_csv(ATTENTION_PATH / ATTENTION_CSV_NAME, index=False)
    else:
        attention_df.to_csv(ATTENTION_PATH / ATTENTION_CSV_NAME, index=False)

        # Write crops_sorted to CROP_CSV_NAME without erasing existing data.
    if (ATTENTION_PATH / CROP_CSV_NAME).exists():
        existing_crops = pd.read_csv(ATTENTION_PATH / CROP_CSV_NAME)
        updated_crops = pd.concat([crops_sorted, existing_crops], ignore_index=True)
        updated_crops = updated_crops.drop_duplicates(subset=[RELEVANT_IMAGE_PATH])
        updated_crops.to_csv(ATTENTION_PATH / CROP_CSV_NAME, index=False)

    else:
        crops_sorted.to_csv(ATTENTION_PATH / CROP_CSV_NAME, index=False)


def parse_arguments ( argv: Optional[Sequence[str]] ):
    """
    Here are all the arguments in the attention stage.
    """
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=int, nargs='+', help='Specific image number(s) to run')
    parser.add_argument("-c", "--count", type=int, default=300, help="Max images to run")
    parser.add_argument('-f', '--in_csv_file', type=str, help='CSV file to read')
    parser.add_argument('-nd', '--no_debug', action='store_true', help='Show debug info')
    parser.add_argument('--attention_csv_file', type=str, help='CSV to write results to')

    args = parser.parse_args(argv)

    args.debug = not args.no_debug

    return args


def main ( argv=None ):
    """
    It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually examine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module.

    :param argv: In case you want to programmatically run this.
    """

    args: Namespace = parse_arguments(argv)
    default_csv_file: Path = DATA_DIR / TFLS_CSV
    csv_filename: Path = Path(args.in_csv_file) if args.in_csv_file else default_csv_file

    # This is your output CSV, look for CSV_OUTPUT const to see its columns.
    # No need to touch this function, if your curious about the result, put a break point and look at the result
    meta_table: DataFrame = prepare_list(csv_filename, args)
    print(f"About to run attention on {len(meta_table)} images. Watch out there!")

    # When you run your find find_tfl_lights, you want to add each time the output (x,y coordinates, color etc.)
    # to the output Dataframe, look at CSV_OUTPUT to see the names of the column and in what order.
    all_results: DataFrame = run_on_list(meta_table, test_find_tfl_lights, args)
    combined_df: DataFrame = pd.concat([pd.DataFrame(columns=CSV_OUTPUT), all_results], ignore_index=True)

    # make crops out of the coordinates from the DataFrame
    crops_df: DataFrame = create_crops(combined_df)

    # save the DataFrames in the right format for stage two.
    save_df_for_part_2(crops_df, combined_df)
    print(f"Got a total of {len(combined_df)} results")
    # ---------------------------------------------------------------------------
    for i in range(len(COUNTER)):
        print(f"zoom ratio iter {i} added points {COUNTER[i]}")
    # ---------------------------------------------------------------------------
    if args.debug:
        plt.show(block=True)


if __name__ == '__main__':
    main()
