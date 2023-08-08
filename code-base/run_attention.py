# This file contains the skeleton you can use for traffic light attention
import json
import argparse
from datetime import datetime
from argparse import Namespace
from pathlib import Path
from typing import Sequence, Optional, List, Any, Dict

from matplotlib.axes import Axes

# Internal imports... Should not fail
from consts import IMAG_PATH, JSON_PATH, NAME, SEQ_IMAG, X, Y, COLOR, RED, GRN, DATA_DIR, TFLS_CSV, CSV_OUTPUT, \
    CSV_INPUT,SEQ, CROP_DIR, CROP_CSV_NAME, ATTENTION_RESULT, ATTENTION_CSV_NAME, ZOOM, RELEVANT_IMAGE_PATH, COL, ATTENTION_PATH
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




def convolve_2d_with_kernel(image_channel:np.ndarray, kernel, mode='same') -> np.ndarray:
    """
    Convolve a 2D image with a given kernel.
    :param image: a 2D array.
    :param kernel: a 2D array.
    :return: The convolved image.
    """
    return sg.convolve2d(image_channel, kernel, mode=mode)
def circle_kernael(size:int, radius:int ,shift:int):
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
            dist = dist - shift
            dist = 0 if dist < 0 else dist
            # clip the distance to the radius. all values above the radius will have the tiniest value. (a black circle
            # around the kernel)
            dist = np.clip(dist, 0, radius)
            # make the kernel values -1 to 1. 1 in the center of the kernel and -1 in the edges.
            kernel[i, j] = 1 - (dist / radius)
    # make the kernel high-pass filter. (sum of the kernel values is 0)
    kernel = kernel - np.mean(kernel)
    kernel = kernel / np.max(kernel)
    print(np.sum(kernel))

    # plt.imshow(kernel, cmap='gray')
    # plt.show()
    return kernel


def find_light_point(image_c, threshold_value):
    """
    get- image_array
        threshold_value-
    return list of cord(x,y) of the lightest points in the image
    """
    midpoints = []
    filter_size = 15
    filtered_image = maximum_filter(image_c, size=filter_size)
    # Invert pixel values using a for loop
    for i in range(image_c.shape[0]):
        for j in range(image_c.shape[1]):
            if (image_c[i,j]==filtered_image[i,j]) and image_c[i,j]>threshold_value:
              #  print( image_c[i, j])
                midpoints.append((i,j))
    return midpoints




def find_tfl_lights(c_image: np.ndarray, **kwargs) -> Dict[str, Any]:
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement.

    :param c_image: a H*W*3 RGB image of dtype np.uint8 (RGB, 0-255).
    :param kwargs: Whatever you want.
    :return: Dictionary with at least the following keys: 'x', 'y', 'col', each containing a list (same lengths).
    # Note there are no explicit strings in the code-base. ALWAYS USE A CONSTANT VARIABLE INSTEAD!.
    """
    low_pass_kernel = np.array([[1 / 9, 1 / 9, 1 / 9],
                       [1 / 9, 1 / 9, 1 / 9],
                       [1 / 9, 1 / 9, 1 / 9]])
    kernel = circle_kernael(25, 10, 6)
    kernel = kernel.astype(np.float32)
    green_channel = c_image[:,:,1]
    red_chanel = c_image[:,:,0]

    # red_chanel = convolve_2d_with_kernel(red_chanel, low_pass_kernel)
    # green_channel = convolve_2d_with_kernel(green_channel, low_pass_kernel)

    filtered_red_chanel = convolve_2d_with_kernel(red_chanel, kernel)
    filtered_green_chanel = convolve_2d_with_kernel(green_channel, kernel)
    # filtered_red_chanel = convolve_2d_with_kernel(filtered_red_chanel, low_pass_kernel)
    # filtered_green_chanel = convolve_2d_with_kernel(filtered_green_chanel, low_pass_kernel)

    # plt.imshow(filtered_red_chanel, cmap='gray')
    # plt.show()
    # plt.imshow(filtered_green_chanel, cmap='gray')
    # plt.show()

    red_peaks = find_light_point(filtered_red_chanel, 100)
    green_peaks = find_light_point(filtered_green_chanel, 100)
    red_peaks_x = [x[1] for x in red_peaks]
    red_peaks_y = [x[0] for x in red_peaks]
    green_peaks_x = [x[1] for x in green_peaks]
    green_peaks_y = [x[0] for x in green_peaks]
    return {X: red_peaks_x + green_peaks_x,
            Y: red_peaks_y + green_peaks_y,
            COLOR: [RED] * len(red_peaks_x) + [GRN] * len(green_peaks_x),
                }




def test_find_tfl_lights(row: Series, args: Namespace) -> DataFrame:
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
    attention_dict: Dict[str, Any] = find_tfl_lights(image, some_threshold=42, debug=args.debug)
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
        # Now let's convolve. Cannot convolve a 3D image with a 2D kernel, so I create a 2D image
        # Note: This image is useless for you, so you solve it yourself
        useless_image: np.ndarray = np.std(image, axis=2)  # No. You don't want this line in your code-base
        highpass_kernel_from_lecture: np.ndarray = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) - 1 / 9
        hp_result: np.ndarray = sg.convolve(useless_image, highpass_kernel_from_lecture, 'same')
        plt.subplot(212, sharex=ax, sharey=ax)
        plt.imshow(hp_result)
        plt.title('Some useless image for you')
        plt.suptitle("When you zoom on one, the other zooms too :-)")
    return attention


def prepare_list(in_csv_file: Path, args: Namespace) -> DataFrame:
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


def run_on_list(meta_table: pd.DataFrame, func: callable, args: Namespace) -> pd.DataFrame:
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


def save_df_for_part_2(crops_df: DataFrame, results_df: DataFrame):
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
        attention_df = attention_df._append(row_template, ignore_index=True)
    attention_df.to_csv(ATTENTION_PATH / ATTENTION_CSV_NAME, index=False)
    crops_sorted.to_csv(ATTENTION_PATH / CROP_CSV_NAME, index=False)


def parse_arguments(argv: Optional[Sequence[str]]):
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


def main(argv=None):
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

    if args.debug:
        plt.show(block=True)


if __name__ == '__main__':
    main()


