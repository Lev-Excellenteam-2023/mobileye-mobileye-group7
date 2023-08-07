from scipy.ndimage import maximum_filter

def find_light_point(image_c, threshold_value):
    """
    get- image_array
        threshold_value-
    return list of cord(x,y) of the lightest points in the image
    """
    midpoints=[]
    filter_size = 15
    filtered_image = maximum_filter(image_c, size=filter_size)
    # Invert pixel values using a for loop
    for i in range(image_c.shape[0]):
        for j in range(image_c.shape[1]):
            if (image_c[i,j]==filtered_image[i,j]) and image_c[i,j]>threshold_value:
                print( image_c[i, j])
                midpoints.append((i,j))
    return midpoints