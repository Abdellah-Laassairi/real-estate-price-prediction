import glob
import math
import os

from PIL import Image


def concatenate_images(path_image):
    # create a list of all image file names
    image_files = glob.glob(f'{path_image}/*.jpg')

    # create an empty list to store image objects
    images = []

    # loop through the image files and open them as image objects
    for file in image_files:
        img = Image.open(file)
        images.append(img)

    # calculate the dimensions of the output image
    num_images = len(images)
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    max_width = max([img.size[0] for img in images])
    max_height = max([img.size[1] for img in images])
    out_width = max_width * cols
    out_height = max_height * rows

    # create a new image object to store the concatenated image
    concatenated_image = Image.new('RGB', (out_width, out_height),
                                   (255, 255, 255))

    # loop through the images and paste them into the concatenated image
    x_offset = 0
    y_offset = 0
    for img in images:
        concatenated_image.paste(img, (x_offset, y_offset))
        x_offset += max_width
        if x_offset >= out_width:
            x_offset = 0
            y_offset += max_height
    return concatenated_image


# save the concatenated image
