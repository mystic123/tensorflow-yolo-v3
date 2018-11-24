import numpy as np
from PIL import Image


def letter_box_image(image: Image.Image, output_height: int, output_width: int, fill_value)-> np.ndarray:
    """
    Fit image with final image with output_width and output_height.
    :param image: PILLOW Image object.
    :param output_height: width of the final image.
    :param output_width: height of the final image.
    :param fill_value: fill value for empty area. Can be uint8 or np.ndarray
    :return: numpy image fit within letterbox. dtype=uint8, shape=(output_height, output_width)
    """

    height_ratio = float(output_height)/image.size[1]
    width_ratio = float(output_width)/image.size[0]
    fit_ratio = min(width_ratio, height_ratio)
    fit_height = int(image.size[1] * fit_ratio)
    fit_width = int(image.size[0] * fit_ratio)
    fit_image = np.asarray(image.resize((fit_width, fit_height), resample=Image.BILINEAR))

    if isinstance(fill_value, int):
        fill_value = np.full(fit_image.shape[2], fill_value, fit_image.dtype)

    to_return = np.tile(fill_value, (output_height, output_width, 1))
    pad_top = int(0.5 * (output_height - fit_height))
    pad_left = int(0.5 * (output_width - fit_width))
    to_return[pad_top:pad_top+fit_height, pad_left:+fit_width] = fit_image
    return to_return


def letter_box_pos_to_original_pos(letter_pos, current_size, ori_image_size)-> np.ndarray:
    """
    Parameters should have same shape and dimension space. (Width, Height) or (Height, Width)
    :param letter_pos: The current position within letterbox image including fill value area.
    :param current_size: The size of whole image including fill value area.
    :param ori_image_size: The size of image before being letter box.
    :return:
    """
    letter_pos = np.asarray(letter_pos, dtype=np.float)
    current_size = np.asarray(current_size, dtype=np.float)
    ori_image_size = np.asarray(ori_image_size, dtype=np.float)
    final_ratio = min(current_size[0]/ori_image_size[0], current_size[1]/ori_image_size[1])
    pad = 0.5 * (current_size - final_ratio * ori_image_size)
    pad = pad.astype(np.int32)
    to_return_pos = (letter_pos - pad) / final_ratio
    return to_return_pos

