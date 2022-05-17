import tensorflow as tf
import numpy as np

# based on https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
class Dict(dict):
    """
    Example:
    m = Dict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(Dict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
          for k, v in kwargs.items():
            self[k] = v

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Dict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Dict, self).__delitem__(key)
        del self.__dict__[key]



# all functions are implemented based on
# <http://www.easyrgb.com/en/math.php#text8>


def lab_to_xyz(image: tf.Tensor) -> tf.Tensor:
    """
    Convert an image from LAB color space to XYZ color space
    Parameters
    ----------
    image: tf.Tensor
    Returns
    -------
    tf.Tensor : LAB image
    """
    l, a, b = tf.unstack(image, axis=-1)

    var_y = (l + 16) / 116
    var_x = a / 500 + var_y
    var_z = var_y - b / 200
    var_z = tf.clip_by_value(var_z, 0.0, tf.float32.max)

    var_x = tf.where(tf.pow(var_x, 3) > 0.008856, tf.pow(var_x, 3),
                     (var_x - 16 / 116) / 7.787)
    var_y = tf.where(tf.pow(var_y, 3) > 0.008856, tf.pow(var_y, 3),
                     (var_y - 16 / 116) / 7.787)
    var_z = tf.where(tf.pow(var_z, 3) > 0.008856, tf.pow(var_z, 3),
                     (var_z - 16 / 116) / 7.787)

    refx = 95.047
    refy = 100.00
    refz = 108.883

    x = var_x * refx
    y = var_y * refy
    z = var_z * refz
    xyz_image = tf.stack([x, y, z], axis=-1)
    return xyz_image


def xyz_to_lab(image: tf.Tensor) -> tf.Tensor:
    """
    Convert an image from XYZ color space to LAB color space
    Parameters
    ----------
    image: tf.Tensor
    Returns
    -------
    tf.Tensor : LAB image
    """
    x, y, z = tf.unstack(image, axis=-1)

    refx = 95.047
    refy = 100.00
    refz = 108.883

    var_x = x / refx
    var_y = y / refy
    var_z = z / refz

    var_x = tf.where(var_x > 0.008856, tf.pow(var_x, 1 / 3),
                     (7.787 * var_x) + (16 / 116))
    var_y = tf.where(var_y > 0.008856, tf.pow(var_y, 1 / 3),
                     (7.787 * var_y) + (16 / 116))
    var_z = tf.where(var_z > 0.008856, tf.pow(var_z, 1 / 3),
                     (7.787 * var_z) + (16 / 116))

    l = (116 * var_y) - 16
    a = 500 * (var_x - var_y)
    b = 200 * (var_y - var_z)
    lab_image = tf.stack([l, a, b], axis=-1)
    return lab_image


def xyz_to_rgb(image: tf.Tensor) -> tf.Tensor:
    """
    Convert an image from XYZ color space to RGB color space
    Parameters
    ----------
    image: tf.Tensor
    Returns
    -------
    tf.Tensor: RGB image
    """
    x, y, z = tf.unstack(image, axis=-1)
    var_x = x / 100
    var_y = y / 100
    var_z = z / 100

    var_r = var_x * 3.2406 + var_y * -1.5372 + var_z * -0.4986
    var_g = var_x * -0.9689 + var_y * 1.8758 + var_z * 0.0415
    var_b = var_x * 0.0557 + var_y * -0.2040 + var_z * 1.0570

    var_rgb = tf.cast(tf.stack([var_r, var_g, var_b], axis=-1), tf.float32)
    # sometimes reaches a small negative number, which causes NaNs
    var_rgb = tf.clip_by_value(var_rgb, 1e-6, tf.float32.max)

    rgb_image = tf.where(var_rgb > 0.0031308,
                     1.055 * tf.pow(var_rgb, (1 / 2.4)) - 0.055,
                     12.92 * var_rgb)
    # var_g = tf.where(var_g > 0.0031308,
    #                  1.055 * tf.pow(var_g, (1 / 2.4)) - 0.055,
    #                  12.92 * var_g)
    # var_b = tf.where(var_b > 0.0031308,
    #                  1.055 * tf.pow(var_b, (1 / 2.4)) - 0.055,
    #                  12.92 * var_b)
    # r = var_r * 255
    # g = var_g * 255
    # b = var_b * 255
    # rgb_image = tf.cast(tf.stack([r, g, b], axis=-1), tf.uint8)

    return rgb_image


def rgb_to_xyz(image: tf.Tensor) -> tf.Tensor:
    """
    Convert an image from RGB color space to XYZ color space
    Parameters
    ----------
    tf.Tensor: tf.Tensor
    Returns
    -------
    tf.tensor: XYZ image
    """
    r, g, b = tf.unstack(image, axis=-1)
    var_r = r / 255
    var_g = g / 255
    var_b = b / 255

    var_r = tf.where(var_r > 0.04045, tf.pow((var_r + 0.055) / 1.055, 2.4),
                     var_r / 12.92)
    var_g = tf.where(var_g > 0.04045, tf.pow((var_g + 0.055) / 1.055, 2.4),
                     var_g / 12.92)
    var_b = tf.where(var_b > 0.04045, tf.pow((var_b + 0.055) / 1.055, 2.4),
                     var_b / 12.92)
    var_r = var_r * 100
    var_g = var_g * 100
    var_b = var_b * 100

    x = var_r * 0.4124 + var_g * 0.3576 + var_b * 0.1805
    y = var_r * 0.2126 + var_g * 0.7152 + var_b * 0.0722
    z = var_r * 0.0193 + var_g * 0.1192 + var_b * 0.9505

    image_xyz = tf.stack([x, y, z], axis=-1)
    return image_xyz


def rgb_to_lab(image: tf.Tensor) -> tf.Tensor:
    """
    Convert an image from RGB color space to LAB color space
    RGB -> XYZ -> LAB
    Parameters
    ----------
    image: tf.Tensor
    Returns
    -------
    tf.tensor: LAB image
    """
    xyz = rgb_to_xyz(image)
    lab_image = xyz_to_lab(xyz)
    return preprocess_lab(lab_image)


def lab_to_rgb(l, ab: tf.Tensor) -> tf.Tensor:
    """
    Convert an image from LAB color space to RGB color space
    LAB -> XYZ -> RGB
    Parameters
    ----------
    image: tf.tensor
    Returns
    -------
    tf.Tensor: RGB image
    """
    lab = deprocess_lab(l, ab)
    xyz = lab_to_xyz(lab)
    rgb_image = xyz_to_rgb(xyz)
    rgb_image = tf.clip_by_value(rgb_image, 0.0, 1.0)
    return rgb_image


def preprocess_lab(lab):
    L_chan = lab[:,:,:,:1]
    ab_chan = lab[:,:,:,1:]
    return L_chan / 100, ab_chan / 110


def deprocess_lab(L_chan, ab_chan):
    return tf.concat([L_chan * 100, ab_chan * 110], axis=-1)


# def tanh_range(l, r, initial=None):
#
#     def get_activation(left, right, initial):
#
#         def activation(x):
#             if initial is not None:
#                 bias = math.atanh(2 * (initial - left) / (right - left) - 1)
#             else:
#                 bias = 0
#             return tanh(x + bias) * (right - left) + left
#
#         return activation
#
#     return get_activation(l, r, initial)

