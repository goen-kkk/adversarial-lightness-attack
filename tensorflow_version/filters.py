import tensorflow as tf
# import torch
import numpy as np
# import tensorflow.contrib.layers as ly
# from utils import lerp
import cv2
import math


class Filter:

    # def __init__(self, net, cfg):
    def __init__(self, cfg):
        self.cfg = cfg
        # self.height, self.width, self.channels = list(map(int, net.get_shape()[1:]))
        self.height, self.width, self.channels = cfg.img_size, cfg.img_size, cfg.img_channels
        # Specified in child classes
        self.num_filter_parameters = None
        self.short_name = None
        self.filter_parameters = None

    def get_short_name(self):
        assert self.short_name
        return self.short_name

    def get_num_filter_parameters(self):
        assert self.num_filter_parameters
        return self.num_filter_parameters

    # def extract_parameters(self, features):
    #     output_dim = self.get_num_filter_parameters(
    #     ) + self.get_num_mask_parameters()
    #     features = ly.fully_connected(
    #         features,
    #         self.cfg.fc1_size,
    #         scope='fc1',
    #         activation_fn=lrelu,
    #         weights_initializer=tf.contrib.layers.xavier_initializer())
    #     features = ly.fully_connected(
    #         features,
    #         output_dim,
    #         scope='fc2',
    #         activation_fn=None,
    #         weights_initializer=tf.contrib.layers.xavier_initializer())
    #     return features[:, :self.get_num_filter_parameters()], \
    #            features[:, self.get_num_filter_parameters():]

    # Should be implemented in child classes
    def filter_param_regressor(self, features):
        assert False

    # Process the whole image, without masking
    # Should be implemented in child classes
    def process(self, img, param):
        assert False

    def debug_info_batched(self):
        return False

    def no_high_res(self):
        return False

    # Apply the whole filter with masking
    '''
    def apply(self,
                img,
                img_features=None,
                specified_parameter=None,
                high_res=None):
    '''
    def apply(self, img, filter_parameters):

        # assert (img_features is None) ^ (specified_parameter is None)
        # if img_features is not None:
        #     filter_features, mask_parameters = self.extract_parameters(img_features)
        #     filter_parameters = self.filter_param_regressor(filter_features)
        # else:
        #     assert not self.use_masking()
        #     filter_parameters = specified_parameter
        #     mask_parameters = tf.zeros(
        #         shape=(1, self.get_num_mask_parameters()), dtype=np.float32)


        # if high_res is not None:
        #     # working on high res...
        #     pass
        filter_parameters = self.filter_param_regressor(filter_parameters)
        # debug_info = {}
        # We only debug the first image of this batch
        # if self.debug_info_batched():
        #     debug_info['filter_parameters'] = filter_parameters
        # else:
        #     debug_info['filter_parameters'] = filter_parameters[0]
        # self.mask_parameters = mask_parameters
        # self.mask = self.get_mask(img, mask_parameters)

        # self.mask = tf.ones(shape=(1, 1, 1, 1), dtype=tf.float32)
        # debug_info['mask'] = self.mask[0]
        # low_res_output = lerp(img, self.process(img, filter_parameters), self.mask)
        # if high_res is not None:
        #     if self.no_high_res():
        #         high_res_output = high_res
        #     else:
        #         self.high_res_mask = self.get_mask(high_res, mask_parameters)
        #         high_res_output = lerp(high_res,
        #                                self.process(high_res, filter_parameters),
        #                                self.high_res_mask)
        # else:
        #     high_res_output = None
        # return low_res_output, high_res_output, debug_info
        return self.process(img, filter_parameters)

    # def use_masking(self):
    #     return self.cfg.masking

    # def get_num_mask_parameters(self):
    #     return 6

  # Input: no need for tanh or sigmoid
  # Closer to 1 values are applied by filter more strongly
  # no additional TF variables inside
  #   def get_mask(self, img, mask_parameters):
  #       if not self.use_masking():
  #           print('* Masking Disabled')
  #           return tf.ones(shape=(1, 1, 1, 1), dtype=tf.float32)
  #       else:
  #           print('* Masking Enabled')
  #       with tf.name_scope(name='mask'):
  #           # Six parameters for one filter
  #           filter_input_range = 5
  #           assert mask_parameters.shape[1] == self.get_num_mask_parameters()
  #           mask_parameters = tanh_range(
  #               l=-filter_input_range, r=filter_input_range,
  #               initial=0)(mask_parameters)
  #           size = list(map(int, img.shape[1:3]))
  #           grid = np.zeros(shape=[1] + size + [2], dtype=np.float32)
  #
  #           shorter_edge = min(size[0], size[1])
  #           for i in range(size[0]):
  #               for j in range(size[1]):
  #                   grid[0, i, j,
  #                       0] = (i + (shorter_edge - size[0]) / 2.0) / shorter_edge - 0.5
  #                   grid[0, i, j,
  #                       1] = (j + (shorter_edge - size[1]) / 2.0) / shorter_edge - 0.5
  #           grid = tf.constant(grid)
  #           # Ax + By + C * L + D
  #           inp = grid[:, :, :, 0, None] * mask_parameters[:, None, None, 0, None] + \
  #                   grid[:, :, :, 1, None] * mask_parameters[:, None, None, 1, None] + \
  #                   mask_parameters[:, None, None, 2, None] * (rgb2lum(img) - 0.5) + \
  #                   mask_parameters[:, None, None, 3, None] * 2
  #           # Sharpness and inversion
  #           inp *= self.cfg.maximum_sharpness * mask_parameters[:, None, None, 4,
  #                                                             None] / filter_input_range
  #           mask = tf.sigmoid(inp)
  #           # Strength
  #           mask = mask * (
  #               mask_parameters[:, None, None, 5, None] / filter_input_range * 0.5 +
  #               0.5) * (1 - self.cfg.minimum_strength) + self.cfg.minimum_strength
  #           print('mask', mask.shape)
  #       return mask

    def visualize_filter(self, debug_info, canvas):
        # Visualize only the filter information
        assert False

    def visualize_mask(self, debug_info, res):
        return cv2.resize(
            debug_info['mask'] * np.ones((1, 1, 3), dtype=np.float32),
            dsize=res,
            interpolation=cv2.cv2.INTER_NEAREST)

    def draw_high_res_text(self, text, canvas):
        cv2.putText(
            canvas,
            text, (30, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 0),
            thickness=5)
        return canvas


class LightnessFilter(Filter):

    # def __init__(self, net, cfg):
    #     Filter.__init__(self, net, cfg)
    def __init__(self, cfg):
        Filter.__init__(self, cfg)
        self.curve_steps = cfg.curve_steps
        self.channels = cfg.img_channels
        self.short_name = 'L'
        self.num_filter_parameters = self.channels * cfg.curve_steps

    def filter_param_regressor(self, features):
        '''
        [B, filter_params] -> [B, 1, 1, channels, curve_steps]
        '''
        lab_curve = tf.reshape(
            features, shape=(-1, self.channels,
                             self.cfg.curve_steps))[:, None, None, :]
    #     lab_curve = tanh_range(
    #         *self.cfg.lab_curve_range, initial=1)(lab_curve)
        return lab_curve

    def process(self, img, param):
        '''Unrestricted ALA with Contrast Relaxation'''
        lab_curve = param
        # There will be no division by zero here unless the color filter range lower bound is 0
        lab_curve_sum = tf.reduce_sum(param, axis=4) + 1e-30

        max = tf.reduce_max(img)
        min = tf.reduce_min(img)
        total_image = img * 0
        for i in range(self.cfg.curve_steps):
            total_image += tf.clip_by_value(img - min - (max - min) * i / self.cfg.curve_steps, 0, (max - min) / self.cfg.curve_steps) * \
                           lab_curve[:, :, :, :, i]
            # total_image += tf.clip_by_value(img * i / self.cfg.curve_steps, 0, 1.0 / self.cfg.curve_steps) * \
            #                lab_curve[:, :, :, :, i]
            # print(tf.reduce_max(total_image), tf.reduce_min(total_image))
        # total_image *= self.cfg.curve_steps / lab_curve_sum
        total_image = total_image * (self.cfg.curve_steps / lab_curve_sum) + min
        total_image = tf.clip_by_value(total_image, min, max)
        # print(tf.reduce_max(img), tf.reduce_min(img), tf.reduce_max(total_image), tf.reduce_min(total_image))
        return total_image

    def visualize_filter(self, debug_info, canvas):
        curve = debug_info['filter_parameters']
        height, width = canvas.shape[:2]
        for i in range(self.channels):
            values = np.array([0] + list(curve[0][0][i]))
            values /= sum(values) + 1e-30
            scale = 1
            values *= scale
            for j in range(0, self.cfg.curve_steps):
                values[j + 1] += values[j]
            for j in range(self.cfg.curve_steps):
                p1 = tuple(
                    map(int, (width / self.cfg.curve_steps * j, height - 1 -
                              values[j] * height)))
                p2 = tuple(
                    map(int, (width / self.cfg.curve_steps * (j + 1), height - 1 -
                              values[j + 1] * height)))
                color = []
                for t in range(self.channels):
                    color.append(1 if t == i else 0)
                cv2.line(canvas, p1, p2, tuple(color), thickness=1)


# class LightnessFilterTorch():
#     def __init__(self, cfg):
#         self.curve_steps = cfg.curve_steps
#         self.channels = cfg.img_channels
#         self.short_name = 'L'
#         self.num_filter_parameters = self.channels * cfg.curve_steps
#
#     def filter_param_regressor(self, features):
#         '''
#         [B, filter_params] -> [B, 1, 1, channels, curve_steps]
#         '''
#         lab_curve = torch.reshape(
#             features, shape=(-1, self.channels,
#                              self.curve_steps))[:, None, None, :]
#         return lab_curve
#
#     def process(self, img, param):
#         '''Unrestricted ALA with Contrast Relaxation'''
#         lab_curve = param
#         # There will be no division by zero here unless the color filter range lower bound is 0
#         lab_curve_sum = torch.sum(param, dim=4) + 1e-30
#
#         max = torch.max(img)
#         min = torch.min(img)
#         total_image = img * 0
#         for i in range(self.curve_steps):
#             total_image += torch.clamp(img - min - (max - min) * i / self.curve_steps, 0, (max - min) / self.curve_steps) * \
#                            lab_curve[:, :, :, :, i]
#         total_image = total_image * (self.curve_steps / lab_curve_sum) + min
#         total_image = torch.clamp(total_image, min, max)
#         return total_image
