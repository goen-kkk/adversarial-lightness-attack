import tensorflow as tf
from tensorflow import keras

# import torch
# from torch import nn
# from advertorch.utils import to_one_hot
# import torch.nn.functional as F


class CarliniWagnerLoss(keras.losses.Loss):
    def __init__(self, conf=0.2):
        super(CarliniWagnerLoss, self).__init__()
        self.conf = conf

    def call(self, input, target):
        """
        :param input: pre-softmax/logits.
        :param target: true labels.
        :return: CW loss value.
        """
        if input.shape[1] == 1:
            l_0 = 1. - input
            new_input = tf.concat([l_0, input], 1)
            num_classes = new_input.shape[1]
            label_mask = tf.one_hot(target, depth=num_classes)
            # print(label_mask)
            correct_logit = tf.reduce_sum(label_mask * new_input, axis=1)
            # print(correct_logit)
            wrong_logit = tf.reduce_max((1. - label_mask) * new_input, axis=1)
            # print(wrong_logit)
            loss = tf.reduce_sum(tf.nn.relu(correct_logit - wrong_logit + self.conf))
        else:
            num_classes = input.shape[1]
            label_mask = tf.one_hot(target, num_classes=num_classes)
            correct_logit = tf.reduce_sum(label_mask * input, axis=1)
            wrong_logit = tf.reduce_max((1. - label_mask) * input, axis=1)
            loss = tf.reduce_sum(-tf.nn.relu(correct_logit - wrong_logit + self.conf))
        return loss


class RegularizationLoss():

    def __init__(self, T):
        super(RegularizationLoss, self).__init__()
        self.T = T

    def __call__(self, theta):
        loss = -tf.reduce_sum(tf.abs(theta)) / self.T
        return loss


# https://github.com/BorealisAI/advertorch/blob/master/advertorch/utils.py
# class CarliniWagnerLossTorch(nn.Module):
#     """
#     Carlini-Wagner Loss: objective function #6.
#     Paper: https://arxiv.org/pdf/1608.04644.pdf
#     """
#     def __init__(self, conf=50.):
#         super(CarliniWagnerLoss, self).__init__()
#         self.conf = conf
#
#     def forward(self, input, target):
#         """
#         :param input: pre-softmax/logits.
#         :param target: true labels.
#         :return: CW loss value.
#         """
#         # num_classes = input.size(1)
#         l_0 = 1. - input
#         new_input = tf.concat([l_0, input], 1)
#         num_classes = new_input.shape[1]
#         label_mask = to_one_hot(target, num_classes=num_classes).float()
#         correct_logit = torch.sum(label_mask * input, dim=1)
#         wrong_logit = torch.max((1. - label_mask) * input, dim=1)[0]
#         loss = -F.relu(correct_logit - wrong_logit + self.conf).sum()
#         return loss
#
#
# class RegularizationLossTorch():
#
#     def __init__(self, T):
#         super(RegularizationLoss, self).__init__()
#         self.T = T
#
#     def __call__(self, theta):
#         loss = -torch.sum(torch.abs(theta)) / self.T
#         return loss