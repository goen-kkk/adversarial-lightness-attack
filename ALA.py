import tensorflow as tf
import numpy as np
import os
# import skimage.io as io
import random
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from filters import *
from utils import *
from config import cfg
from  Meso_DFDC import MesoInception4, build_data


def attack(L, ab, target, params, cfg, filter):
    with tf.GradientTape(persistent=True) as gt:
        gt.watch(params)
        L_adv = filter.apply(L, params)
        x_adv = lab_to_rgb(L_adv,ab)
        '''
        vid_adv = frame2video(x_adv) # 这两个过程需要存取文件，不是tensor形式
        input = func(vid_adv)      # 这两个过程需要存取文件，不是tensor形式
        loss = loss_func(model(input), y) # loss可以得到对params的梯度吗
        '''
        CWloss = cfg.CWloss(cfg.model(x_adv), target)
        RLloss = cfg.beta * cfg.RLloss(params)
        loss = CWloss + RLloss
        print(loss)
    grad = gt.gradient(loss, params)
    # print(grad)
    params_adv = params - cfg.alpha * grad / (tf.norm(grad, axis=1, keepdims=True) + 1e-30)
    # params_adv = tf.clip_by_value(params_adv, cfg.m, cfg.n)
    return params_adv, x_adv


# def process(generator, cfg, path, list):
#     T = cfg.curve_steps
#     filter = LightnessFilter(cfg)
#     idx = 0
#     for img in generator:
#         # 将图像进行颜色空间转换
#         # img = img.astype('float32')
#         L, ab = rgb_to_lab(img)
#         params = np.array([[random.uniform(cfg.m, cfg.n) for _ in range(T)] for i in range(img.shape[0])])
#         params = tf.convert_to_tensor(params, dtype=tf.float32)
#         x_filtered = None
#         x_adv = tf.zeros_like(img)
#         y = np.array([1.]*img.shape[0])
#         for i in range(cfg.iter_n):
#             params, x_adv = attack(L,ab,y,params,cfg,filter)
#             yp = cfg.model(x_adv)
#             yp = np.where(yp < 0.5, 0, 1).squeeze().tolist()
#             # print(yp)
#             if np.sum(yp == y) / cfg.batch_size < 0.1:
#                 print("attack successfully!")
#                 # break
#         x_filtered = x_adv
#
#         for image in x_filtered:
#             if idx >= len(list):
#                 break
#             # savepath = path + list[str(idx)]
#             savepath = path + list[idx]['path']
#             if not os.path.exists(savepath[:-8]):
#                 os.makedirs(savepath[:-8])
#             rgb_image = tf.cast(image * 255, tf.uint8)
#             im = Image.fromarray(np.array(rgb_image).astype(np.uint8))
#             im = im.resize(size=list[idx]['shape'])
#             im.save(savepath)
#             idx += 1
#         print("done")

            # # [batch_size, H, W, C]
            # for img in filtered_images:
            #     print('img', img.shape)
            # filtered_images = tf.stack(values=filtered_images, axis=1)
            # print('    filtered_images:', filtered_images.shape)
    # return x_filtered
    #     break
    #     if idx >= len(list):
    #         break


def read_list(file):
    file_info = np.load(file, allow_pickle=True)
    # reader = csv.reader(csvFile)
    # file_list = {}
    # for item in reader:
    #     file_list[item[0]] = item[1]
    #
    # csvFile.close()
    return file_info


def process(img, y, cfg, path, list):
    T = cfg.curve_steps
    filter = LightnessFilter(cfg)
    idx = 0
    # 将图像进行颜色空间转换
    # img = img.astype('float32')
    L, ab = rgb_to_lab(img)
    params = np.array([[random.uniform(cfg.m, cfg.n) for _ in range(T)] for i in range(img.shape[0])])
    params = tf.convert_to_tensor(params, dtype=tf.float32)
    x_filtered = None
    x_adv = tf.zeros_like(img)
    y = np.array([1.]*img.shape[0])
    for i in range(cfg.iter_n):
        params, x_adv = attack(L,ab,y,params,cfg,filter)
        yp = cfg.model(x_adv)
        yp = np.where(yp < 0.5, 0, 1).squeeze().tolist()
        # print(yp)
        if np.sum(yp == y) / cfg.batch_size < 0.1:
            print("attack successfully!")
            # break
    x_filtered = x_adv

    for image in x_filtered:
        if idx >= len(list):
            break
        # savepath = path + list[str(idx)]
        savepath = path + list[idx]['path']
        if not os.path.exists(savepath[:-8]):
            os.makedirs(savepath[:-8])
        rgb_image = tf.cast(image * 255, tf.uint8)
        im = Image.fromarray(np.array(rgb_image).astype(np.uint8))
        im = im.resize(size=list[idx]['shape'])
        im.save(savepath)
        idx += 1
    print("done")


if __name__ =='__main__':
    # dataGenerator = ImageDataGenerator()
    # generator = dataGenerator.flow_from_directory(
    #       '2_face_img/fake',
    #       target_size=(256, 256),
    #       batch_size=cfg.batch_size,
    #       class_mode=None,
    #       subset='training',
    #       shuffle=False)
    # # img = io.imread('2_face_img/real/000/0000.jpg')
    # savedir = 'filtered_img/tmp'
    # if not os.path.exists(savedir):
    #     os.makedirs(savedir)
    # file_list = read_list('file.npy')
    # print(file_list[0]['path'])
    # process(generator, cfg, savedir, file_list)
    input_path = './DFDC/'
    output_path = './ACE_dfdc/'
    meso = MesoInception4()

    image_id_list = []
    for p in os.listdir(input_path):
        if not os.path.isdir(os.path.join(output_path, p)):
            os.makedirs(os.path.join(output_path, p))
        path = input_path + p
        image_id = list(filter(lambda x: '.png' in x, os.listdir(path)))
        for i in range(len(image_id)):
            image_id_list.append(p + '/' + image_id[i])

    # print(image_id_list[0])
    num_batches = np.int(np.ceil(len(image_id_list) / cfg.batch_size))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for k in range(0, num_batches):
        batch_size_cur = min(cfg.batch_size, len(image_id_list) - k * cfg.batch_size)
        for i in range(batch_size_cur):
            meso.data = []
            meso.y = []
            meso.load_data(input_path+image_id_list[k*cfg.batch_size+i], 'FAKE')

        x = meso.data
        y = meso.y
        process(x, y, cfg, output_path, image_id_list)

    