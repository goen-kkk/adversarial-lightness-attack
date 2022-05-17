import numpy as np
from classifiers import *
from pipeline import *
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1 - Load the model and its pretrained weights
classifier = Meso4()
classifier.load('weights/Meso4_DF.h5')
# classifier.trainable = False

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)
batch_size = 1
alpha = 0.01  # 0.5
epsilon = 0.05
dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
        '2_face_img/real/',
        target_size=(256, 256),
        batch_size=30,
        class_mode=None)



def fgsm(model, x, y, loss_fn, epsilon=epsilon):
    with tf.GradientTape() as gt:
        gt.watch(x)
        loss = loss_fn(model(x), y)
        print(loss)
    grad = gt.gradient(loss, x)
    noise = epsilon * tf.sign(grad)
    return noise

def ifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20):
    x_adv = x
    for i in range(num_iter):
        noise = fgsm(model, x_adv, y, loss_fn, alpha)
        x_adv = x_adv + noise
        diff = x_adv - x
        diff = tf.clip_by_value(diff, -epsilon, epsilon)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    return x_adv


def gen_adv_examples(model, generator, attack, loss_fn):
    # model.eval()
    # adv_names = []
    train_acc, train_loss = 0.0, 0.0
    pred = []
    for i, x in enumerate(generator):
        y = np.array([0]*batch_size)
        # noise = fgsm(model, x, y, loss_fn)
        # x_adv = x + noise
        x_adv = attack(model, x, y, loss_fn)
        yp = model.predict(x_adv)
        pred += np.where(yp < 0.5, 0, 1).squeeze().tolist()
        train_acc += np.sum(yp == y)
        train_loss += loss_fn(yp, y)
        adv_ex = tf.clip_by_value(x_adv, 0, 1)
        adv_ex = tf.clip_by_value(adv_ex*255, 0, 255)
        adv_ex = tf.reshape(adv_ex, (batch_size, 256, 256, 3))
        adv_examples = adv_ex if i == 0 else np.r_[adv_examples, adv_ex]



def epoch_benign(model, generator, loss_fn):
    train_acc, train_loss = 0.0, 0.0
    pred = []
    i = 0
    for i, x in enumerate(generator):
        if i > 10:
          break
        y = np.array([1.]*batch_size)
        yp = model.predict(x)
        loss = loss_fn(yp, y)
        print(yp)
        yp = np.where(yp < 0.5, 0, 1).squeeze().tolist()
        print(yp)
        pred += yp
        train_acc += np.sum(yp == y)
        train_loss += loss
    # return train_acc / len(loader.dataset), train_loss / len(loader.dataset)
    return pred, train_acc / len(pred), train_loss / len(pred)


if __name__ == '__main__':
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    # X = generator.next()
    # print('Predicted :', classifier.predict(X), '\nReal class :', y)
    pred, _, _ = epoch_benign(classifier, generator, loss_fn)
    print(pred)

