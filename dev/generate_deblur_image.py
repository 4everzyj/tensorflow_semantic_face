# coding: utf-8

import model
import cv2
import numpy as np
import tensorflow as tf
import os


def im2uint8(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)


def prepare_image(img_path):
    if not os.path.isdir('../out/resize'):
        os.makedirs('../out/resize')
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('../out/resize/resize_%s' % img_path.rsplit('/', 1)[-1], img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float) / 255.0
    print(img.shape)
    half_img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    img = img.reshape((1, 128, 128, 3))
    half_img = half_img.reshape((1, 64, 64, 3))
    return img, half_img


def generate_deblur_image(pure_name, img, half_img):
    with tf.Session() as sess:
        image = tf.placeholder("float", [1, 128, 128, 3])
        halfimg = tf.placeholder("float", [1, 64, 64, 3])
        feed_dict = {image: img, halfimg: half_img}

        SF = model.Semantic_face('../net_P_P_S_F.mat', '../net_G_P_S_F.mat')
        with tf.name_scope("Semantic_face"):
            SF.build(image, halfimg)
        out = sess.run(SF.convG32, feed_dict=feed_dict)
        print(out.shape)
        out = out[0]
        out = im2uint8(out)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite('../out/deblur_%s.jpg' % pure_name, out)


if __name__ == '__main__':
    img_dir = '/hdd/data/training_images/face'
    img_list = os.listdir(img_dir)
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        img, half_img = prepare_image(img_path)
        pure_name = img_name.rsplit('.', 1)[0]
        generate_deblur_image(pure_name, img, half_img)
