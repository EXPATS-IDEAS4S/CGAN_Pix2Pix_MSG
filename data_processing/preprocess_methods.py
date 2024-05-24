# %%
import tensorflow as tf
import numpy as np
import random

# image size DO NOT CHANGE
IMG_SIZE = 128

# %%
# put methods here to normalize and augment the dataset

# %%
def normalize(input_image, real_image):
    input_image = (tf.cast(input_image, tf.float32) / 127.5) - 1.0 # [-1 to 1] range
    real_image = (tf.cast(real_image, tf.float32) / 127.5) - 1.0 # [-1 to 1] range
    return input_image, real_image

def resize(input_image, real_image):
    input_image = tf.image.resize(input_image, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image

def random_jitter(input_image, real_image):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    return input_image, real_image
