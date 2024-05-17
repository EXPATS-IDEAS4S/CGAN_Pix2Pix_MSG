# %%
import tensorflow as tf
import numpy as np

# %%
def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, channels=1)
    print(tf.shape(image))
    return
    
    #image = tf.image.decode_jpeg(image, channels=3)
    #w = tf.shape(image)[1]
    #w = w//2
    #real_image = image[:, :w, :]
    #input_image = image[:, w:, :]
   # 
    #input_image = tf.cast(input_image, tf.float32)
    #real_image = tf.cast(real_image, tf.float32)
    #return input_image, real_image

# %%
# path to project directory
PROJECT_DIR = "/net/merisi/pbigalke/teaching/METFUT2024/CGAN_Pix2Pix_MSG"

# load example image
example_path = f"{PROJECT_DIR}/test_data/"

input_image_file = f"{example_path}IR_108/20230701_1012_IR_108.png"    
input_image = tf.io.read_file(input_image_file)
input_image = tf.image.decode_png(input_image, channels=IMAGE_CHANNELS)
input_image = tf.cast(input_image, tf.float32)
print(input_image.shape)

real_image_file = f"{example_path}VIS006/20230701_1012_VIS006.png"
real_image = tf.io.read_file(real_image_file)
real_image = tf.image.decode_png(real_image, channels=IMAGE_CHANNELS)
real_image = tf.cast(real_image, tf.float32)
# %%
