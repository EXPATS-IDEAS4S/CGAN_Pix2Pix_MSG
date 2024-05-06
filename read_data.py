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
image_path = "/net/merisi/pbigalke/teaching/METFUT2024/GANS/Pix2Pix/test_data/2023/"
test_image_imshow = f"{image_path}07_imshow/20230709_1012_IR_108_size128.png"
test_image_imsave = f"{image_path}07_imsave/20230709_1012_IR_108_size128.png"
print(image_path)

print("loading imshow file")
load(test_image_imshow)

print("loading imsave file")
load(test_image_imsave)
# %%
