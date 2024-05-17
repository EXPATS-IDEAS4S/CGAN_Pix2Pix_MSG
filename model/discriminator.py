# %%
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU
from keras.layers import Conv2DTranspose, Dropout, ReLU, Input, Concatenate, ZeroPadding2D
from keras.optimizers import Adam
from keras.utils import plot_model

# %%
# downsample block (same as in generator)
def downsample(filters, size, batchnorm=True):
    init = tf.random_normal_initializer(0.,0.02)
    result = Sequential()
    result.add(Conv2D(filters, size, strides=2, padding="same", 
                      kernel_initializer=init, use_bias=False))
    if batchnorm == True:
        result.add(BatchNormalization())
        
    result.add(LeakyReLU())
    return result

def discriminator(image_size=128, image_channels=1, kernel_size=4):
    init = tf.random_normal_initializer(0., 0.02)
    
    inp = Input(shape = [image_size, image_size, image_channels], name="input_image")
    tar = Input(shape = [image_size, image_size, image_channels], name="target_image")
    x = Concatenate()([inp, tar])  # (batch_size, 128, 128, 2)

    down1 = downsample(64, kernel_size, batchnorm=False)(x)  # (batch_size, 64, 64, 64)
    down2 = downsample(128, kernel_size)(down1)  # (batch_size, 32, 32, 128)
    down3 = downsample(256, kernel_size)(down2)  # (batch_size, 16, 16, 256)
    
    zero_pad1 = ZeroPadding2D()(down3)  # (batch_size, 18, 18, 256)
    conv = Conv2D(256, kernel_size, strides=1, 
                  kernel_initializer=init, 
                  use_bias=False)(zero_pad1)  # (batch_size, 15, 15, 256)
    leaky_relu = LeakyReLU()(conv)  # (batch_size, 15, 15, 256)

    zero_pad2 = ZeroPadding2D()(leaky_relu)  # (batch_size, 17, 17, 256)
    last = Conv2D(1, kernel_size, strides=1, 
                  kernel_initializer=init)(zero_pad2)  # (batch_size, 14, 14, 1)

    return Model(inputs=[inp, tar], outputs=last)

# %% 
if __name__ == '__main__':
    IMAGE_SIZE = 128
    IMAGE_CHANNELS = 1

    # path to project directory
    PROJECT_DIR = "/net/merisi/pbigalke/teaching/METFUT2024/CGAN_Pix2Pix_MSG"

    disc = discriminator(image_size=IMAGE_SIZE, image_channels=IMAGE_CHANNELS)
    disc.summary()
    model_overview_path = f"{PROJECT_DIR}/output/model/architecture_discriminator.png"
    plot_model(disc, to_file=model_overview_path, show_shapes=True)


# %%
