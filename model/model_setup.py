# %%
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Conv2D, BatchNormalization, LeakyReLU
from keras.layers import Conv2DTranspose, Dropout, ReLU, Input, Concatenate, ZeroPadding2D
from keras.utils import plot_model

# %%
# downsample block
def downsample(filters, size, batchnorm = True):
    init = tf.random_normal_initializer(0.,0.02)
    result = Sequential()
    result.add(Conv2D(filters, size, strides=2, padding="same", 
                      kernel_initializer=init, use_bias=False))
    if batchnorm == True:
        result.add(BatchNormalization())
        
    result.add(LeakyReLU())
    return result


# upsample block
def upsample(filters, size, dropout = False):
    init = tf.random_normal_initializer(0, 0.02)
    result = Sequential()
    result.add(Conv2DTranspose(filters, size, strides=2, padding="same", 
                               kernel_initializer=init, use_bias=False))
    result.add(BatchNormalization())
    if dropout == True:
        result.add(Dropout(0.5))
    result.add(ReLU())
    return result


def generator(image_size=128, image_channels=1, kernel_size=4):
    inputs = Input(shape = [image_size, image_size, image_channels])  # (batch_size, 128, 128, 1)
    print("input layer", inputs.shape)
    down_stack = [
        downsample(64, kernel_size, batchnorm=False),  # (batch_size, 64, 64, 64) 
        downsample(128, kernel_size),  # (batch_size, 32, 32, 128) 
        downsample(256, kernel_size),  # (batch_size, 16, 16, 256) 
        downsample(512, kernel_size),  # (batch_size, 8, 8, 512) 
        downsample(512, kernel_size),  # (batch_size, 4, 4, 512)
        downsample(512, kernel_size),  # (batch_size, 2, 2, 512) 
        downsample(512, kernel_size),  # (batch_size, 1, 1, 512)  
    ]
    
    
    up_stack = [
        upsample(512, kernel_size, dropout=True),  # (batch_size, 2, 2, 512) 
        upsample(512, kernel_size, dropout=True),  # (batch_size, 4, 4, 512)
        upsample(512, kernel_size),  # (batch_size, 8, 8, 512) 
        upsample(256, kernel_size),  # (batch_size, 32, 32, 256)  
        upsample(128, kernel_size),  # (batch_size, 64, 64, 128)  
        upsample(64, kernel_size),  # (batch_size, 
    ]
    init = tf.random_normal_initializer(0., 0.02)
    last = Conv2DTranspose(image_channels, kernel_size, strides=2, 
                           padding="same", kernel_initializer=init, 
                           activation="tanh")  # (batch_size, 
    x = inputs

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
        print("down", x.shape)
    skips = reversed(skips[:-1])
    
    for up, skip in zip(up_stack, skips):
        x = up(x)
        print("up", x.shape)
        x = Concatenate()([x, skip])
        print("up + skip", x.shape)
    
    x = last(x)
    print("last", x.shape)
    return Model(inputs=inputs, outputs=x)

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

    gen = generator(image_size=IMAGE_SIZE, image_channels=IMAGE_CHANNELS)
    gen.summary()
    model_overview_path = f"{PROJECT_DIR}/output/model/architecture_generator.png"
    plot_model(gen, to_file=model_overview_path, show_shapes=True)

    disc = discriminator(image_size=IMAGE_SIZE, image_channels=IMAGE_CHANNELS)
    disc.summary()
    model_overview_path = f"{PROJECT_DIR}/output/model/architecture_discriminator.png"
    plot_model(disc, to_file=model_overview_path, show_shapes=True)



# %%
