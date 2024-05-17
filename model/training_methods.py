# %%
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU
from keras.layers import Conv2DTranspose, Dropout, ReLU, Input, Concatenate, ZeroPadding2D
from keras.optimizers import Adam
from keras.utils import plot_model
import mapplotlib.pyplot as plt
from keras.losses import BinaryCrossentropy
import time

# %%
# define the loss functions
def generator_loss(disc_generated_output, gen_output, target):
    
    loss_function = BinaryCrossentropy(from_logits=True)

    gan_loss = loss_function(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (100 * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    
    loss_function = BinaryCrossentropy(from_logits=True)

    real_loss = loss_function(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_function(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    
    return total_disc_loss

# %% saving intermediate states during training
def save_images(model, test_input, target, epoch):
    prediction = model(test_input, training= True)
    plt.figure(figsize = (15,15))
    display_list= [test_input[0], target[0], prediction[0]]
    title = ["Input Image", "Ground Truth", "Predicton Image"]
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")
    plt.savefig(f"output/epoch_{epoch}.jpg")
    plt.close()

def save_model_state(model, epoch):
    #TODO: implement method that saves the model state for this epoch
    return None

# %%
# define the training process
@tf.function
def train_step(input_image, target, generator, discriminator, gen_optimizer, disc_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # generate VIS images with generator
        gen_output = generator(input_image, training=True)

        # run discriminator once with true VIS image and once with generated VIS image    
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        # calculate losses
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        # calculate gradients of models
        generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        # optimize models according to gradients
        gen_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        # return losses
        return gen_total_loss, disc_loss
    
def fit(generator, discriminator, train_dataset, test_dataset, epochs, save_each_epoch=5):

    # create optimizers for generator and discriminator
    gen_optimizer = Adam(lr=2e-4, beta_1=0.5)
    disc_optimizer = Adam(lr=2e-4, beta_1=0.5)

    # loop over number of epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        start = time.time()

        # save example images of this epoch
        for input_image, target in test_dataset.take(1):
            save_images(generator, input_image, target, epoch)
        
        # loop over batches in training dataset
        for n, (input_image, target) in train_dataset.enumerate():
            # perform training step
            gen_loss, disc_loss = train_step(input_image, target, generator, discriminator, 
                                             gen_optimizer, disc_optimizer)
        
        # print some information on the progress of training
        print(f"Generator loss: {gen_loss:.2f}, Discriminator loss: {disc_loss:.2f}")
        print(f"Time for epoch {epoch+1}: {(time.time()-start)/60.} min.")
        print(f"Total runtime: {time.time()-start/60./60.} hours.")
    
# %% 
if __name__ == '__main__':
    # test training here

    print("I dont know how to test it here.")
