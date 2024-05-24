# %%
import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
import time
import random

# %%
# define the loss functions
def generator_loss(discr_generated_output, gen_output, target):
    
    loss_function = BinaryCrossentropy(from_logits=True)

    gan_loss = loss_function(tf.ones_like(discr_generated_output), discr_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (100 * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    
    loss_function = BinaryCrossentropy(from_logits=True)

    real_loss = loss_function(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_function(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    
    return total_disc_loss

# %%
# define the training process
@tf.function
def train_step(input_image, target, generator, discriminator, gen_optimizer, discr_optimizer):
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
        discr_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        # return losses
        return gen_total_loss, disc_loss
    

def save_model_state(model, epoch):
    #TODO: implement method that saves the model state for this epoch
    return None

# %%
# evaluate model
def eval_example_images(dataset, gen_model, batchsize):
     # select random batch of test dataset
    for ir_batch, vis_batch in dataset.take(1):

        # select random image of batch
        rand_idx = random.randint(0, batchsize-1)
        ir_img = ir_batch.numpy()[rand_idx]
        vis_img = vis_batch.numpy()[rand_idx]

        # predict vis img from ir with generator
        predict_vis_batch = gen_model(ir_batch, training=True) 
        ### Note: The training=True is intentional here since you want the batch statistics,
        ### while running the model on the test dataset. If you use training=False, you get 
        ### the accumulated statistics learned from the training dataset (which you don't want).
        predict_vis_img = predict_vis_batch.numpy()[rand_idx]

        return ir_img, predict_vis_img, vis_img
    

def pick_image_pair_from_dataset(dataset, batchsize):
     # select random batch of test dataset
    for ir_batch, vis_batch in dataset.take(1):

        # select random image of batch
        rand_idx = random.randint(0, batchsize-1)
        ir_img = ir_batch.numpy()[rand_idx]
        vis_img = vis_batch.numpy()[rand_idx]

        return ir_img, vis_img