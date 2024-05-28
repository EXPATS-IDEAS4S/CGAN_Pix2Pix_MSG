# %%
import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
import time
import random

# %%
# define the loss function for the discriminator
def _discriminator_loss(discr_output_real_vis, discr_output_fake_vis):
    """defines the discriminator loss
    Args:
        discr_output_real_vis (tensor): output of discriminator when evaluating the "real" VIS image
        discr_output_fake_vis (tensor): output of discriminator when evaluating the "fake" VIS image produced by the generator
    Returns:
        total_discr_loss: total loss of discriminator
    """
    # specify loss function to use
    loss_function = BinaryCrossentropy(from_logits=True)

    # how close is discriminator output of real VIS image to vector of ones (meaning the discriminator evaluates the image as real)
    real_loss = loss_function(tf.ones_like(discr_output_real_vis), discr_output_real_vis)
    
    # how close is discriminator output of fake VIS image to vector of zeros (meaning the discriminator evaluates the image as fake)
    fake_loss = loss_function(tf.zeros_like(discr_output_fake_vis), discr_output_fake_vis)

    # add both for total discriminator loss
    total_discr_loss = real_loss + fake_loss
    
    return total_discr_loss

# define the loss function for the generator
def _generator_loss(discr_output_fake_vis, fake_vis_img, real_vis_img):
    """defines the generator loss
    Args:
        discr_output_fake_vis (tensor): output of discriminator when evaluating the "fake" VIS image produced by the generator
        fake_vis_img (tensor): "fake" VIS image produced by the generator
        real_vis_img (tensor): corresponding "real" VIS image
    Returns:
        total_gen_loss: total loss of generator
        gan_loss: loss corresponding to how well the generator tricked the discriminator
        l1_loss: loss corresponding to how well the generator recreated the real VIS images
    """
    # define loss function that should be used
    loss_function = BinaryCrossentropy(from_logits=True)

    # how close is discriminator output of fake VIS image to vector of ones (meaning the discriminator evaluates the image as real)
    gan_loss = loss_function(tf.ones_like(discr_output_fake_vis), discr_output_fake_vis)

    # how close is the fake VIS image to the corresponding real VIS image
    l1_loss = tf.reduce_mean(tf.abs(real_vis_img - fake_vis_img))

    # combine both losses to total generator loss
    total_gen_loss = gan_loss + (100 * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


# %%
# define the training procedure
@tf.function
def train_step(ir_img, real_vis_img, generator, discriminator, gen_optimizer, discr_optimizer):
    """procedure during one step of training
    Args:
        ir_img (tf.tensor): infra-red image batch
        real_vis_img (tf.tensor): real visible image batch
        generator (tf.Model): generator model
        discriminator (tf.Model): discriminator model
        gen_optimizer (tf.Optimizer): optimizer to use for generator
        discr_optimizer (tf.Optimizer): optimizer to use for discriminator
    Returns:
        gen_total_loss, discr_loss: generator and discriminator losses
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # generate VIS images with generator
        fake_vis_img = generator(ir_img, training=True)

        # run discriminator once with true VIS image and once with generated VIS image    
        discr_output_real_vis = discriminator([ir_img, real_vis_img], training=True)
        discr_output_fake_vis = discriminator([ir_img, fake_vis_img], training=True)

        # calculate losses
        discr_loss = _discriminator_loss(discr_output_real_vis, discr_output_fake_vis)
        gen_total_loss, gen_gan_loss, gen_l1_loss = _generator_loss(discr_output_fake_vis, fake_vis_img, real_vis_img)

        # calculate gradients of models
        generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(discr_loss, discriminator.trainable_variables)

        # optimize models according to gradients
        gen_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discr_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        # return losses
        return discr_loss, gen_total_loss, gen_gan_loss, gen_l1_loss
    

# %%
# evaluate model
def eval_on_test_images(test_dataset, gen_model):
     # select random batch of test dataset
    #for ir_batch, real_vis_batch in test_dataset.take(1):
    for n, (ir_img, real_vis_img) in test_dataset.enumerate():

        # predict vis img from ir with generator
        fake_vis_img = gen_model(ir_img, training=True) 
        ### Note: The training=True is intentional here since you want the batch statistics,
        ### while running the model on the test dataset. If you use training=False, you get 
        ### the accumulated statistics learned from the training dataset (which you don't want).

    return ir_img, fake_vis_img, real_vis_img
