# %%
import tensorflow as tf
from keras.losses import BinaryCrossentropy
import csv
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
    discr_loss = real_loss + fake_loss
    
    return discr_loss

# define the loss function for the generator
def _generator_loss(discr_output_fake_vis, fake_vis_batch, real_vis_batch):
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

    # fooling the discriminator: 
    # how close is discriminator output of fake VIS image to vector of ones (meaning the discriminator evaluates the image as real)
    fooling_discr_loss = loss_function(tf.ones_like(discr_output_fake_vis), discr_output_fake_vis)

    # how close is the fake VIS image to the corresponding real VIS image
    real_vis_similarity_loss = tf.reduce_mean(tf.abs(real_vis_batch - fake_vis_batch))

    # combine both losses to total generator loss
    total_gen_loss = fooling_discr_loss + (100 * real_vis_similarity_loss)

    return total_gen_loss, fooling_discr_loss, real_vis_similarity_loss


# method to perform one training step: 
# - generate images of one batch, calculate losses and the model gradient, update the models' parameter
@tf.function
def train_step(ir_batch, real_vis_batch, generator, discriminator, gen_optimizer, discr_optimizer):
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
        fake_vis_batch = generator(ir_batch, training=True)

        # run discriminator once with true VIS image and once with generated VIS image    
        discr_output_real_vis = discriminator([ir_batch, real_vis_batch], training=True)
        discr_output_fake_vis = discriminator([ir_batch, fake_vis_batch], training=True)

        # calculate losses
        discr_loss = _discriminator_loss(discr_output_real_vis, discr_output_fake_vis)
        total_gen_loss, fooling_discr_loss, real_vis_similarity_loss = _generator_loss(discr_output_fake_vis, fake_vis_batch, real_vis_batch)

        # calculate gradients of models
        generator_gradients = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(discr_loss, discriminator.trainable_variables)

        # optimize models according to gradients
        gen_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discr_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        # return losses
        return discr_loss, total_gen_loss, fooling_discr_loss, real_vis_similarity_loss


# %%
# evaluate model
def eval_model(test_dataset, generator, discriminator):

    discr_loss = 0
    total_gen_loss = 0
    fooling_discr_loss = 0
    real_vis_similarity_loss = 0
    n_batches = len(test_dataset)

    # loop over batches
    for n, (ir_batch, real_vis_batch) in test_dataset.enumerate():

        # generate fake VIS images with generator
        fake_vis_batch = generator(ir_batch, training=True)

        # run discriminator once with true VIS image and once with generated VIS image    
        discr_output_real_vis = discriminator([ir_batch, real_vis_batch], training=True)
        discr_output_fake_vis = discriminator([ir_batch, fake_vis_batch], training=True)

        # calculate losses
        discr_loss_batch = _discriminator_loss(discr_output_real_vis, discr_output_fake_vis)
        total_gen_loss_batch, fooling_discr_loss_batch, real_vis_similarity_loss_batch = _generator_loss(discr_output_fake_vis, fake_vis_batch, real_vis_batch)

        # add up batch losses
        discr_loss += discr_loss_batch
        total_gen_loss += total_gen_loss_batch
        fooling_discr_loss += fooling_discr_loss_batch
        real_vis_similarity_loss += real_vis_similarity_loss_batch

    # take average of summed batch losses
    discr_loss /= n_batches
    total_gen_loss /= n_batches
    fooling_discr_loss /= n_batches
    real_vis_similarity_loss /= n_batches

    return discr_loss, total_gen_loss, fooling_discr_loss, real_vis_similarity_loss


def generate_example_images(dataset, gen_model):
    # select random batch of test dataset
    for ir_batch, vis_batch in dataset.take(1):

        # select random image of batch
        rand_idx = random.randint(0, len(ir_batch)-1)
        ir_img = ir_batch.numpy()[rand_idx]
        vis_img = vis_batch.numpy()[rand_idx]

        # predict vis img from ir with generator
        predict_vis_batch = gen_model(ir_batch, training=True) 
        ### Note: The training=True is intentional here since you want the batch statistics,
        ### while running the model on the test dataset. If you use training=False, you get 
        ### the accumulated statistics learned from the training dataset (which you don't want).
        predict_vis_img = predict_vis_batch.numpy()[rand_idx]

        return ir_img, predict_vis_img, vis_img


def pick_image_pair_from_dataset(dataset):
    # select random batch of test dataset
    for ir_batch, vis_batch in dataset.take(1):

        # select random image of batch
        rand_idx = random.randint(0, len(ir_batch)-1)
        ir_img = ir_batch.numpy()[rand_idx]
        vis_img = vis_batch.numpy()[rand_idx]

        return ir_img, vis_img
