# %%
import tensorflow as tf
import glob
import numpy as np
from preprocess_methods import normalize, resize, random_jitter

# %%
def get_paired_IR_VIS_file_lists(path):
    # get all ir images from path
    ir_images = sorted(glob.glob(f"{path}/*IR_108.png"))

    # get corresponding vis images
    vis_images = [ir.replace("IR_108", "VIS006") for ir in ir_images]

    return ir_images, vis_images

def load_single_image(image_file):
    # read in image
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, channels=1)
    return image

# %%
def preprocess_train_images(ir_image_file, vis_image_file):

    # load images as tf.tensor
    ir_image = load_single_image(ir_image_file)
    vis_image = load_single_image(vis_image_file)

    ##### do some preprocessing #####
    ir_image, vis_image = resize(ir_image, vis_image)
    ir_image, vis_image = random_jitter(ir_image, vis_image)
    ir_image, vis_image = normalize(ir_image, vis_image)

    return ir_image, vis_image

def load_train_dataset(train_path, batch_size):

    # get all ir and vis images from path, sorted into paired lists
    train_ir_images, train_vis_images = get_paired_IR_VIS_file_lists(train_path)

    # create dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_ir_images, train_vis_images))

    # preprocess images
    train_dataset = train_dataset.map(preprocess_train_images)

    # organize dataset into batches
    train_dataset = train_dataset.shuffle(10).batch(batch_size)

    return train_dataset

# %%
def preprocess_test_images(ir_image_file, vis_image_file):

    # load images as tf.tensor
    ir_image = load_single_image(ir_image_file)
    vis_image = load_single_image(vis_image_file)

    ##### do some preprocessing #####
    ir_image, vis_image = resize(ir_image, vis_image)
    ir_image, vis_image = normalize(ir_image, vis_image)
    
    return ir_image, vis_image

def load_test_dataset(test_path, batch_size):

    # get all ir and vis images from path, sorted into paired lists
    test_ir_images, test_vis_images = get_paired_IR_VIS_file_lists(test_path)

    # create dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_ir_images, test_vis_images))

    # preprocess images
    test_dataset = test_dataset.map(preprocess_test_images)

    # organize dataset into batches
    test_dataset = test_dataset.batch(batch_size)

    return test_dataset

# %%
if __name__ == "__main__":
    # path to project directory
    PROJECT_DIR = "/net/merisi/pbigalke/teaching/METFUT2024/CGAN_Pix2Pix_MSG"

    # load example image
    IMAGE_PATH = f"{PROJECT_DIR}/VIS_IR_images"
    BATCH_SIZE = 10

    # get all training data
    train_path = f"{IMAGE_PATH}/train"
    train_dataset = load_train_dataset(train_path, BATCH_SIZE)
    #print(train_dataset)
    #print(len(list(train_dataset)))

    # get all test data
    #test_path = f"{IMAGE_PATH}/val"
    #test_dataset = load_test_dataset(test_path, BATCH_SIZE)
    #print(test_dataset)
    #print(len(list(test_dataset)))



# %%
