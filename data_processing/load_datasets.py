# %%
import tensorflow as tf
import glob

IMAGE_SIZE = 128

# %%
def _get_paired_IR_VIS_file_lists(path):
    # get all ir image files from path
    ir_image_list = glob.glob(f"{path}/*IR_108.png")
    # get corresponding vis image files
    vis_image_list = [ir.replace("IR_108", "VIS006") for ir in ir_image_list]
    # return matched lists
    return ir_image_list, vis_image_list

def _load_single_image(image_file):
    # read in image
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image, channels=1)
    return image

def _resize(ir_image, vis_image):
    # ensures that all images have the correct size
    ir_image = tf.image.resize(ir_image, [IMAGE_SIZE, IMAGE_SIZE], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    vis_image = tf.image.resize(vis_image, [IMAGE_SIZE, IMAGE_SIZE], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ir_image, vis_image

def _normalize(ir_image, vis_image):
    # normalize the images to range [-1..1]
    ir_image = (tf.cast(ir_image, tf.float32) / 127.5) - 1.0 
    vis_image = (tf.cast(vis_image, tf.float32) / 127.5) - 1.0 
    return ir_image, vis_image

def _random_jitter(ir_image, vis_image):
    # randomly flip the images left-right
    if tf.random.uniform(()) > 0.5:
        ir_image = tf.image.flip_left_right(ir_image)
        vis_image = tf.image.flip_left_right(vis_image)
    return ir_image, vis_image

# %%
def _preprocess_train_images(ir_image_file, vis_image_file):

    # load images as tf.tensor
    ir_image = _load_single_image(ir_image_file)
    vis_image = _load_single_image(vis_image_file)

    # do some preprocessing
    ir_image, vis_image = _resize(ir_image, vis_image)
    ir_image, vis_image = _random_jitter(ir_image, vis_image)
    ir_image, vis_image = _normalize(ir_image, vis_image)

    return ir_image, vis_image

def load_train_dataset(train_path, batch_size, buffer_size=100):

    # get all ir and vis images from path, sorted into paired lists
    train_ir_images, train_vis_images = _get_paired_IR_VIS_file_lists(train_path)

    # create tensorflow dataset object from list of images 
    train_dataset = tf.data.Dataset.from_tensor_slices((train_ir_images, train_vis_images))

    # preprocess images. The map function of tensorflow calls _preprocess_train_images on each element
    train_dataset = train_dataset.map(_preprocess_train_images)

    # organize dataset into batches after shuffling
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)


    return train_dataset

# %%
def _preprocess_test_images(ir_image_file, vis_image_file):

    # load images as tf.tensor
    ir_image = _load_single_image(ir_image_file)
    vis_image = _load_single_image(vis_image_file)

    # do some preprocessing
    ir_image, vis_image = _resize(ir_image, vis_image)
    ir_image, vis_image = _normalize(ir_image, vis_image)
    
    return ir_image, vis_image

def load_test_dataset(test_path):

    # get all ir and vis images from path, sorted into paired lists
    test_ir_images, test_vis_images = _get_paired_IR_VIS_file_lists(test_path)

    # create dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_ir_images, test_vis_images))

    # preprocess images
    test_dataset = test_dataset.map(_preprocess_test_images)

    # organize dataset into batches
    test_dataset = test_dataset.batch(len(test_ir_images))

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
