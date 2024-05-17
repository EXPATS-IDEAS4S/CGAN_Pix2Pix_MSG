# %%
import numpy as np
import glob
import random

# %%
# write code for sorting images into training and testing folder etc.
def sort_images_into_datasets(data_path, train_ratio=0.8):

    VIS_path = f"{data_path}/VIS006" 
    IR_path = f"{data_path}/IR_108" 

    # read in all IR images
    IR_files = sorted(glob.glob(f"{IR_path}/*.png"))

    # random select training images
    random.shuffle(IR_files)
    split_idx = int(train_ratio*len(IR_files))
    train_IR_files = IR_files[:split_idx]
    val_IR_files = IR_files[split_idx:]

    # sort corresponding VIS image
    VIS_files = sorted(glob.glob(f"{VIS_path}/*.png"))






# %%
# path to project directory
PROJECT_DIR = "/net/merisi/pbigalke/teaching/METFUT2024/CGAN_Pix2Pix_MSG"
data_path = f"{PROJECT_DIR}/test_data"

sort_images_into_datasets(data_path)
# %%
