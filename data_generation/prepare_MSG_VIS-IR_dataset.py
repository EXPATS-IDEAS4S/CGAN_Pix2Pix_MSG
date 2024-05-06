# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import glob
from pbigalke.teaching.METFUT2024.CGAN_Pix2Pix_MSG.data_generation.helper_data_generation import get_all_files_in_study_period

def crop_to_image_size(data_array, image_size):
    n_pix_lat, n_pix_lon = data_array.shape

    # get indices to crop data to given image size
    idx_lat_start = int(n_pix_lat/2) - int(image_size/2)
    idx_lat_end = int(n_pix_lat/2) + int(image_size/2)
    idx_lon_start = int(n_pix_lon/2) - int(image_size/2)
    idx_lon_end = int(n_pix_lon/2) + int(image_size/2)

    return data_array[idx_lat_start: idx_lat_end, idx_lon_start: idx_lon_end]

def datetime_to_string(timestamp):
    """converts np.datetime64 to 

    Args:
        timestamp (_type_): _description_

    Returns:
        _type_: _description_
    """
    return str(timestamp).replace('-', '').replace('T', '_').replace(':', '')[:13]

def generate_images_from_msgfiles(msg_path, out_path, channel_settings, 
                                  image_size=200, morning="08:00:00", evening="18:00:00"):
    # get all files in given folder
    msg_files = glob.glob(msg_path + '/*.nc')

    # loop over files
    for f in msg_files:
        # get datetime from filename
        dt_string = os.path.basename(f)[:8]
        day_start = np.datetime64(f"{dt_string[:4]}-{dt_string[4:6]:02}-{dt_string[6:]}T{morning}")
        day_end = np.datetime64(f"{dt_string[:4]}-{dt_string[4:6]:02}-{dt_string[6:]}T{evening}")

        # load dataset
        data_test = xr.load_dataset(f)

        # loop over timestamps
        for timestamp in data_test.time.values:

            # check if is in given daytime range
            if timestamp > day_start and timestamp < day_end:
                
                # select data of this timestamp
                data_daytime = data_test.sel(time=timestamp)

                # loop over channels that you want to convert into images
                for ch in channel_settings:

                    # get data from channel and dimensions
                    data_channel = data_daytime[ch].values

                    # crop data to given image size
                    data_channel_crop = crop_to_image_size(data_channel, image_size)
                    
                    # convert to image and save
                    plt.imshow(data_channel_crop[::-1,::-1], # array must be flipped along both axes to fit to imshow input
                            cmap=channel_settings[ch]['cmap'], 
                            vmin=channel_settings[ch]['vmin'], 
                            vmax=channel_settings[ch]['vmax'])
                    plt.axis('off')
                    plt.savefig(f"{out_path}{datetime_to_string(timestamp)}_{ch}.png", bbox_inches='tight')


# %%
if __name__ == "__main__":
    year = 2023
    month = 7
    morning = "08:00:00"
    evening = "18:00:00"
    image_size = 200

    #TODO: define min an max reflectance/brightness temperatures to calibrate colormaps of all images
    channel_settings = {"VIS006": {"cmap": "gray", 
                                "vmin": 0.540,  # use minimum of dataset or some quantile?
                                "vmax": 95.286},  # max of dataset: 93.858  # 0.999 quantile: 83.018
                        "IR_108": {"cmap": "gray_r", 
                                "vmin": 212.531,  # min of dataset: 212.531, # 0.001 quantile: 216.636
                                "vmax": 302.123}}  # max of dataset: 302.123  # 0.999 quantile: 297.985

    msg_path = f"/net/merisi/pbigalke/teaching/METFUT2024/msg_test_data/{year}/{month:02}/"
    out_path = f"/net/merisi/pbigalke/teaching/METFUT2024/GANS/Pix2Pix/test_data/{year}/{month:02}_same_range/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # generate images from msg files
    generate_images_from_msgfiles(msg_path, out_path, channel_settings, 
                                  image_size=image_size, morning=morning, evening=evening)

# %%
