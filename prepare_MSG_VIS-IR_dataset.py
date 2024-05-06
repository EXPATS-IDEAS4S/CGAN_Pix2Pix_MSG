# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import glob

def crop_to_image_size(data_array, image_size):
    n_pix_lat, n_pix_lon = data_array.shape

    # get indices to crop data to given image size
    idx_lat_start = int(n_pix_lat/2) - int(image_size/2)
    idx_lat_end = int(n_pix_lat/2) + int(image_size/2)
    idx_lon_start = int(n_pix_lon/2) - int(image_size/2)
    idx_lon_end = int(n_pix_lon/2) + int(image_size/2)

    return data_array[idx_lat_start: idx_lat_end, idx_lon_start: idx_lon_end]

# %%
def datetime_to_string(timestamp):
    return str(timestamp).replace('-', '').replace('T', '_').replace(':', '')[:13]

def generate_images_from_msgfiles(msg_path, out_path, channel_settings, 
                                  image_size=200, morning="10:00:00", evening="16:00:00"):
    # get all files in given folder
    msg_files = glob.glob(msg_path + '/*.nc')

    # loop over files
    for f in msg_files[:1]:
        # get datetime from filename
        dt_string = os.path.basename(f)[:8]
        day_start = np.datetime64(f"{dt_string[:4]}-{dt_string[4:6]:02}-{dt_string[6:]}T{morning}")
        day_end = np.datetime64(f"{dt_string[:4]}-{dt_string[4:6]:02}-{dt_string[6:]}T{evening}")

        # load dataset
        data_test = xr.load_dataset(f)

        count_test = -1
        # loop over timestamps
        for timestamp in data_test.time.values:

            # check if is in given daytime range
            if timestamp > day_start and timestamp < day_end:
                count_test += 1
                if count_test % 4 == 0:
                    # select data of this timestamp
                    data_daytime = data_test.sel(time=timestamp)

                    # loop over channels that you want to convert into images
                    for ch in channel_settings:

                        # get data from channel and dimensions
                        data_channel = data_daytime[ch].values

                        # crop data to given image size
                        data_channel_crop = crop_to_image_size(data_channel, image_size)
                        

                        # convert to image and save
                        plt.figure(figsize=(image_size/10., image_size/10.), dpi=10)
                        plt.imshow(data_channel_crop[::-1,::-1], # array must be flipped along both axes to fit to imshow input
                                    cmap=channel_settings[ch]['cmap'], 
                                    vmin=channel_settings[ch]['vmin'], 
                                    vmax=channel_settings[ch]['vmax'])
                        plt.axis('off')
                        plt.savefig(f"{out_path}{datetime_to_string(timestamp)}_{ch}_size{image_size}.png", 
                                    bbox_inches='tight', dpi=10)
                        
                        #plt.imsave(fname=f"{out_path}{datetime_to_string(timestamp)}_{ch}_size{image_size}.png", 
                        #        arr=data_channel_crop[::-1,::-1], 
                        #        cmap=channel_settings[ch]['cmap'], 
                        #        vmin=channel_settings[ch]['vmin'], 
                        #        vmax=channel_settings[ch]['vmax'], 
                        #        format='png')



# %%
if __name__ == "__main__":
    year = 2023
    month = 7
    morning = "10:00:00"
    evening = "16:15:00"
    image_size = 128

    # channel names and colormap ranges
    channel_settings = {"VIS006": {"cmap": "gray", 
                                "vmin": 2.60, 
                                "vmax": 64.24},
                        "IR_108": {"cmap": "gray_r", 
                                "vmin": 222.08,
                                "vmax": 303.43}}

#channel:  VIS006
#[0, 0.001, 0.01, 0.99, 0.999, 1]
#[ 1.29997873  2.26788759  2.59995699 64.24230194 73.6897049  95.28556061]
#channel:  IR_108
#[0, 0.001, 0.01, 0.99, 0.999, 1]
#[201.64987183 213.68383789 222.08338928 303.43145752 306.58432007 313.42861938]

    msg_path = f"/net/merisi/pbigalke/teaching/METFUT2024/msg_test_data/{year}/{month:02}/"
    project_path = "/net/merisi/pbigalke/teaching/METFUT2024/GANS/Pix2Pix/"
    output_path = f"{project_path}test_data/{year}/{month:02}_imshow/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # generate images from msg files
    generate_images_from_msgfiles(msg_path, output_path, channel_settings, 
                                  image_size=image_size, morning=morning, evening=evening)

# %%
