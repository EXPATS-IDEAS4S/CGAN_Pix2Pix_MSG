# %%
import os
import numpy as np
import xarray as xr
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl

# questions to discuss with Claudia:
#   - which preprocessing was done in ncdf files 2023-07? 
#   - where to put the data for the student?

# %%
def get_quantiles_of_dataset(data_path, output_path, channel, quantiles, morning="08:00:00", evening="18:00:00"):
    # avoid errors when only one quantile is given
    if not isinstance(quantiles, list):
        quantiles = list(quantiles)

    # store values in this array
    all_values = None

    # store quantiles here
    quantile_values = []

    # define colors for quantiles to draw later into histogram
    cmap = mpl.colormaps['plasma']
    colors = cmap(np.linspace(0, 1, len(quantiles)))

    # get all files in given folder
    msg_files = glob.glob(data_path + '/*.nc')

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

            if timestamp > day_start and timestamp < day_end:
                # select data of this timestamp
                # check if is in given daytime range
                data_daytime = data_test.sel(time=timestamp)[channel].values.flatten()
                if all_values is None:
                    all_values = data_daytime[np.newaxis, :]
                else:
                    all_values = np.concatenate((all_values, data_daytime[np.newaxis, :]), axis=0)
    
    # calculate quantiles:
    quantile_values = np.quantile(all_values, quantiles)

    # plot distribution including quantiles
    plt.hist(all_values.flatten(), bins=100, histtype="stepfilled")
    for q, quant in enumerate(quantile_values):
        plt.axvline(quant, label=f"{quantiles[q]:.3f} quantile = {quant:.2f}", color=colors[q])
    plt.ylabel("occurence N")
    plt.xlabel("reflectance" if "VIS" in channel else "brightness temp [K]")
    plt.legend(loc="upper center")
    plt.savefig(f"{output_path}value_distribution_{channel}.png", bbox_inches="tight")
    plt.close()
        
    return quantile_values
"""
                # minimum
                min_tmp = float(data_daytime[channel].min(skipna=True).values)
                min_value = min_tmp if min_tmp < min_value else min_value
                quant_0001_tmp = float(data_daytime[channel].quantile(0.001, skipna=True).values)
                quant_0001 = quant_0001_tmp if quant_0001_tmp < quant_0001 else quant_0001

                # maximum
                max_tmp = float(data_daytime[channel].max(skipna=True).values)
                max_value = max_tmp if max_tmp > max_value else max_value
                quant_0999_tmp = float(data_daytime[channel].quantile(0.999, skipna=True).values)
                quant_0999 = quant_0999_tmp if quant_0999_tmp > quant_0999 else quant_0999

    return min_value, quant_0001, max_value, quant_0999
"""            

# %%
if __name__ == "__main__":
    year = 2023
    month = 7
    morning = "10:00:00"
    evening = "16:00:00"

    msg_path = f"/net/merisi/pbigalke/teaching/METFUT2024/msg_test_data/{year}/{month:02}/"
    project_path = "/net/merisi/pbigalke/teaching/METFUT2024/GANS/Pix2Pix/"
    output_path = f"{project_path}output/preprocessing_dataset/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    channels = ["VIS006", "VIS008", "IR_108"]
    quantiles = [0, 0.001, 0.01, 0.99, 0.999, 1]
    for ch in channels:
        print("channel: ", ch)
        print(quantiles)
        print(get_quantiles_of_dataset(msg_path, output_path, ch, quantiles, morning=morning, evening=evening))


# %%
