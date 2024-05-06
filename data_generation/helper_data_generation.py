import glob


def get_all_files_in_study_period(msg_path, years, months):
    """gets paths to all files within defined study period
    Args:
        msg_path (str): root path to msg data
        years (list(int)): list of years considered in study
        months (list(int)): list of months considered in study
    Returns:
        list(str): list of all file paths in study period
    """
    msg_files = []
    # loop over given years and months and collect path to all files
    for year in years:
        for month in months:
            msg_path_month = f"{msg_path}/{year}/{month:02}"
            msg_files.extend(glob.glob(msg_path_month + '/*.nc'))

    return msg_files

def datetime_to_string(timestamp):
    """converts np.datetime64 to 

    Args:
        timestamp (_type_): _description_

    Returns:
        _type_: _description_
    """
    return str(timestamp).replace('-', '').replace('T', '_').replace(':', '')[:13]