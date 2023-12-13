import os
import pickle
import time
from typing import Dict, List

import pandas as pd

from dot import Dot
from ui import get_path


def read_from_csv(path):
    df = pd.read_csv(path, header=0)
    return df.drop(index=0).reset_index(drop=True)


def load_dot_files(
    files: List[str] | str = "", datapath: str = "", session_trial_numbers: List[int] = [0]
) -> Dict[int, List]:
    """
    Load dot files and return a dictionary of dot data.

    Args:
        files (List[str] | str): List of file names or a single file name to load.
        datapath (str): Path to the directory containing the dot files.
        session_trial_numbers (List[int]): List of session trial numbers to load.

    Returns:
        Dict[int, List]: A dictionary where the keys are sensor IDs and the values are lists of dataframes for each trial.
    """

    dotdata = {}

    # get all sensor numbers from first element in file name string
    sensorids = sorted(list(set([f.split("_")[0] for f in files])))

    for id in sensorids:
        # find all possible file names
        available_files = sorted([file for file in files if file.split("_")[0] == str(id)])

        # filter for the files that correspond to the session trial numbers
        indexed_files = [available_files[i] for i in [t - 1 for t in session_trial_numbers]]

        # build list of dataframes for each trial
        temp = []
        for trial_file_name in indexed_files:
            data = read_from_csv(os.path.join(datapath, trial_file_name))
            temp.append(data)

        dotdata[int(id)] = temp

    return dotdata


# def load_dot_files(filenames: List[str] | str = "", datapath: str = "", ui: bool = True) -> Dict[int, Dot]:
#     """
#     Load data from .csv files into a dictionary.

#     This function reads data from a list of .csv files, each file corresponding to single dot sensor file.
#     The data from each file is stored in a dictionary where the keys are the dot sensor IDs and the values are Dot objects.

#     Args:
#         ui (bool): If True, a UI will be displayed to select the files to load. Defaults to True.
#         filenames (List[str]): A list of filenames to read data from.
#         datapath (str): The path to the directory where the .csv files are stored.

#     Returns:
#         dotdata (Dict[str, Dot]): A dictionary where the keys are sensor IDs and the values are Dot objects containing.

#     IMPORTANT ASSUMPTIONS:
#     - the file names are prefixed with an integer corresponding to the real-life label of the sensor (what you call it in the Movella App).

#     - The sensor names are suffixed with something that is unique to each instance of a collection. You can rename these if you wish, so long as all the sensors you collected with have the same suffix for each collection.
#     [sensor #]_[whatever you want]_[file id].csv

#         Dot sensors default to:
#         [sensor code]_[sensor id]_[date of collection].csv
#         sensor code = the label you have given it in the Movella app
#         sensor id = the unique id of the sensor
#         date of collection = the date the data was collected
#     """
#     if ui:
#         direc, files = get_path("*.csv")
#     else:
#         direc = datapath
#         files = filenames

#     data_dict = {}
#     dotdata = {}
#     filecounter = 0
#     previous_id = int(files[0].split("_")[0])
#     final_id = int(files[-1].split("_")[0])
#     dotid = [previous_id]

#     for i, file in enumerate(files):
#         temp = read_from_csv(os.path.join(direc, file))
#         id = int(file.split("_")[0])  # get the sensor ID

#         if id != previous_id:
#             dotdata[int(dotid[-1])] = Dot(data_dict)
#             data_dict = {}
#             dotid.append(id)
#             filecounter = 0

#         data_dict[filecounter] = temp
#         filecounter += 1
#         previous_id = id
#         i += 1
#         if i == len(files) and id == final_id:
#             dotdata[int(dotid[-1])] = Dot(data_dict)
#     return dotdata


def save_dot_calibrations(calibration_data: dict, path: str = "dot_calibrations.pkl") -> None:
    with open(path, "wb") as f:
        pickle.dump(calibration_data, f)
    print(f"Calibrations saved to {path}")


def load_dot_calibrations(path: str = "dot_calibrations.pkl") -> Dict[str, pd.DataFrame]:
    with open(path, "rb") as f:
        return pickle.load(f)


def __dot_data_indices():
    """Specify each data channel to export from the DOT sensor.

    Returns:
        XsIntArray: A list of data channels to export from the DOT sensor. Type is native to the C++ API used behind the scenes.
    """
    import movelladot_pc_sdk as sdk  # type: ignore

    exportData = sdk.XsIntArray()  # type: ignore
    exportData.push_back(sdk.RecordingData_Timestamp)  # type: ignore
    exportData.push_back(sdk.RecordingData_Euler)  # type: ignore
    exportData.push_back(sdk.RecordingData_Quaternion)  # type: ignore
    exportData.push_back(sdk.RecordingData_Acceleration)  # type: ignore
    exportData.push_back(sdk.RecordingData_AngularVelocity)  # type: ignore
    exportData.push_back(sdk.RecordingData_MagneticField)  # type: ignore
    exportData.push_back(sdk.RecordingData_Status)  # type: ignore
    return exportData


def dir_exists_or_makeit(path: str) -> None:
    """
    Check if a directory exists at the given path, and if not, create it.

    Args:
        path (str): The path to the directory to check or create.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def export_dot_data(dots2export: List[int] | str = "all", path: str = "data/") -> None:
    # TODO: This is still not functioning as desired -> despite trying to index a given connected dot sensor, the api seems to skip over all the sensors and just moves to close the ports. A version is seperated into git branch: select-dots2export for further development of this.
    """
    Exports data from USB connected Dot devices to CSV files.

    This function initializes a connection to all detected Dot devices, selects the data channels to export,
    and exports the data to CSV files.
    The files are named in the format "dot_{device_id}_{fileIndex}.csv"
    and are saved in a /data/ folder within the current working dir.

    Args:
        dots2export (Union[List[int], str], optional): A list of indices for which Dots to export or "all" to export all.
            Defaults to "all".
        path (str, optional): The path to the directory to save the CSV files to. Defaults to "../data/".
    Returns:
        None
    """
    from xdpchandler import XdpcHandler

    if path != "data/":
        raise NotImplementedError("Only the default path is currently supported.")
    dir_exists_or_makeit(path)

    dotio = XdpcHandler()
    dotio.initialize()
    dotio.detectUsbDevices()

    numDevices = len(dotio.detectedDots())

    print(f"Number of Dots Found: {numDevices}")
    dotio.connectDots()  # connect to all found dot sensors

    exportData = __dot_data_indices()  # get data channels to export

    for deviceIndex in range(numDevices):
        device = dotio.connectedUsbDots()[deviceIndex]  # specifiy which dot to use
        device_id = device.deviceId().toXsString()
        for fileIndex in range(1, 9):
            device.selectExportData(exportData)
            csvFilename = os.path.join(path, f"dot_{device_id}_{fileIndex}.csv")

            device.enableLogging(csvFilename)
            device.startExportRecording(fileIndex)

            # wait until export is done
            while not dotio.exportDone():
                time.sleep(0.1)

            dotio.resetExportDone()  # reset the handler's global export done flag

            device.disableLogging()
    dotio.cleanup()
    print("Done exporting data!")


def set_dot_location_names(locations, dotdata):
    model_data = {}
    for i, j in enumerate(locations.values()):
        loc = list(locations.keys())[list(locations.values()).index(j)]
        model_data[loc] = dotdata[j]
    return model_data


def get_trial_numbers(filepath: str, participant_code: List[str], sheetnamne: str = "sensortrials"):
    """Loads the numbers corresponding to the trials collected with the sensors.

    Args:
        filepath (str): full path to file containing trial numbers
        sheetname (str): name of sheet containing trial numbers
        participant_code (str | List[str]): participant id codes as they appear in the trial number file

    Returns:
        pd.DataFrame: table of trial numbers for each specified participant. rows are: 'pid', 'a_npose', 'a_flean', 'a_rcycle', 'a_lcycle', 'a_walkcal',
       'a_base', 'a_1', 'a_2', 'a_3', 'a_4', 'b_npose', 'b_flean', 'b_rcycle',
       'b_lcycle', 'b_walkcal', 'b_base', 'b_1', 'b_2', 'b_3', 'b_4'.
    """
    trial_numbers = pd.read_excel(filepath, sheet_name=sheetnamne, header=0, index_col=0)

    trial_numbers = trial_numbers.iloc[:, trial_numbers.columns.isin(participant_code)]

    return trial_numbers


if __name__ == "__main__":
    import config

    dotdata_raw = load_dot_files()
    dot_locs = config.sensor_locations
    model_data = set_dot_location_names(dot_locs, dotdata_raw)
