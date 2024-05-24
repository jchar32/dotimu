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
    files: List[str] | str = "",
    datapath: str = "",
    session_trial_numbers: List[int] = [0],
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
        available_files = sorted(
            [file for file in files if file.split("_")[0] == str(id)]
        )

        # filter for the files that correspond to the session trial numbers
        indexed_files = [
            available_files[i] for i in [t - 1 for t in session_trial_numbers]
        ]

        # build list of dataframes for each trial
        temp = []
        for trial_file_name in indexed_files:
            data = read_from_csv(os.path.join(datapath, trial_file_name))
            temp.append(data)

        dotdata[int(id)] = temp

    return dotdata


def save_dot_calibrations(
    calibration_data: dict, path: str = "dot_calibrations.pkl"
) -> None:
    with open(path, "wb") as f:
        pickle.dump(calibration_data, f)
    print(f"Calibrations saved to {path}")


def load_dot_calibrations(
    path: str = "dot_calibrations.pkl",
) -> Dict[str, pd.DataFrame]:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle_file(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")


def load_pickle_file(filename):
    with open(filename, "rb") as f:
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


def get_trial_numbers(
    filepath: str, participant_code: List[str], sheetname: str = "sensortrials"
):
    """Loads the numbers corresponding to the trials collected with the sensors.

    Args:
        filepath (str): full path to file containing trial numbers
        sheetname (str): name of sheet containing trial numbers
        participant_code (str | List[str]): participant id codes as they appear in the trial number file

    Returns:
        pd.DataFrame: table of trial numbers for each specified participant. rows are based on trial names in the csv or xlsx file. Columns are the participant codes.
    """
    trial_numbers = pd.read_excel(filepath, sheet_name=sheetname, header=0, index_col=0)
    participant_trial_numbers = trial_numbers.iloc[
        :, trial_numbers.columns == int(participant_code[0])
    ]

    return participant_trial_numbers


if __name__ == "__main__":
    import config

    dotdata_raw = load_dot_files()
    dot_locs = config.sensor_locations
    model_data = set_dot_location_names(dot_locs, dotdata_raw)
