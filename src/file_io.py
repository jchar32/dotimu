import pandas as pd
import os
import time
from dot import Dot
from typing import List, Dict

def read_from_csv(path):
    df = pd.read_csv(path, header=0)
    return df.drop(index=0).reset_index(drop=True)


def load_dot_file(filenames: str, datapath: str) -> Dict[str, Dot]:
    """
    Load data from .csv files into a dictionary.

    This function reads data from a list of .csv files, each file corresponding to single dot sensor file.
    The data from each file is stored in a dictionary where the keys are the dot sensor IDs and the values are Dot objects.

    Args:
        filenames (List[str]): A list of filenames to read data from.
        datapath (str): The path to the directory where the .csv files are stored.

    Returns:
        dotdata (Dict[str, Dot]): A dictionary where the keys are sensor IDs and the values are Dot objects containing.
    """
    data_dict = {}
    dotdata = {}
    filenum = 0
    last_id = filenames[0].split("_")[0]
    dotid = [last_id]

    for i, file in enumerate(filenames):
        temp = read_from_csv(os.path.join(datapath, file))
        id = file.split("_")[0]  # get the sensor ID

        if id != last_id:
            dotdata[dotid[-1]] = Dot(data_dict)
            data_dict = {}
            dotid.append(id)
            filenum = 0
        if i == len(filenames) - 1:
            data_dict[f"{filenum}"] = temp
            dotdata[dotid[-1]] = Dot(data_dict)

        data_dict[f"{filenum}"] = temp
        filenum += 1
    return dotdata


def __dot_data_indices():
    """Specify each data channel to export from the DOT sensor.

    Returns:
        XsIntArray: A list of data channels to export from the DOT sensor. Type is native to the C++ API used behind the scenes.
    """
    import movelladot_pc_sdk as sdk
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
            csvFilename = (os.path.join(path, f"dot_{device_id}_{fileIndex}.csv"))

            device.enableLogging(csvFilename)
            device.startExportRecording(fileIndex)

            # wait until export is done
            while not dotio.exportDone():
                time.sleep(0.1)

            dotio.resetExportDone()  # reset the handler's global export done flag

            device.disableLogging()
    dotio.cleanup()
    print("Done exporting data!")
