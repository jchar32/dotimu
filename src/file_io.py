import pandas as pd
import os
from dot import Dot


def read_from_csv(path):
    df = pd.read_csv(path, header=0)
    return df.drop(index=0).reset_index(drop=True)


def read_dot_files(filenames, datapath):
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
