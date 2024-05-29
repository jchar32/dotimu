from dataclasses import dataclass
import numpy as np
import copy

@dataclass
class Dot:
    data: dict


def find_common_start_time(data: dict) -> dict:

    trial_nums = data[list(data.keys())[0]].data.keys()

    # collect the first time stamp for each trial on each sensor
    test = np.full((len(data.keys()), len(trial_nums)), np.nan)
    for i, trial in enumerate(trial_nums):
        for j, s in enumerate(data.keys()):
            test[j, i] = data[s].data[trial].loc[0, "SampleTimeFine"]

    latest_starttime = np.max(test, axis=0)  # the highest common timestamp

    # get the row index for the common time stamps for every sensor across each trial.
    index_to_trim_start = {s: [] for s in data.keys()}
    for t in range(len(trial_nums)):
        for s in data.keys():
            index_to_trim_start[s].append(np.where(data[s].data[t].loc[:, "SampleTimeFine"] == latest_starttime[t])[0][0])
    return index_to_trim_start


def sync_multi_dot(data, syncidx):
    syncd_data = copy.deepcopy(data)
    for s in syncd_data.keys():
        for t in syncd_data[s].data.keys():
            syncd_data[s].data[t] = syncd_data[s].data[t].iloc[syncidx[s][t]:, :]
            syncd_data[s].data[t].reset_index(inplace=True)
    return syncd_data
