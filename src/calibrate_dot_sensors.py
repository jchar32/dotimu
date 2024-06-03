# DOT axes:
# when looking at front of senor (orange side) and reading Xsens logo.
# Z+ is out of the front
# X+ is from bottom to top pointing up
# Y+ is from right to left, pointing left

# <-y ^x .z

# file 1 = bias (~ 5 sec)
# file 2 = x up
# file 3 = x down
# file 4 = y up
# file 5 = y down
# file 6 = z up
# file 7 = z down
# file 8 = mag rotations
"""A script that can be run to perform sensor calibrations for the accelerometer and gyroscope signals from DOT sensors"""

# %%
if __name__ == "__main__":
    import file_io
    import calibration
    import os
    import numpy as np

    # Specify data path for calibration files
    datapath = "../data/raw/calibrations/"
    filenames = os.listdir("../data/raw/calibrations/")
    filenames = sorted(filenames, key=lambda x: int(x.split("_")[0]))

    dotdata = file_io.load_dot_files(filenames, datapath, [1, 2, 3, 4, 5, 6, 7, 8])
    # dotdata = file_io.load_dot_files(filenames, datapath, [1, 2, 3, 4, 5, 6, 7, 8])

    # calculate mean signals for each sensor and orientation collected
    meandotdata = calibration.mean_dot_signals(dotdata)

    # calculate the calibration corrections for each sensor
    dot_calibs = calibration.ori_and_bias(meandotdata)

    # save the calibration corrections to a pickle file
    calib_path = "../data/processed/calibrations/"
    calib_filename = "dot_calibrations.pkl"
    file_io.save_dot_calibrations(dot_calibs, os.path.join(calib_path, calib_filename))

    dotdata_cal = calibration.apply_sensor_correction(dotdata, dot_calibs)

    for id in dot_calibs.keys():
        cal = np.vstack(
            [dot_calibs[id].matrix, dot_calibs[id].accel_bias, dot_calibs[id].gyro_bias]
        )
        np.savetxt(
            os.path.join(calib_path, f"{id}_calibration.csv"), cal, delimiter=","
        )
