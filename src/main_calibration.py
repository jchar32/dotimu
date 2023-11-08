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
"""A script that can be run to perform sensor calibrations for the accelerometer and gyroscope signals from DOT sensors
"""

if __name__ == "__main__":
    import file_io
    import calibration
    import os

    # Specify data path for calibration files
    datapath = "../data/raw/calibrations/"
    filenames = os.listdir("../data/raw/calibrations/")

    dotdata = file_io.load_dot_files(filenames, datapath, ui=False)

    # calculate mean signals for each sensor and orientation collected
    meandotdata = calibration.mean_dot_signals(dotdata)

    # calculate the calibration corrections for each sensor
    dot_calibs = calibration.ori_and_bias(meandotdata)

    # save the calibration corrections to a pickle file
    calib_path = "../data/processed/calibrations/"
    calib_filename = "dot_calibrations.pkl"
    file_io.save_dot_calibrations(dot_calibs, os.path.join(calib_path, calib_filename))

    dotdata_cal = calibration.apply_sensor_correction(dotdata, dot_calibs)
