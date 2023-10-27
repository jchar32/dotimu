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
import file_io
import calibration
import os
import pandas as pd
from dot import Dot
import numpy as np

# Specify data path for calibration files
datapath = "../data/raw/calibrations/dot11_to_15/"
filenames = os.listdir("../data/raw/calibrations/dot11_to_15/")

dotdata = file_io.read_dot_files(filenames, datapath)

meandotdata = calibration.mean_data(dotdata)

dot_calibs = calibration.get_calibration(meandotdata)
